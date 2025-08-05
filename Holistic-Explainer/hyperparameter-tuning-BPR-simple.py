import optuna
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import  AutoTokenizer
# LOGIN CREDENTIALS: HuggingFACE:
import json
import random
import numpy as np
import torch._dynamo
# torch._dynamo.config.cache_size_limit = 64  # Increase as needed
torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from train_dataset_BPR import get_BPRdataset_loader
from data_utils import load_pickle, load_json, readTargetItem
from trainer_BPR_simple import train_custom_single
from utils import seedSet
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for holistic explainer script")
    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    
    parser.add_argument('--epochs',type=int,default=5,help='Number of Training Epochs')
    parser.add_argument("--batch_size",type=int,default=16,help='Batch Size for training')
   
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument('--debug',action='store_true',help='Debug or not?')
    
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')
    parser.add_argument('--stage',type=int,default=0,help='Stage Setting')
    parser.add_argument("--checkpoint",type=str,default=None,help='Checkpoint path to load')

    args = parser.parse_args()
    return args


def save_all_trials(study, trial):
    """Saves all trial results in a single JSON file."""
    trials_data = [
        {
            "trial_number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state),
            "datetime_start": str(t.datetime_start),
            "datetime_end": str(t.datetime_complete),
        }
        for t in study.trials
    ]

    # Save as JSON
    with open(f"{SAVE_DIR}/optuna_all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=4)

    print(f"Saved all trials to optuna_all_trials.json")
    print('='*50)


def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    torch.cuda.empty_cache()
    
    # variables: id_embed_dim ; hidden_dim (4x,2x,x) ; dropout; learning_rate ; weight_decay ; warmup_ratio ; 
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    id_embed_dim = trial.suggest_categorical("id_embed_dim", [64,128,256])
    warmup_ratio = trial.suggest_uniform("warmup_ratio", 0.01, 0.1)
    dropout = trial.suggest_uniform("dropout", 0.05, 0.3)
    
    hidden_dim = [id_embed_dim * 4, id_embed_dim * 2, id_embed_dim]
    global train_loader, val_loader, epochs, batch_size, num_batches_train, num_batches_val, local_rank, clip_grad_norm, group_num, USER_NUM, ITEM_NUM, gradient_accumulation_steps, stage, checkpoint, dataset, debug
    
    output_dir = f'snap/{dataset}-hp/stage-{stage}'
    output_dir += f'/trial_{trial.number}'
    print("="*50)
    print(f"TRIAL: trial_{trial.number}")
    train_custom_single(seed=seed,
                        early_stopping_patience=3,
                        gpu = local_rank,
                        num_batches_train = num_batches_train,
                        num_batches_val = num_batches_val,
                        gradient_accumulation_steps = gradient_accumulation_steps,
                        num_epochs = epochs,
                        clip_grad_norm = clip_grad_norm,
                        train_loader=train_loader, 
                        val_loader = val_loader,
                        user_num = USER_NUM + 1,
                        item_num = ITEM_NUM + 1,
                        output_dir = output_dir,
                        id_embed_dim = id_embed_dim,
                        hidden_dim = hidden_dim,
                        warmup_ratio = warmup_ratio,
                        learning_rate = learning_rate,
                        weight_decay = weight_decay,
                        dropout = dropout,
                        dataset = dataset,
                        stage=stage,
                        checkpoint=checkpoint,
                        debug=debug
                       )
    torch.cuda.empty_cache()

    # Retrieve the best validation loss for tuning
    file_dir = f'snap/{dataset}-hp/stage-{stage}'
    file_dir += f'/trial_{trial.number}/best_val_loss.pt'
    best_val_loss = torch.load(file_dir)
    return best_val_loss

def run_study(n_trials=10):
    
    study = optuna.create_study(
            study_name=f"stage_tuning",
            direction="minimize",
            load_if_exists=True
        )
        
    study.optimize(objective, n_trials = n_trials, callbacks=[save_all_trials])
    return
    


# Initialize global DataLoaders (to avoid reloading for each trial)
train_loader = None
val_loader = None

def initialize_dataloaders(dataset, data_path, batch_size, local_rank):
    """Creates and caches the global DataLoaders to avoid memory issues."""
    global train_loader, val_loader, USER_NUM, ITEM_NUM
    
    datamaps = load_json(os.path.join(data_path, dataset, 'datamaps.json'))
    
    item2id = datamaps['item2id']
    
    USER_NUM = len(datamaps['user2id'])
    ITEM_NUM = len(item2id)
    
    
    train_loader, train_dataset = get_BPRdataset_loader(dataset = dataset, data_path = data_path, mode='train',batch_size=batch_size, workers=0, distributed=False,shuffle=True)
    
    val_loader, val_dataset = get_BPRdataset_loader(dataset = dataset, data_path = data_path, mode='val',batch_size=batch_size, workers=0, distributed=False,shuffle=True)
    
    USER_NUM = max(train_dataset.user_num, val_dataset.user_num)
    ITEM_NUM = max(train_dataset.item_num, val_dataset.item_num)
    
    print("#training samples: ",len(train_dataset))
    print("#validation samples: ",len(val_dataset))
    
    print("#training batches: ",len(train_loader))
    print("#validation batches: ",len(val_loader))
    return

    

if __name__ == "__main__":
    args = parse_args()
    print("Args: ",args)
    
    stage = args.stage
    
    dataset = args.dataset
    data_path = args.data_path
    batch_size = args.batch_size
    seed = args.seed
    clip_grad_norm = args.clip_grad_norm
    gradient_accumulation_steps = args.gradient_accumulation_steps
    checkpoint = args.checkpoint
    
    # Limit training to a fixed number of batches per trial for efficiency
    num_batches_train = args.num_batches_train  # 951 for yelp ; 1230 for clothing  ; 600 for toys ; 700 for beauty (10% of train data)
    num_batches_val = args.num_batches_val # 190 for yelp ; 246 for clothing ; 120 for toys ; 140 for beauty (5% of val data)
    
    debug = args.debug
    epochs = args.epochs
    
    local_rank = args.local_rank 
    
    SAVE_DIR = f"optuna_results/{dataset}"  # Directory to store trial JSON files
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    seedSet(seed)
    
    
    # Initialize global DataLoaders (only once)
    initialize_dataloaders(
        dataset=dataset,
        data_path=data_path,
        batch_size=batch_size,
        local_rank=local_rank,
    )
    
    if debug:
        trials= 1
    else:
        trials = 20
    
    run_study(n_trials=trials)
    