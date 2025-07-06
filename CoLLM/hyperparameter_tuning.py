import optuna
import os
import torch
import torch.distributed as dist
from trainer import train_custom_single
from data_loading import get_dataset_loader

import json
import random
import numpy as np
import torch._dynamo
# torch._dynamo.config.cache_size_limit = 64  # Increase as needed
torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import seedSet
import argparse

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



# Initialize global DataLoaders (to avoid reloading for each trial)
train_loader = None
val_loader = None
train_dataset = None
def initialize_dataloaders(dataset, data_path, batch_size, local_rank):
    """Creates and caches the global DataLoaders to avoid memory issues."""
    global train_loader, val_loader, USER_NUM, ITEM_NUM, train_dataset, group_num

    if train_loader is None or val_loader is None:
        train_sample_number, val_sample_number = 5, 2
        
        train_loader, train_dataset = get_dataset_loader(train_sample_number, mode='train', dataset=dataset,data_path=data_path, batch_size=batch_size, num_groups=group_num, workers=0, distributed=False,local_rank=local_rank,shuffle=True)
        
        val_loader, val_dataset = get_dataset_loader(val_sample_number,mode='val', dataset=dataset, data_path=data_path, batch_size=batch_size, workers=0, distributed=False,local_rank=local_rank,shuffle=True)
        
        USER_NUM, ITEM_NUM = max(train_dataset.user_num,val_dataset.user_num) + 1, max(train_dataset.item_num,val_dataset.item_num) + 1


def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    torch.cuda.empty_cache()

    # Define hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    rec_dim = trial.suggest_categorical("rec_dim", [64,128,256])
    warmup_ratio = trial.suggest_uniform("warmup_ratio", 0.01, 0.1)

    global train_loader, val_loader, epochs, batch_size, num_batches_train, num_batches_val, local_rank, clip_grad_norm, maxlen, fair_reweight, group_num, lora_alpha, lora_dropout, lora_r, llama_ckpt, USER_NUM, ITEM_NUM, prompt_path, project_mid, gradient_accumulation_steps, train_dataset
    
    output_dir = f'snap/{dataset}-hp'
    if fair_reweight:
        output_dir += '-FAIR-IPS'
        
    output_dir += f'/trial_{trial.number}'
    
    print("="*50)
    print(f"TRIAL: trial_{trial.number}")
    train_custom_single(seed=seed,
                        early_stopping_patience=3,
                        gpu = local_rank,
                        rec_dim = rec_dim,
                        prompt_path = prompt_path,
                        output_dir = output_dir,
                        maxlen = maxlen,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout = lora_dropout,
                        project_mid=project_mid,
                        llama_ckpt = llama_ckpt,
                        rec_ckpt = None,
                        num_batches_train = num_batches_train,
                        num_batches_val = num_batches_val,
                        gradient_accumulation_steps = gradient_accumulation_steps,
                        num_epochs = epochs,
                        warmup_ratio = warmup_ratio,
                        learning_rate = learning_rate,
                        weight_decay = weight_decay,
                        clip_grad_norm = clip_grad_norm,
                        train_loader=train_loader, 
                        val_loader = val_loader,
                        user_num = USER_NUM,
                        item_num = ITEM_NUM,
                        fair_reweight = fair_reweight,
                        group_num = group_num,
                        train_dataset = train_dataset,       
                       )
    torch.cuda.empty_cache()

    # Retrieve the best validation loss for tuning
    file_dir = f'snap/{dataset}-hp'
    if fair_reweight:
        file_dir += '-FAIR-IPS'
    
    file_dir += f'/trial_{trial.number}/best_val_loss.pt'
    
    best_val_loss = torch.load(file_dir)
    return best_val_loss


def run_study(n_trials=10):
    
    study = optuna.create_study(
            study_name="llama_hp_tuning",
            direction="minimize",
            load_if_exists=True
        )
        
    study.optimize(objective, n_trials = n_trials, callbacks=[save_all_trials])
    



def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    
    # Adding arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--maxlen', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    
    parser.add_argument("--lora_r",type=int,default=8,help="Lora R Value")
    parser.add_argument("--lora_alpha",type=int,default=16,help="Lora Alpha Value")
    parser.add_argument('--lora_dropout', type=float, default=0.2, help='Lora Dropout')
    parser.add_argument("--project_mid",type=int, default=10,help='Scale of the MLP Projection Layer in terms of how much than the recommender embedding size')
    
    parser.add_argument("--llama_ckpt", type=str, default=None, help="Checkpoint to load LLaMa model weights alone: give just the directory")
    parser.add_argument("--prompt_path",type=str, default="prompts/collm_amazon.txt",help='Path to load the prompt styles: e.g.: "prompts/collm_amazon.txt"')
        
    parser.add_argument("--fair_reweight", action='store_true', help='Flag to train with reweighting the input data while finetuning')
    parser.add_argument('--group_num', type=int, default=5, help='Number of item groups for popularity fairness (default is 5 as per IFairLRS paper)')
    
    parser.add_argument("--debug", action='store_true', help='Flag to Debug')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    
    args = parse_args()
    print("Args: ",args)
    
    dataset = args.dataset
    data_path = args.data_path
    batch_size = args.batch_size
    seed = args.seed
    fair_reweight = args.fair_reweight
    group_num = args.group_num
    
    clip_grad_norm = args.clip_grad_norm
    gradient_accumulation_steps = args.gradient_accumulation_steps
    maxlen = args.maxlen
    
    SAVE_DIR = f"optuna_results/{dataset}"  # Directory to store trial JSON files
    if fair_reweight:
        SAVE_DIR += '-FAIR-IPS'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    
    # Limit training to a fixed number of batches per trial for efficiency
    num_batches_train = args.num_batches_train  # 951 for yelp ; 1230 for clothing  ; 600 for toys ; 700 for beauty (10% of train data)
    num_batches_val = args.num_batches_val # 190 for yelp ; 246 for clothing ; 120 for toys ; 140 for beauty (5% of val data)
    n_trials=20
    
    if args.debug:
        num_batches_train = 100
        num_batches_val = 100
        n_trials=2
    
    epochs = args.num_epochs
    lora_alpha = args.lora_alpha
    lora_r = args.lora_r
    lora_dropout = args.lora_dropout
    project_mid = args.project_mid
    prompt_path = args.prompt_path
    llama_ckpt = args.llama_ckpt
        
    local_rank = args.local_rank  # Alternate between GPU 0 and GPU 1 # int(os.environ["LOCAL_RANK"])
    
    seedSet(seed)
    # Initialize global DataLoaders (only once)
    initialize_dataloaders(
        dataset=dataset,
        data_path=data_path,
        batch_size=batch_size,
        local_rank=local_rank,
    )
    run_study(n_trials=n_trials)

