import optuna
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from trainer import train_custom_single
from data_loading import get_dataset_loader
from concurrent.futures import ProcessPoolExecutor
from transformers import  AutoTokenizer  # noqa: F402

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

def initialize_dataloaders(tokenizer, dataset, data_path, batch_size, distributed, local_rank,train_on_inputs,maxlen):
    """Creates and caches the global DataLoaders to avoid memory issues."""
    global train_loader, val_loader,train_dataset

    if train_loader is None or val_loader is None:
        train_sample_number, val_sample_number = 5, 2
        train_loader, train_dataset = get_dataset_loader(
            tokenizer, train_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,
            mode='train', dataset=dataset, data_path=data_path, batch_size=batch_size,
            workers=0, distributed=distributed, local_rank=local_rank,shuffle=True
        )

        val_loader, _ = get_dataset_loader(
            tokenizer, val_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,
            mode='val', dataset=dataset, data_path=data_path, batch_size=batch_size,
            workers=0, distributed=distributed, local_rank=local_rank,shuffle=True
        )

def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    torch.cuda.empty_cache()

    # Define hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    lora_dropout = trial.suggest_uniform("lora_dropout", 0.05, 0.3)
    warmup_ratio = trial.suggest_uniform("warmup_ratio", 0.01, 0.1)

    global dataset,data_path, base_model, train_loader, val_loader, tokenizer, epochs, batch_size, num_batches_train, num_batches_val, local_rank, clip_grad_norm, maxlen, train_on_inputs, fair_reweight, group_num, train_dataset
    
    output_dir = f'snap/{dataset}-hp'
    if fair_reweight:
        output_dir += '-FAIR-IPS'
        
    output_dir += f'/trial_{trial.number}'
    train_custom_single(
        local_rank=local_rank,
        base_model=base_model,
        data_path=data_path,
        dataset=dataset,
        output_dir=output_dir,
        seed=999,
        clip_grad_norm=clip_grad_norm, # 1.0,
        batch_size=batch_size,
        micro_batch_size=batch_size // 2,
        num_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        maxlen=maxlen,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=["q_proj", "v_proj"],
        train_on_inputs=train_on_inputs,
        warmup_ratio=warmup_ratio,
        resume_on_checkpoint=None,
        num_batches_train=num_batches_train,
        num_batches_val=num_batches_val,
        train_loader=train_loader,  # Pass global train_loader
        val_loader=val_loader,       # Pass global val_loader
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


def run_study(n_trials=10, n_gpus=2):
    
    study = optuna.create_study(
            study_name="llama_hp_tuning",
            direction="minimize",
            load_if_exists=True
        )
    world_size = n_gpus  # Number of GPUs to use
    
    
    
    study.optimize(objective, n_trials = n_trials, callbacks=[save_all_trials])
    
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for training script")

    # Adding arguments
    parser.add_argument('--base_model', type=str, required=True, help='Base model path or name: eg: meta-llama/Llama-3.2-1B')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--maxlen', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--train_on_inputs', action='store_true', help='Flag to train on inputs')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument("--local_rank",default=2,type=int, help='local-rank (GPU)')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument("--fair_reweight", action='store_true', help='Flag to train with reweighting the input data while finetuning')
    parser.add_argument("--debug", action='store_true', help='Flag to debug')
    parser.add_argument('--group_num', type=int, default=5, help='Number of item groups for popularity fairness (default is 5 as per IFairLRS paper)')
    
    args = parser.parse_args()
    return args    


if __name__ == "__main__":
    
    args = parse_args()
    print("Args: ",args)

    
    base_model = args.base_model
    dataset = args.dataset
    data_path = args.data_path
    batch_size = args.batch_size
    seed = args.seed
    fair_reweight = args.fair_reweight
    group_num = args.group_num
    
    clip_grad_norm = args.clip_grad_norm
    maxlen = args.maxlen
    train_on_inputs = args.train_on_inputs
    
    SAVE_DIR = f"optuna_results/{dataset}"  # Directory to store trial JSON files
    if fair_reweight:
        SAVE_DIR += '-FAIR-IPS'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    AUTH_TOKEN="YOUR_HF_TOKEN"
    tokenizer = AutoTokenizer.from_pretrained(base_model,use_auth_token=AUTH_TOKEN)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    # Limit training to a fixed number of batches per trial for efficiency
    num_batches_train = args.num_batches_train  # 951 for yelp ; 1230 for clothing  ; 600 for toys ; 700 for beauty (10% of train data)
    num_batches_val = args.num_batches_val # 190 for yelp ; 246 for clothing ; 120 for toys ; 140 for beauty (5% of val data)
    
    n_trials = 20
    if args.debug:
        n_trials=2
        num_batches_train = 100
        num_batches_val = 100
    
    epochs = args.num_epochs

    
    local_rank = args.local_rank # trial.number % 2  # Alternate between GPU 0 and GPU 1 # int(os.environ["LOCAL_RANK"])
    
    seedSet(seed)
    # Initialize global DataLoaders (only once)
    initialize_dataloaders(
        tokenizer=tokenizer,  # Pass None since get_dataset_loader initializes it
        dataset=dataset,
        data_path=data_path,
        batch_size=batch_size,
        distributed=False,
        local_rank=local_rank,
        maxlen=maxlen,
        train_on_inputs=train_on_inputs
    )    
    run_study(n_trials=n_trials, n_gpus=2)

