import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import List
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, get_scheduler, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from packaging import version
import torch.multiprocessing as mp
from datetime import datetime, timedelta
from time import time
import random
import numpy as np
import inspect

from utils import seedSet,unwrap_peft_model,safe_save_peft_adapter,verify_loaded_peft,load_model_checkpoint,load_best_val_loss,save_checkpoint_peft
from IPSReweighter import IPS
from data_loading import get_dataset_loader

mp.set_start_method('spawn', force=True)


# LOGIN CREDENTIALS: HuggingFACE: in AUTH_TOKEN

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

def compute_loss(logits, labels, weights, VOCAB_SIZE):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, VOCAB_SIZE)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    loss = torch.mean(weights * torch.mean(loss_fct(shift_logits, shift_labels).view(weights.shape[0], -1)))

    return loss


def create_opt_lr(model,data_len,gradient_accumulation_steps,num_epochs,warmup_ratio,learning_rate,weight_decay=1e-4,adam_eps=1e-6): 
    batch_per_epoch = data_len
    t_total = batch_per_epoch // gradient_accumulation_steps * num_epochs
    warmup_iters = int(t_total * warmup_ratio)
    print("Batch per epoch: %d" % batch_per_epoch)
    print("Total Iters: %d" % t_total)
    print('Warmup ratio:', warmup_ratio)
    print("Warm up Iters: %d" % warmup_iters)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = adam_eps)
    num_training_steps = num_epochs * data_len
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,warmup_iters,t_total)
    return optimizer, lr_scheduler

def train_custom_single(
    local_rank: int,
    base_model: str,
    data_path: str,
    dataset: str,
    output_dir: str,
    seed: int,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    maxlen: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    train_on_inputs: bool = True,
    resume_on_checkpoint: str = None,  # either training checkpoint or final adapter
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1,
    early_stopping_patience = 3,
    num_workers: int = 0,
    num_batches_train: int = -1,
    num_batches_val: int = -1,
    train_loader = None,
    val_loader = None,  
    fair_reweight = False,
    variance_control = 0.1, # Avoid Variance control for Zero division cases upon reweighting
    group_num = 5,
    train_dataset = None,
):
    
    AUTH_TOKEN="YOUR_HF_TOKEN"
    seedSet(seed)
    
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("Arguments passed to train_custom():")
    for arg in args:
        print(f"\t {arg} = {values[arg]}")
        
    start = time()
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    os.makedirs(output_dir, exist_ok=True)
    
    device_map={"": local_rank}
    
    
    model = AutoModelForCausalLM.from_pretrained(base_model,load_in_8bit=True, torch_dtype=torch.float16,device_map=device_map,use_auth_token=AUTH_TOKEN)
    
    print(f"Training model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model,use_auth_token=AUTH_TOKEN)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    model = load_model_checkpoint(resume_on_checkpoint, model)
    
    verify_loaded_peft(unwrap_peft_model(model))
    
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.config.use_cache = False

    
    model = model.to(f"cuda:{local_rank}")
    
    if train_loader is None or val_loader is None:
    
        train_sample_number, val_sample_number = 5, 2
        train_loader, train_dataset = get_dataset_loader(tokenizer, train_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,mode='train', dataset=dataset, data_path=data_path, batch_size=micro_batch_size, workers=num_workers, distributed=False,local_rank=local_rank)

        val_loader, val_dataset = get_dataset_loader(tokenizer, val_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,mode='val', dataset=dataset, data_path=data_path, batch_size=micro_batch_size, workers=num_workers, distributed=False,local_rank=local_rank)
    
    else:
        print("Already loaded train and val loaders")
    
    num_batches_train = num_batches_train if num_batches_train > -1 else len(train_loader)
    print(f"num_batches_train: {num_batches_train}\n")
    
    num_batches_val = num_batches_val if num_batches_val > -1 else len(val_loader)
    print(f"num_batches_val: {num_batches_val}\n")
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,gradient_accumulation_steps,num_epochs,warmup_ratio,learning_rate,weight_decay,adam_eps=1e-6)
    
    scaler = GradScaler()
    

    best_val_loss = float('inf')
    
    terminate = False
    
    VOCAB_SIZE = model.config.vocab_size
    print("Vocab Size: ",VOCAB_SIZE)
    
    if fair_reweight:
        # FAIR RANKER REWEIGHTING CASE
        Fair_Ranker = IPS(dataset = train_dataset, group_num = group_num, group_weight = np.ones(group_num), variance_control=variance_control)
    
    for epoch in range(num_epochs):
        
        
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            
            # batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'data_point'}
            if step > num_batches_train:
                break

            with autocast():
                outputs = model(input_ids=batch['input_ids'].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch['labels'].to(device, non_blocking=True))
                if fair_reweight:
                    weights = Fair_Ranker.reweight(input_dict={'target_items':batch['target_items']})
                    weights = torch.tensor(weights).to(device)
                    loss = compute_loss(outputs.get('logits'),batch['labels'],weights,VOCAB_SIZE)
                else:
                    loss = outputs.loss
            
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            
            
            if lr_scheduler:
                if version.parse(torch.__version__) >= version.parse("1.4"):
                    lr = lr_scheduler.get_last_lr()[0]
                else:
                    lr = lr_scheduler.get_lr()[0]
            else:
                try:
                    lr = optimizer.get_lr()[0]
                except AttributeError:
                    lr = learning_rate
            
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f"epoch: {epoch} batch: {step} lr: {lr:.6f} loss: {total_loss / max(1,step):.6f}")
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for stepv, batch in enumerate(tqdm(val_loader)):
                # batch = {k: v.to(f"cuda:{torch.cuda.current_device()}", non_blocking=True) for k, v in batch.items() if k != 'data_point'}
                
                if stepv > num_batches_val:
                    break
                
                outputs = model(input_ids=batch['input_ids'].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch['labels'].to(device, non_blocking=True))
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / max(1,num_batches_val)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Clear unused memory
        torch.cuda.empty_cache()

        # Inside train_custom function (After validation step) | Do check ONLY in one core!

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(best_val_loss, f"{output_dir}/best_val_loss.pt")
            early_stopping_counter = 0
            terminate=False
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                terminate=True
        
        
        if terminate:
            save_checkpoint_peft(model,f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
            safe_save_peft_adapter(model, output_dir)
            print("Exitting now...")
            print(f'It took {time() - start:.1f}s')
            return
        
        save_checkpoint_peft(model,f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    safe_save_peft_adapter(model, output_dir)
    print("Training Complete. Model saved.")
    print(f'It took {time() - start:.1f}s')
    return

