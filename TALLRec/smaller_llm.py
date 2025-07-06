import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import List
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, get_scheduler, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from data_loading import get_dataset_loader
from packaging import version
import torch.multiprocessing as mp
import argparse
from datetime import datetime, timedelta
from time import time
import random
import numpy as np

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from utils import unwrap_peft_model,safe_save_peft_adapter,verify_loaded_peft,load_model_checkpoint,load_best_val_loss,save_checkpoint_peft, seedSet

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for training script")

    # Adding arguments
    parser.add_argument('--base_model', type=str, required=True, help='Base model path or name: eg: meta-llama/Llama-3.2-1B')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--micro_batch_size', type=int, default=16, help='Micro batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimization')
    parser.add_argument('--num_workers',type=int,default=1,help='Number of parallel dataloader workers')
    parser.add_argument('--maxlen', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--warmup_ratio', type=float, required=True, help='Warmup ratio for learning rate schedule')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, required=True, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, default='[q_proj,v_proj]', help='List of target modules for LoRA')
    parser.add_argument('--train_on_inputs', action='store_true', help='Flag to train on inputs')
    parser.add_argument('--resume_on_checkpoint', type=str, default=None, help="Checkpoint to resume training from (checkpoint_epoch_{}.pt files if applicable)")
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument('--distributed', action='store_true', help='Distributed training flag')
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    
    args = parser.parse_args()
    return args


mp.set_start_method('spawn', force=True)


# LOGIN CREDENTIALS: HuggingFACE in AUTH_TOKEN

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

def initialize_distributed():
#     if "RANK" not in os.environ:
#         os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "2"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://",  timeout=timedelta(seconds=3600 * 24))
        
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

def train_custom(
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
    distributed: bool = False,
    train_on_inputs: bool = True,
    resume_on_checkpoint: str = None,  # either training checkpoint or final adapter
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1,
    early_stopping_patience = 3,
    num_workers: int = 1,
    num_batches_train: int = -1,
    num_batches_val: int = -1,
    train_loader = None,
    val_loader = None,     
):
    AUTH_TOKEN="YOUR_HF_TOKEN"
    seedSet(seed)
    start = time()
    print(f"Early stopping patience: {early_stopping_patience}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    os.makedirs(output_dir, exist_ok=True)
    
    device_map={"": local_rank}
    
    
    model = AutoModelForCausalLM.from_pretrained(base_model,load_in_8bit=True, torch_dtype=torch.float16,device_map=device_map,use_auth_token=AUTH_TOKEN)
    
    print(f"Training model: {base_model}")
    
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
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
    
    

#     old_state_dict = model.state_dict
#     model.state_dict = (
#         lambda self, *_, **__: get_peft_model_state_dict(
#             self, old_state_dict()
#         )
#     ).__get__(model, type(model))

#     if torch.__version__ >= "2" and sys.platform != "win32":
#         print("Compile the model!")
#         model = torch.compile(model)

    
    model = model.to(f"cuda:{torch.cuda.current_device()}")
    
    if distributed:
        
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.to(f"cuda:{local_rank}")
    
    if train_loader is None or val_loader is None:
    
        train_sample_number, val_sample_number = 5, 2
        train_loader, train_dataset = get_dataset_loader(tokenizer, train_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,mode='train', dataset=dataset, data_path=data_path, batch_size=micro_batch_size, workers=num_workers, distributed=distributed,local_rank=local_rank)



        val_loader, val_dataset = get_dataset_loader(tokenizer, val_sample_number, cutoff_len=maxlen, train_on_inputs=train_on_inputs,mode='val', dataset=dataset, data_path=data_path, batch_size=micro_batch_size, workers=num_workers, distributed=distributed,local_rank=local_rank)
    
    else:
        print("Already loaded train and val loaders")
    
    num_batches_train = num_batches_train if num_batches_train > -1 else len(train_loader)
    print(f"num_batches_train: {num_batches_train}\n")
    
    num_batches_val = num_batches_val if num_batches_val > -1 else len(val_loader)
    print(f"num_batches_val: {num_batches_val}\n")
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,gradient_accumulation_steps,num_epochs,warmup_ratio,learning_rate,weight_decay,adam_eps=1e-6)
    
    scaler = GradScaler()
    
    if distributed:
        dist.barrier()
        print(f"🔍 Model class at training start: {type(model.module)}")
    if dist.get_rank() == 0:
        best_val_loss = load_best_val_loss(resume_on_checkpoint.strip('.pt').split('checkpoint_epoch_')[0] + 'best_val_loss.pt') if resume_on_checkpoint and os.path.exists(resume_on_checkpoint.strip('.pt').split('checkpoint_epoch_')[0] + 'best_val_loss.pt') else float('inf')
    
    # Diagnostic: check optimizer param groups
    print("🔍 Checking optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        nonzero_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        print(f"  Group {i}: {nonzero_params} trainable parameters")
    

    # Diagnostic: snapshot a LoRA weight before training
    lora_keys = [k for k in unwrap_peft_model(model).state_dict() if 'lora_' in k]
    if lora_keys:
        print(f"🔍 Tracking LoRA weight change for: {lora_keys[0]}")
        lora_before = unwrap_peft_model(model).state_dict()[lora_keys[0]].clone()
    else:
        print("⚠️ No LoRA keys found before training!")
    
    start = int(resume_on_checkpoint.strip('.pt').split('checkpoint_epoch_')[-1]) if resume_on_checkpoint and 'checkpoint_epoch_' in resume_on_checkpoint else 0
    print("Starting at epoch :",start)
    
    early_stopping_counter = 0
    terminate = False
    
    for epoch in range(start, start + num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in ['input_ids','attention_mask','labels']}
            # print(f"Batch shapes at step {step}: {[v.shape for k, v in batch.items()]}")

            if step > num_batches_train:
                break
            
            with autocast():
                if distributed:
                    try:
                        # print(f"[DEBUG] Model forward arguments: {model.module.forward.__code__.co_varnames}")
                        outputs = model.module(input_ids=batch["input_ids"].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch["labels"].to(device, non_blocking=True))
                        

                    except RuntimeError as e:
                        print(f"RuntimeError encountered: {e}")
                        print(f"Input shape: {[v.shape for k, v in batch.items()]}")
                        print(f"Model num_heads: {model.module.config.num_attention_heads}")
                        print(f"Model hidden_size: {model.module.config.hidden_size}")
                        print(f"Expected Head Dim: {model.module.config.hidden_size // model.module.config.num_attention_heads}")
                        raise
                else:
                    outputs = model(input_ids=batch["input_ids"].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch["labels"].to(device, non_blocking=True))
                
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            for param in model.parameters():
                param.grad = None
            
            
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
            
            # optimizer.zero_grad()
            if step % 10 == 0:
                print(f"epoch: {epoch+1} batch: {step} lr: {lr:.6f} loss: {total_loss / max(1,step):.6f}")
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        if distributed:
            dist.barrier()
        
        if dist.get_rank() == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for stepv, batch in enumerate(tqdm(val_loader)):
                    # batch = {k: v.to(f"cuda:{torch.cuda.current_device()}", non_blocking=True) for k, v in batch.items()}
                    if stepv > num_batches_val:
                        break
                    if distributed:
                        outputs = model.module(input_ids=batch["input_ids"].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch["labels"].to(device, non_blocking=True))
                    else:
                        outputs = model(input_ids=batch["input_ids"].to(device, non_blocking=True),attention_mask=batch["attention_mask"].to(device, non_blocking=True),labels=batch["labels"].to(device, non_blocking=True))
                    val_loss += outputs.loss.item()
            avg_val_loss = val_loss / max(1,num_batches_val)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
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

        terminate = torch.tensor(terminate, dtype=torch.bool).cuda()
        dist.broadcast(terminate, 0)
        if terminate:
            # torch.save(unwrap_peft_model(model).state_dict(), f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint_peft(model,f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
            

            safe_save_peft_adapter(model, output_dir)


            print("Exitting now...")
            print(f'It took {time() - start:.1f}s')
            if distributed:
                dist.destroy_process_group()
            exit()
        
        if distributed:
            dist.barrier()
        
        # torch.save(unwrap_peft_model(model).state_dict(), f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint_peft(model,f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")

    if distributed:
        safe_save_peft_adapter(model.module, output_dir)
    
    else:
        safe_save_peft_adapter(model, output_dir)


    
    # Diagnostic: compare before/after weight
    if lora_keys:
        lora_after = unwrap_peft_model(model).state_dict()[lora_keys[0]]
        diff = torch.mean(torch.abs(lora_after - lora_before)).item()
        print(f"🔍 Mean abs diff in tracked LoRA weight: {diff:.6f}")
        if diff < 1e-6:
            print("❌ LoRA weights did not change — training may not have updated them!")
        else:
            print("✅ LoRA weights were updated.")
    
    
    state_dict = unwrap_peft_model(model).state_dict()
    lora_keys_post = [k for k in state_dict if "lora_" in k]
    print(f"Saved {len(lora_keys_post)} LoRA weights")
    if len(lora_keys_post) == 0:
        print("⚠️ WARNING: No LoRA weights found in unwrap_peft_model(model).state_dict() — did you save from the wrong model?")

    
    print("Training Complete. Model saved.")
    print(f'It took {time() - start:.1f}s')
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    args = parse_args()
    print("Args: ",args)
    
    if args.distributed:
        initialize_distributed()
    else:
        os.environ["LOCAL_RANK"] = "0"
    
    
    train_custom(
        local_rank= int(os.environ["LOCAL_RANK"]),
        base_model = args.base_model,
        data_path=args.data_path,
        dataset=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        clip_grad_norm=args.clip_grad_norm,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        maxlen=args.maxlen,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules = eval(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        warmup_ratio=args.warmup_ratio,
        distributed=args.distributed,
        train_on_inputs=args.train_on_inputs,
        resume_on_checkpoint = args.resume_on_checkpoint,
        num_batches_train=args.num_batches_train,
        num_batches_val = args.num_batches_val,
        train_loader=None,  # Pass global train_loader
        val_loader=None      # Pass global val_loader
    )

