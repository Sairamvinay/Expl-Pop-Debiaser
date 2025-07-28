import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from packaging import version
from time import time
from datetime import timedelta
import inspect

from model_arch import LLARA
from utils import seedSet
from IPSReweighter import IPS


def compute_loss(logits, labels, weights, VOCAB_SIZE):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, VOCAB_SIZE)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    
    FCT = loss_fct(shift_logits, shift_labels).view(weights.shape[0], -1)

    loss = torch.mean(weights * torch.mean(FCT))

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
            "params": [p for n, p in model.projector.named_parameters()],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.llama_model_lora.named_parameters()] ,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = adam_eps)
    num_training_steps = num_epochs * data_len
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,warmup_iters,t_total)
    return optimizer, lr_scheduler


def train_custom_single(seed=999,
                        early_stopping_patience=3,
                        gpu = 0,
                        rec_dim = 64,
                        output_dir = 'snap',
                        prompt_path = 'prompts/llara_amazon.txt',
                        maxlen = 512,
                        lora_r=16,
                        lora_alpha=16,
                        lora_dropout = 0.0,
                        llama_ckpt = None,
                        rec_ckpt = '../MF/snap/beauty/checkpoint_epoch_BEST.pt',
                        proj_ckpt = None,
                        num_batches_train = 100,
                        num_batches_val = 10,
                        gradient_accumulation_steps = 1,
                        num_epochs = 2,
                        warmup_ratio = 0.1,
                        learning_rate = 1e-4,
                        weight_decay = 1e-4,
                        clip_grad_norm = 1.0,
                        train_loader = None,
                        val_loader = None,
                        user_num = 22363,
                        item_num = 12101,
                        fair_reweight = False,
                        group_num = 5,
                        train_dataset = None,      
                        variance_control=0.1, # Avoid Variance control for Zero division cases upon reweighting
                       ):
    
    
    seedSet(seed) 
    x, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("Arguments passed to train_custom_single():")
    for arg in x:
        print(f"\t {arg} = {values[arg]}")
    
    print(f"Early stopping patience: {early_stopping_patience}")
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    base_model = "meta-llama/Llama-3.2-1B"
    
    device_map={"": gpu}
    
    # Step 0: Train setup basics
    start = time()
    check_step = 4
    
    if fair_reweight:
        # FAIR RANKER REWEIGHTING CASE
        Fair_Ranker = IPS(dataset = train_dataset, group_num = group_num, group_weight = np.ones(group_num), variance_control=variance_control)
    
    # Step 2: Model Creation
    model = LLARA(
        rec_model="MF",
        user_num=user_num,
        item_num=item_num,
        embedding_size=rec_dim,
        freeze_rec=True,
        freeze_lora=False,
        freeze_proj=False,
        llama_model=base_model,
        max_txt_len=maxlen,
        end_sym='\n',
        low_resource=True,  # use 8 bit
        device_8bit= gpu,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=["q_proj","v_proj"],
        lora_dropout=lora_dropout,
        llama_ckpt=llama_ckpt,
        rec_ckpt=rec_ckpt,
        proj_ckpt=proj_ckpt,
        fair_reweight = fair_reweight,
    )

    model = model.to(f"cuda:{gpu}")
    
    VOCAB_SIZE = model.llama_model_lora.config.vocab_size
    print("Vocab Size: ",VOCAB_SIZE)
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,gradient_accumulation_steps,num_epochs,warmup_ratio,learning_rate,weight_decay,adam_eps=1e-6)
    
    # Diagnostic: check optimizer param groups
    print("🔍 Checking optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        nonzero_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        print(f"  Group {i}: {nonzero_params} trainable parameters")
    
    scaler = GradScaler()
    
    model.set_mode('v2')
    
    start_epoch = 0
    
    early_stopping_counter = 0
    terminate = False
    
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 4: Training loop
    print("Starting at epoch :",start_epoch)
    for epoch in range(start_epoch, start_epoch + num_epochs): 
        model.to_be_trained()
        
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if step > num_batches_train:
                break
            
            with torch.cuda.amp.autocast():
                if fair_reweight:
                    outputs = model(batch)
                    logits = outputs['logits']
                    labels = outputs['labels']
                    weights = Fair_Ranker.reweight(input_dict={'target_items':batch['TargetItemID']})
                    weights = torch.tensor(weights).to(device)
                    loss = compute_loss(logits, labels, weights, VOCAB_SIZE)
                else:
                    loss = model(batch)["loss"]
            
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
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
            
            
            if step % 10 == 0:
                if fair_reweight:
                    print(f"epoch: {epoch + 1} batch: {step} lr: {lr:.6f} loss: {total_loss / max(1,step):.6f}",end='\t')
                    print(f"weights: {weights}")
                else:
                    print(f"epoch: {epoch+1} batch: {step} lr: {lr:.6f} loss: {total_loss / max(1,step):.6f}")
            
            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Clear unused memory
        torch.cuda.empty_cache()
        
        # Step 5 : Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for stepv, batch in enumerate(tqdm(val_loader)):
                if stepv > num_batches_val:
                    break
                outputs = model.generate_for_samples(batch)

                val_loss += outputs['loss'].item()
                torch.cuda.empty_cache()

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
            print("Exitting now...")
            print(f'It took {time() - start:.1f}s')    
            exit()
        
    
    print("Training Complete.")
    print(f'It took {time() - start:.1f}s')
    model.set_mode(None)
    
    return
