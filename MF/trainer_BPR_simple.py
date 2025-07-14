import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pickle
import os
import torch.multiprocessing as mp
import argparse
from time import time
from packaging import version
from datetime import timedelta
import inspect

from model_utils import MatrixFactorization, bpr_loss
from utils import seedSet, load_checkpoint

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

def train_custom_single(seed = 999,
                        early_stopping_patience=3,
                        gpu = 0,
                        num_batches_train = 10,
                        num_batches_val = 10,
                        gradient_accumulation_steps = 1,
                        num_epochs = 2,
                        clip_grad_norm = 1.0,
                        train_loader=None, 
                        val_loader = None,
                        user_num = 22363,
                        item_num = 12101,
                        output_dir = 'snap',
                        dataset = 'beauty',
                        id_embed_dim = 64,
                        warmup_ratio = 0.1,
                        learning_rate = 1e-4,
                        weight_decay = 1e-4,
                        checkpoint=None,
                        debug= False,
                        temperature = 5,
                       ):
    
    seedSet(seed) 
    x, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("Arguments passed to train_custom_single():")
    for arg in x:
        print(f"\t {arg} = {values[arg]}")
    
    print(f"Early stopping patience: {early_stopping_patience}")
    
    start = time()
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    device_map={"": gpu}
    
    
    
    model = MatrixFactorization(user_num=user_num, item_num=item_num, embedding_size=id_embed_dim)
    
    model = model.to(f"cuda:{gpu}")
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,gradient_accumulation_steps,num_epochs,warmup_ratio,learning_rate,weight_decay,adam_eps=1e-6)

    # Diagnostic: check optimizer param groups
    print("🔍 Checking optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        nonzero_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        print(f"  Group {i}: {nonzero_params} trainable parameters")
    
    scaler = GradScaler()
    early_stopping_counter = 0
    to_print = True
    terminate = False
    best_val_loss = float('inf')
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    if checkpoint:
        model = load_checkpoint(model, checkpoint)
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc = f"Epoch {epoch + 1}")):
            if step > num_batches_train:
                break
            
            torch.cuda.empty_cache()
            with autocast(): 
                
                user_ids = torch.tensor(batch['UserID']).to(device)
                pos_item_ids = torch.tensor(batch['PosItem']).to(device)
                neg_item_ids = torch.tensor(batch['NegItem']).to(device)
                interact_score = model(user_ids = user_ids, item_ids = pos_item_ids)
                uninteract_score = model(user_ids = user_ids, item_ids = neg_item_ids)
                
                if debug:
                    print("batch: ",batch)
                    print("user_ids min/max:", user_ids, "expected 0 to", user_num-1)
                    print("pos_item_ids min/max:", pos_item_ids, "expected 0 to", item_num-1)
                    print("neg_item_ids min/max:", neg_item_ids, "expected 0 to", item_num-1)

                loss = bpr_loss(interact_score, uninteract_score)
                msg = ""
                if step == 0:
                    print("user ids: ",user_ids, end = ' ,')
                    print("pos item ids: ",pos_item_ids, end = ' ,')
                    print("neg item ids: ",neg_item_ids, end = ' ,')

                    print("pos_score: ",interact_score, end = ' ,')
                    print("neg_score: ",uninteract_score, end = ' ,')
                    print("loss: ",loss.item(),end= ' ,')

                if debug:
                    with torch.no_grad():
                        print(f"\t\tInteract Score Logits: mean={interact_score.mean().item():.4f}, std={interact_score.std().item():.4f}, max={interact_score.max().item():.4f}")
                        print(f"\t\tUnInteract Score Logits: mean={uninteract_score.mean().item():.4f}, std={uninteract_score.std().item():.4f}, max={uninteract_score.max().item():.4f}")
                
                
                
                
            
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
                print(f"epoch: {epoch+1} batch: {step} lr: {lr:.6f} {msg} loss: {total_loss / max(1,step):.6f}")
            
            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        torch.cuda.empty_cache()
        
        # Step 5 : Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for stepv, batchv in enumerate(tqdm(val_loader)):
                    if stepv > num_batches_val:
                        break
                
                    with autocast():
                        
                        user_ids = torch.tensor(batchv['UserID']).to(device)
                        pos_item_ids = torch.tensor(batchv['PosItem']).to(device)
                        neg_item_ids = torch.tensor(batchv['NegItem']).to(device)

                        interact_score = model(user_ids = user_ids, item_ids = pos_item_ids)
                        uninteract_score = model(user_ids = user_ids, item_ids = neg_item_ids)
                        val_loss = bpr_loss(interact_score, uninteract_score)
                        
                    torch.cuda.empty_cache()
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / max(1,num_batches_val)
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
            return
            # exit()
            
    print("Training Complete.")
    print(f'It took {time() - start:.1f}s')
    return
    