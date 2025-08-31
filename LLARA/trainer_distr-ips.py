import argparse
import os
import random
from tqdm import tqdm
import numpy as np
# import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from packaging import version
from time import time
from datetime import timedelta

from model_arch import LLARA
from utils import save_checkpoint,seedSet, unwrap_dist_model
from data_loading import get_dataset_loader,get_dataset_object

from IPSReweighter import IPS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    
    # Adding arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimization')
    parser.add_argument('--num_workers',type=int,default=1,help='Number of parallel dataloader workers')
    parser.add_argument('--maxlen', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--warmup_ratio', type=float, required=True, help='Warmup ratio for learning rate schedule')
        
    
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')
    parser.add_argument('--distributed', action='store_true', help='Distributed training flag')
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    
    parser.add_argument("--rec_dim",type=int,default=64,help="Recommender Embedding Size")
    parser.add_argument("--lora_r",type=int,default=8,help="Lora R Value")
    parser.add_argument("--lora_alpha",type=int,default=16,help="Lora Alpha Value")
    parser.add_argument('--lora_dropout', type=float, default=0.2, help='Lora Dropout')
    parser.add_argument("--early_stopping_patience",type=int,default=3,help='Number of attempts to stop validation')
    
    
    parser.add_argument("--llama_ckpt", type=str, default=None, help="Checkpoint to load LLaMa model weights alone: give just the directory")
    parser.add_argument("--rec_ckpt", type=str, default=None, help="Checkpoint to load Recommender model weights alone")
    parser.add_argument("--proj_ckpt", type=str, default=None, help="Checkpoint to load Projector model weights alone")
    
    parser.add_argument("--prompt_path",type=str, default="prompts/collm_amazon.txt",help='Path to load the prompt styles: e.g.: "prompts/collm_amazon.txt"')
    parser.add_argument('--group_num', type=int, default=5, help='Number of item groups for popularity fairness (default is 5 as per IFairLRS paper)')
    parser.add_argument("--debug", action='store_true', help='Debug for training flag')
    
    args = parser.parse_args()
    return args

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
            "params": [p for n, p in unwrap_dist_model(model).projector.named_parameters()],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in unwrap_dist_model(model).llama_model_lora.named_parameters()] ,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = adam_eps)
    num_training_steps = num_epochs * data_len
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,warmup_iters,t_total)
    return optimizer, lr_scheduler


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


def train_custom(args):
    seedSet(args.seed)
    
    early_stopping_patience = args.early_stopping_patience
    variance_control = 0.1
    
    print(f"Early stopping patience: {early_stopping_patience}")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    base_model = "meta-llama/Llama-3.2-1B"
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    output_dir = args.output_dir
    
    os.makedirs(os.path.join(output_dir,"LORA"), exist_ok=True)
    os.makedirs(os.path.join(output_dir,"PROJ"), exist_ok=True)
    
    device_map={"": args.local_rank}
    
    # Step 0: Train setup basics
    start = time()
    check_step = 4
    # Step 1: Data Creation
    train_sample_number, val_sample_number = 5, 2
    
    train_dataset = get_dataset_object(sample_numbers = train_sample_number, mode='train', dataset=args.dataset, data_path=args.data_path, local_rank = args.local_rank,num_groups=args.group_num)
    val_dataset = get_dataset_object(sample_numbers = val_sample_number, mode='val', dataset=args.dataset, data_path=args.data_path, local_rank = args.local_rank,num_groups=args.group_num)
    
    fair_reweight=True
    
    Fair_Ranker = IPS(dataset = train_dataset, group_num = args.group_num, group_weight = np.ones(args.group_num), variance_control=variance_control)
    
    
    # Step 2: Model Creation
    freeze_lora = False
    if args.llama_ckpt:
        freeze_lora=True
    
    model = LLARA(
        rec_model="MF",
        user_num=max(train_dataset.user_num,val_dataset.user_num) + 1,
        item_num=max(train_dataset.item_num,val_dataset.item_num) + 1,
        embedding_size=args.rec_dim,
        freeze_rec=True,
        freeze_lora=freeze_lora,
        freeze_proj=False,
        llama_model=base_model,
        max_txt_len=args.maxlen,
        end_sym='\n',
        low_resource=True,  # use 8 bit
        device_8bit= args.local_rank,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=["q_proj","v_proj"],
        lora_dropout=args.lora_dropout,
        llama_ckpt = args.llama_ckpt,
        rec_ckpt = args.rec_ckpt,
        proj_ckpt = args.proj_ckpt,
        fair_reweight = fair_reweight,
    )
    
    
    model = model.to(f"cuda:{torch.cuda.current_device()}")
    tokenizer = model.llama_tokenizer
    VOCAB_SIZE = model.llama_model_lora.config.vocab_size
    print("Vocab Size: ",VOCAB_SIZE)
    
    if args.distributed:
        
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(f"cuda:{args.local_rank}")
    
    # Step 3: Data Loader
    train_loader = get_dataset_loader(data_obj = train_dataset, tokenizer = tokenizer, prompt_path = args.prompt_path, mode='train', batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed,max_epochs = args.num_epochs,maxlen=args.maxlen, fair_reweight=fair_reweight)
    
    val_loader = get_dataset_loader(data_obj = val_dataset, tokenizer = tokenizer, prompt_path = args.prompt_path, mode='val',  batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed,max_epochs = args.num_epochs,maxlen=args.maxlen)
    
    del tokenizer
    
    if not args.debug:
        num_batches_train = args.num_batches_train if args.num_batches_train > -1 else len(train_loader)
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(val_loader)
        
    else:
        num_batches_train = 50
        num_batches_val = 50
        
    print(f"num_batches_train: {num_batches_train}\n")
    print(f"num_batches_val: {num_batches_val}\n")
    
    
    

    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,args.gradient_accumulation_steps,args.num_epochs,args.warmup_ratio,args.learning_rate,args.weight_decay,adam_eps=1e-6)
    
    # Diagnostic: check optimizer param groups
    print("🔍 Checking optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        nonzero_params = sum(p.numel() for p in group['params'] if p.requires_grad)
        print(f"  Group {i}: {nonzero_params} trainable parameters")
    
    scaler = GradScaler()
    
    if args.distributed:
        dist.barrier()
        model.module.set_mode('v2')
        print(f"🔍 Model class at training start: {type(model.module)}")
    
    else:
        model.set_mode('v2')
    
    start_epoch = 0
    
    early_stopping_counter = 0
    terminate = False
    
    if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
        best_val_loss = float('inf')
    
    
    # Step 4: Training loop
    print("Starting at epoch :",start_epoch)
    for epoch in range(start_epoch, start_epoch + args.num_epochs): 
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            model.module.to_be_trained()
        else:
            model.to_be_trained()
        
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if step > num_batches_train:
                break
            
            with torch.cuda.amp.autocast():
                if args.distributed:
                    outputs = model.module(batch)
                    logits = outputs['logits']
                    labels = outputs['labels']
                    weights = Fair_Ranker.reweight(input_dict={'target_items':batch['TargetItemID']})
                    weights = torch.tensor(weights).to(device)
                    loss = compute_loss(logits, labels, weights, VOCAB_SIZE)
                else:
                    outputs = model(batch)
                    logits = outputs['logits']
                    labels = outputs['labels']
                    weights = Fair_Ranker.reweight(input_dict={'target_items':batch['TargetItemID']})
                    weights = torch.tensor(weights).to(device)
                    loss = compute_loss(logits, labels, weights, VOCAB_SIZE)
            
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            
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
                    lr = args.learning_rate
            
            
            if step % 10 == 0:
                print(f"epoch: {epoch + 1} batch: {step} lr: {lr:.6f} loss: {total_loss / max(1,step):.6f}",end='\t')
                print(f"weights: {weights}")
            
            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        if args.distributed:
            dist.barrier()
        
        # Step 5 : Validation Loop
        if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for stepv, batch in enumerate(tqdm(val_loader)):
                    if stepv > num_batches_val:
                        break
                    if args.distributed:
                        outputs = model.module.generate_for_samples(batch)
                    else:
                        outputs = model.generate_for_samples(batch)
                    
                    val_loss += outputs['loss'].item()
                    torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / max(1,num_batches_val)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            
            # Inside train_custom function (After validation step) | Do check ONLY in one core!
        
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(best_val_loss, f"{output_dir}/best_val_loss.pt")
                save_checkpoint(model, output_dir, "BEST", stage=-1)
                early_stopping_counter = 0
                terminate=False
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    print("Early stopping triggered.")
                    terminate=True
        
        # Early Stopping condition
        terminate = torch.tensor(terminate, dtype=torch.bool).cuda()
        if args.distributed:
            dist.broadcast(terminate, 0)
        if terminate:
            save_checkpoint(model, output_dir, epoch, stage=-1)
            
            print("Exitting now...")
            if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
                print(f'It took {time() - start:.1f}s')
            
            if args.distributed:
                dist.destroy_process_group()
            exit()
        
        
        if args.distributed:
            dist.barrier()
        
        # Save the model
        if (epoch % check_step == 0):
            save_checkpoint(model, output_dir, epoch, stage=-1)
    
    save_checkpoint(model, output_dir, epoch, stage=-1)
    print("Training Complete. Model saved.")
    if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
        print(f'It took {time() - start:.1f}s')
    if args.distributed:
        model.module.set_mode(None)
        dist.destroy_process_group()
    
    else:
        model.set_mode(None)
    
    return
        

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
    args = parse_args()
    print("Args: ",args)
    
    if args.distributed:
        initialize_distributed()
    else:
        os.environ["LOCAL_RANK"] = "0"
    
    train_custom(args)
