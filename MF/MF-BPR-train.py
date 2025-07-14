import torch
import torch.distributed as dist
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
import numpy as np

from data_utils import load_pickle, load_json, readTargetItem
from model_utils import MatrixFactorization, bpr_loss
from utils import seedSet,print_trainable_parameters,save_checkpoint,load_checkpoint,verify_model_weights
from train_dataset_BPR import get_BPRdataset_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for MF script")

    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--debug',action='store_true',help='Debug or not?')
    parser.add_argument("--num_workers",type=int,default=4,help='DataLoader Num Workers')
    
    parser.add_argument('--epochs',type=int,default=5,help='Number of Training Epochs')
    parser.add_argument("--batch_size",type=int,default=16,help='Batch Size for training')
    parser.add_argument('--id_embed_dim',type=int,default=64,help='MatrixFactorization Embedding Dimension')
    
    parser.add_argument("--learning_rate",type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--distributed', action='store_true', help='Distributed training flag')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimization')
    parser.add_argument('--warmup_ratio', type=float, required=True, help='Warmup ratio for learning rate schedule')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument("--temperature",type=float, default=5.0, help='Temperature for Scaling prior to sigmoid')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument("--output_dir",type=str,default='snap/',help='Path to save the model checkpoints')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument("--checkpoint",type=str, default=None, help = 'Load checkpoints')
    
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')

    args = parser.parse_args()
    return args



mp.set_start_method('spawn', force=True)
def initialize_distributed():
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


def main(args, train_loader, val_loader, user_num, item_num, local_rank):
    start = time()
    device = torch.device(f"cuda:{local_rank}")
    device_map={"": local_rank}
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    seedSet(args.seed)
    model_name = "meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-3.2-1B" 
    
    early_stopping_patience = 3
    
    
    model = MatrixFactorization(user_num = user_num, item_num = item_num, embedding_size = args.id_embed_dim).to(device)
    
    if args.distributed:
        
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.to(f"cuda:{local_rank}")
    
    if args.debug:
        num_batches_train = 10
        num_batches_val = 10
    else:
        num_batches_train = args.num_batches_train if args.num_batches_train > -1 else len(train_loader)
        print(f"num_batches_train: {num_batches_train}\n")
        
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(val_loader)
        print(f"num_batches_val: {num_batches_val}\n")
    
    
    check_step = 10
    
    output_dir = os.path.join(args.output_dir, args.dataset)
    
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,args.gradient_accumulation_steps,args.epochs,args.warmup_ratio,args.learning_rate,args.weight_decay,adam_eps=1e-6)
    
    scaler = GradScaler()
    
    early_stopping_counter = 0
    terminate = False
    to_print = True
    
    if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
        best_val_loss = float('inf')
    
    if args.distributed:
        dist.barrier()
        print(f"🔍 Model class at training start: {type(model.module)}")
    
    
        
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
        verify_model_weights(model)
    
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(tqdm(train_loader, desc = f"Epoch {epoch + 1}")):
            if step > num_batches_train:
                break
            
            torch.cuda.empty_cache()
            with autocast(): 

                user_ids = torch.tensor(batch['UserID']).to(device)
                pos_item_ids = torch.tensor(batch['PosItem']).to(device)
                neg_item_ids = torch.tensor(batch['NegItem']).to(device)


                if args.distributed:
                    interact_score = model.module(user_ids = user_ids, item_ids = pos_item_ids)
                    uninteract_score = model.module(user_ids = user_ids, item_ids = neg_item_ids)
                else:
                    interact_score = model(user_ids = user_ids, item_ids = pos_item_ids)
                    uninteract_score = model(user_ids = user_ids, item_ids = neg_item_ids)

                loss = bpr_loss(interact_score, uninteract_score)
                msg = ""
                if args.debug or step == 0:
                    print("user ids: ",user_ids, end = ' ,')
                    print("pos item ids: ",pos_item_ids, end = ' ,')
                    print("neg item ids: ",neg_item_ids, end = ' ,')

                    print("pos_score: ",interact_score, end = ' ,')
                    print("neg_score: ",uninteract_score, end = ' ,')
                    print("loss: ",loss.item(),end= ' ,')


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
                print(f"epoch: {epoch+1} batch: {step} lr: {lr:.6f} {msg} loss: {total_loss / max(1,step):.6f}")
            
            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(1,num_batches_train)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        if args.distributed:
            dist.barrier()
        
        # Step 5 : Validation Loop
        if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
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
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                save_checkpoint(model, output_dir, "BEST")
                terminate = False
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    terminate=True
        
        # Early Stopping condition
        terminate = torch.tensor(terminate, dtype=torch.bool).cuda()
        if args.distributed:
            dist.broadcast(terminate, 0)
        if terminate:
            save_checkpoint(model, output_dir, epoch)
            
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
            save_checkpoint(model, output_dir, epoch)
                
    save_checkpoint(model, output_dir, epoch)
    print("Training Complete. Model saved.")
    if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
        print(f'It took {time() - start:.1f}s')
    if args.distributed:
        dist.destroy_process_group()
    
    
    return

if __name__ == '__main__':
    args = parse_args()
    
    print("Args: ",args)
    
    datamaps = load_json(os.path.join(args.data_path, args.dataset, 'datamaps.json'))
    
    item2id = datamaps['item2id']
    
    user_num = len(datamaps['user2id'])
    item_num = len(item2id)
    
    targetItems = readTargetItem(os.path.join(args.data_path, args.dataset, "targetItems.txt"))
    
    targetItems = [int(item2id[item]) for item in targetItems]
    print("# Target Items: ",len(targetItems))
    print("Sample Items: ",list(targetItems)[:5])
    
    if args.distributed:
        initialize_distributed()
    else:
        os.environ["LOCAL_RANK"] = "0"
    
    train_loader, train_dataset = get_BPRdataset_loader(dataset=args.dataset,data_path=args.data_path, mode='train',batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed)

    val_loader, val_dataset = get_BPRdataset_loader(dataset=args.dataset,data_path=args.data_path, mode='val',batch_size=args.batch_size, workers=args.num_workers, distributed=False, shuffle=True)

    user_num, item_num = max(train_dataset.user_num, val_dataset.user_num) + 1, max(train_dataset.item_num, val_dataset.item_num) + 1
    
    
    
    
    print("#training samples: ",len(train_dataset))
    print("#validation samples: ",len(val_dataset))
    
    print("#training batches: ",len(train_loader))
    print("#validation batches: ",len(val_loader))
    
    
    
    main(args, train_loader, val_loader, user_num, item_num, int(os.environ["LOCAL_RANK"]))

