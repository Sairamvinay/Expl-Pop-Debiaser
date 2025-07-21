import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from model_utils import DeepFM, get_explanation_embedding, bpr_loss
from utils import seedSet,print_trainable_parameters,save_checkpoint,load_checkpoint,verify_model_weights
from train_dataset_BPR import get_BPRdataset_loader
from train_dataset import get_dataset_loader as getExplBPRdataset_loader
from train_dataset_stage2 import get_dataset_loader as getStage2dataset_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for holistic explainer script")

    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--debug',action='store_true',help='Debug or not?')
    parser.add_argument("--num_workers",type=int,default=4,help='DataLoader Num Workers')
    
    parser.add_argument('--epochs',type=int,default=5,help='Number of Training Epochs')
    parser.add_argument("--batch_size",type=int,default=16,help='Batch Size for training')
    parser.add_argument('--id_embed_dim',type=int,default=64,help='DeepFM ID Embedding Dimension')
    parser.add_argument("--hidden_dim",type=str, default='[256,128,64]', help='DeepFM Hidden Dimension List: provide as \"[256,128,64]\"')
    parser.add_argument("--dropout",type=float,default=0.2,help = 'Dropout for DeepFM layers')
    
    parser.add_argument("--learning_rate",type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--distributed', action='store_true', help='Distributed training flag')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimization')
    parser.add_argument('--warmup_ratio', type=float, required=True, help='Warmup ratio for learning rate schedule')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument("--temperature",type=float, default=5.0, help='Temperature for Scaling prior to sigmoid')
    parser.add_argument('--stage',type=int, default=1, help='Which training stage: 0 (Utility training) ; 1(Explanation pairwise) or 2(Fairness Disparity)') 

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument("--expl_path", type=str, default='generated-expls/', help='Directory to load the Explanation generated dataset: such as generated-expls/ which has files in the following directory: {dataset}/{model_name}-preds.pkl')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument("--output_dir",type=str,default='snap/',help='Path to save the model checkpoints')
    parser.add_argument('--num_batches_train', type=int, default=-1, help="Number of batches for training (-1 for all)")
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument("--checkpoint",type=str, default=None, help = 'Load stage 1 checkpoints for stage 2 training and stage 0 checkpoints for stage 1 training')
    
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
    AUTH_TOKEN = "YOUR_HF_TOKEN"

    encoder = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,token=AUTH_TOKEN,device_map=device_map,output_hidden_states=True).eval()
    
    # =============================================================
    # Step 2. Basic Tokenizer Setup
    # =============================================================
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=AUTH_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left' # ADDED BY ME to avoid error for decoder-only model
    
    model = DeepFM(user_num = user_num, item_num = item_num, id_embed_dim = args.id_embed_dim, expl_embed_dim = encoder.config.hidden_size, hidden_dims=eval(args.hidden_dim), dropout_rate = args.dropout).to(device)
    
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
    
    
    check_step = 1
    
    output_dir = os.path.join(args.output_dir, args.dataset, f"stage-{args.stage}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer,lr_scheduler = create_opt_lr(model,num_batches_train,args.gradient_accumulation_steps,args.epochs,args.warmup_ratio,args.learning_rate,args.weight_decay,adam_eps=1e-6)
    
    scaler = GradScaler()
    
    early_stopping_counter = 0
    terminate = False
    to_print = True
    EPSILON = 1e-6
    
    if (not args.distributed) or (args.distributed and dist.get_rank() == 0):
        best_val_loss = float('inf')
    
    if args.distributed:
        dist.barrier()
        print(f"🔍 Model class at training start: {type(model.module)}")
    
    if args.stage == 2:
        disp_ratio = torch.tensor(1/9).to(device)
        
        
        if args.checkpoint:
            model = load_checkpoint(model, args.checkpoint)
            verify_model_weights(model)
        
    if args.stage == 1:
        if args.checkpoint:
            model = load_checkpoint(model, args.checkpoint)
            verify_model_weights(model)
        
        if args.distributed:
            model.module.freeze_components(["user_embed", "item_embed"])
        else:
            model.freeze_components(["user_embed", "item_embed"])
    
    
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
            
                
                
                try:
                    if args.stage == 0:
                        user_ids = torch.tensor(batch['UserID']).to(device)
                        pos_item_ids = torch.tensor(batch['PosItem']).to(device)
                        neg_item_ids = torch.tensor(batch['NegItem']).to(device)

                        zero_emb = torch.zeros((len(user_ids), encoder.config.hidden_size), dtype=torch.float16).to(device) 
                
                        if args.distributed:
                            interact_score = model.module(user_ids = user_ids, item_ids = pos_item_ids, expl_embeds = zero_emb)
                            uninteract_score = model.module(user_ids = user_ids, item_ids = neg_item_ids, expl_embeds = zero_emb)
                        else:
                            interact_score = model(user_ids = user_ids, item_ids = pos_item_ids, expl_embeds = zero_emb)
                            uninteract_score = model(user_ids = user_ids, item_ids = neg_item_ids, expl_embeds = zero_emb)

                        loss = bpr_loss(interact_score, uninteract_score)
                        msg = ""
                        if args.debug or step == 0:
                            print("user ids: ",user_ids, end = ' ,')
                            print("pos item ids: ",pos_item_ids, end = ' ,')
                            print("neg item ids: ",neg_item_ids, end = ' ,')

                            print("pos_score: ",interact_score, end = ' ,')
                            print("neg_score: ",uninteract_score, end = ' ,')
                            print("loss: ",loss.item(),end= ' ,')


                    elif args.stage == 1:
                        user_ids = torch.tensor(batch['UserID']).to(device)
                        item_ids = torch.tensor(batch['ItemID']).to(device)
                        pos_emb = get_explanation_embedding(encoder,tokenizer,batch['pos-expl'],device)
                        neg_emb = get_explanation_embedding(encoder,tokenizer,batch['neg-expl'],device)
                        label = torch.tensor(batch['label']).to(device)
                        
                        if args.distributed:
                            posexplscore = model.module(user_ids = user_ids, item_ids = item_ids, expl_embeds = pos_emb,mild_factor_scale=1/args.temperature)
                            negexplscore = model.module(user_ids = user_ids, item_ids = item_ids, expl_embeds = neg_emb,mild_factor_scale=1/args.temperature)
                        
                        else:
                            posexplscore = model(user_ids = user_ids, item_ids = item_ids, expl_embeds = pos_emb,mild_factor_scale=1/args.temperature)
                            negexplscore = model(user_ids = user_ids, item_ids = item_ids, expl_embeds = neg_emb,mild_factor_scale=1/args.temperature)
                        
                        loss = bpr_loss(posexplscore, negexplscore)
                        msg = ""
                        if args.debug or step == 0:
                            print("user ids: ",user_ids, end = ' ,')
                            print("item ids: ",item_ids, end = ' ,')
                            print("pos expls : ",batch['pos-expl'], end = ' ,')
                            print("neg expls : ",batch['neg-expl'], end = ' ,')
                            print("labels : ",label,end= ' ,')
                            print("pos_score: ",posexplscore, end = ' ,')
                            print("neg_score: ",negexplscore, end = ' ,')
                            print("loss: ",loss.item(),end= ' ,')
                        
                    elif args.stage == 2:
                        user_ids = torch.tensor(batch['UserID']).to(device)
                        item_ids = torch.tensor(batch['ItemID']).to(device)
                        pos_emb = get_explanation_embedding(encoder,tokenizer,batch['pos-expl'],device)
                        neg_emb = get_explanation_embedding(encoder,tokenizer,batch['neg-expl'],device)
                        pop_labels = torch.tensor(batch['pop-label']).to(device)

                        # Select appropriate explanation embeddings
                        expl_embeds = torch.where(pop_labels.unsqueeze(1) == 1, neg_emb, pos_emb)
                        del pos_emb
                        del neg_emb

                        # Forward pass
                        if args.distributed:
                            preds = model.module(user_ids, item_ids, expl_embeds)
                        else:
                            preds = model(user_ids, item_ids, expl_embeds)
                        
                        pred_scores = preds / args.temperature
                        pred_scores = torch.sigmoid(pred_scores).squeeze()

                        # Separate scores
                        pop_scores = pred_scores[pop_labels == 1]
                        niche_scores = pred_scores[pop_labels == 0]

                        if pop_scores.numel() == 0 or niche_scores.numel() == 0:
                            fair_loss = torch.tensor(1.0, device=device, requires_grad=True)

                        else:

                            # Disparity loss (Eq. 16 + 17 from Li et al. 2022)
                            numerator = (pop_scores.sum() - disp_ratio * niche_scores.sum())
                            denominator = (pred_scores.sum() + EPSILON)
                            fair_loss = (numerator / denominator)

                        loss = fair_loss ** 2
                        msg = ""
                        if args.debug or step == 0:
                            print("user ids: ",user_ids, end = ' ,')
                            print("item ids: ",item_ids, end = ' ,')
                            print("pop_labels: ",pop_labels, end= ' ,')
                            print("pred scores before temperature scale: ",preds)
                            print("pred after temperature and sigmoiding: ",pred_scores)
                            print("expl embeds: ",expl_embeds, end= ' ,')
                            print("pop scores: ",pop_scores, end = ' ,')
                            print("niche scores: ",niche_scores, end = ' ,')
                            print("fair_loss: ",fair_loss.item(),end= ' ,')
                            print("total loss: ",loss.item(),end=' ,')

                    else:
                        raise NotImplementedError
                
                except Exception:
                    print("Batch errored: ",batch)

            
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
                        
                        if args.stage == 0:
                            user_ids = torch.tensor(batchv['UserID']).to(device)
                            pos_item_ids = torch.tensor(batchv['PosItem']).to(device)
                            neg_item_ids = torch.tensor(batchv['NegItem']).to(device)
                        
                            zero_emb = torch.zeros((len(user_ids), encoder.config.hidden_size), dtype=torch.float16).to(device) 
                        
                            interact_score = model(user_ids = user_ids, item_ids = pos_item_ids, expl_embeds = zero_emb)
                            uninteract_score = model(user_ids = user_ids, item_ids = neg_item_ids, expl_embeds = zero_emb)
                            val_loss = bpr_loss(interact_score, uninteract_score)
                            

                        elif args.stage == 1:
                            user_ids = torch.tensor(batchv['UserID']).to(device)
                            item_ids = torch.tensor(batchv['ItemID']).to(device)
                            pos_emb = get_explanation_embedding(encoder,tokenizer,batchv['pos-expl'],device)
                            neg_emb = get_explanation_embedding(encoder,tokenizer,batchv['neg-expl'],device)
                            label = torch.tensor(batchv['label']).to(device)

                            posexplscore = model(user_ids = user_ids, item_ids = item_ids, expl_embeds = pos_emb)
                            negexplscore = model(user_ids = user_ids, item_ids = item_ids, expl_embeds = neg_emb)
                            val_loss = bpr_loss(posexplscore, negexplscore)

                        elif args.stage == 2:
                            user_ids = torch.tensor(batchv['UserID']).to(device)
                            item_ids = torch.tensor(batchv['ItemID']).to(device)
                            pos_emb = get_explanation_embedding(encoder,tokenizer,batchv['pos-expl'],device)
                            neg_emb = get_explanation_embedding(encoder,tokenizer,batchv['neg-expl'],device)
                            pop_label = torch.tensor(batchv['pop-label']).to(device)
                    
                            # Select appropriate explanation embeddings
                            expl_embeds = torch.where(pop_label.unsqueeze(1) == 1, neg_emb, pos_emb)
                            del pos_emb
                            del neg_emb
                            # Forward pass

                            if args.distributed:
                                preds = model.module(user_ids, item_ids, expl_embeds)
                            else:
                                preds = model(user_ids, item_ids, expl_embeds)
                            
                            pred_scores = preds / args.temperature
                            pred_scores = torch.sigmoid(pred_scores).squeeze()

                            # Separate scores
                            pop_scores = pred_scores[pop_label == 1]
                            niche_scores = pred_scores[pop_label == 0]

                            if pop_scores.numel() == 0 or niche_scores.numel() == 0:
                                fair_loss = torch.tensor(1.0, device=device, requires_grad=True)

                            else:
                                # Disparity loss (Eq. 16 + 17 from Li et al. 2022)
                                numerator = (pop_scores.sum() - disp_ratio * niche_scores.sum())
                                denominator = (pred_scores.sum() + EPSILON)
                                fair_loss = (numerator / denominator)

                            val_loss = fair_loss ** 2
                        
                        else:
                            raise NotImplementedError

                    torch.cuda.empty_cache()
                    total_val_loss += val_loss.item()
                
            avg_val_loss = total_val_loss / max(1,num_batches_val)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            
            if (avg_val_loss + EPSILON) < best_val_loss:
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
    
    if args.stage == 0:
        train_loader, train_dataset = get_BPRdataset_loader(dataset=args.dataset,data_path=args.data_path, mode='train',batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed)

        val_loader, val_dataset = get_BPRdataset_loader(dataset=args.dataset,data_path=args.data_path, mode='val',batch_size=args.batch_size, workers=args.num_workers, distributed=False, shuffle=True)
        
        user_num, item_num = max(train_dataset.user_num, val_dataset.user_num) + 1, max(train_dataset.item_num, val_dataset.item_num) + 1
    
    elif args.stage == 1:
        TRAIN_SPLIT = 0.8
        expls = load_pickle(os.path.join(args.expl_path, args.dataset, "train.pkl"))
        # 1. Convert keys to a NumPy array
        keys = np.array(list(expls.keys()))
        num_train = int(TRAIN_SPLIT * len(keys))

        # 2. Shuffle and split indices
        shuffled_indices = np.random.permutation(len(keys))
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:]

        # 3. Get the corresponding keys
        train_keys = keys[train_indices]
        val_keys = keys[val_indices]

        train_keys = [tuple(k) for k in train_keys]
        val_keys = [tuple(k) for k in val_keys]

        # 4. Build new dicts
        train_expls = {k: expls[k] for k in train_keys}
        val_expls = {k: expls[k] for k in val_keys}
        
        item2id = datamaps['item2id']
    
        USER_NUM = len(datamaps['user2id'])
        ITEM_NUM = len(item2id)
        
        train_loader, train_dataset = getExplBPRdataset_loader(expls = train_expls, targetItems = targetItems, user_num = USER_NUM, item_num = ITEM_NUM, batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed, mode='train')
    
        val_loader, val_dataset = getExplBPRdataset_loader(expls = val_expls, targetItems = targetItems, user_num = USER_NUM, item_num = ITEM_NUM, batch_size=args.batch_size, workers=args.num_workers, distributed=False, shuffle=True,mode='val')
        
        user_num, item_num = max(train_dataset.user_num, val_dataset.user_num), max(train_dataset.item_num, val_dataset.item_num)
    
    elif args.stage == 2:
        TRAIN_SPLIT = 0.8
        expls = load_pickle(os.path.join(args.expl_path, args.dataset, "train.pkl"))
        # 1. Convert keys to a NumPy array
        keys = np.array(list(expls.keys()))
        num_train = int(TRAIN_SPLIT * len(keys))

        # 2. Shuffle and split indices
        shuffled_indices = np.random.permutation(len(keys))
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:]

        # 3. Get the corresponding keys
        train_keys = keys[train_indices]
        val_keys = keys[val_indices]

        train_keys = [tuple(k) for k in train_keys]
        val_keys = [tuple(k) for k in val_keys]

        # 4. Build new dicts
        train_expls = {k: expls[k] for k in train_keys}
        val_expls = {k: expls[k] for k in val_keys}
        
        item2id = datamaps['item2id']
    
        USER_NUM = len(datamaps['user2id'])
        ITEM_NUM = len(item2id)
        
        train_loader, train_dataset = getStage2dataset_loader(expls = train_expls, targetItems = targetItems, user_num = USER_NUM, item_num = ITEM_NUM, batch_size=args.batch_size, workers=args.num_workers, distributed=args.distributed, mode='train')
    
        val_loader, val_dataset = getStage2dataset_loader(expls = val_expls, targetItems = targetItems, user_num = USER_NUM, item_num = ITEM_NUM, batch_size=args.batch_size, workers=args.num_workers, distributed=False, shuffle=True,mode='val')
        
        user_num, item_num = max(train_dataset.user_num, val_dataset.user_num), max(train_dataset.item_num, val_dataset.item_num)        
    
    
    
    print("#training samples: ",len(train_dataset))
    print("#validation samples: ",len(val_dataset))
    
    print("#training batches: ",len(train_loader))
    print("#validation batches: ",len(val_loader))
    
    
    
    main(args, train_loader, val_loader, user_num, item_num, int(os.environ["LOCAL_RANK"]))

