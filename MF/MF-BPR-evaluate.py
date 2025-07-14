import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import pickle
import os
import argparse
from time import time
from packaging import version
from datetime import timedelta

from data_utils import load_pickle, save_pickle, load_json, readTargetItem
from model_utils import MatrixFactorization, bpr_loss
from utils import seedSet,load_checkpoint,getBasicScores,getFairnessScores,area_curve_metric, verify_model_weights
from test_dataset_BPR import get_eval_dataset_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for MF Eval script")

    parser.add_argument("--output_dir",type=str,default='top-preds/',help='Path to save the model recommendation results')
    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--debug',action='store_true',help='Debug or not?')
    parser.add_argument("--num_workers",type=int,default=4,help='DataLoader Num Workers')
    
    parser.add_argument("--batch_size",type=int,default=100,help='Batch Size for evaluation')
    parser.add_argument('--id_embed_dim',type=int,default=64,help='MatrixFactorization ID Embedding Dimension')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument("--checkpoint",type=str, default=None, help = 'Load checkpoints')
    parser.add_argument("--temperature",type=float,default=1.0,help = 'Temperature for sigmoid')
    parser.add_argument("--softmax",action='store_true',help='Softmax or not (sigmoid is default)?')
    
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')

    args = parser.parse_args()
    return args



def main(args, eval_loader, user_num, item_num, local_rank):
    start = time()
    device = torch.device(f"cuda:{local_rank}")
    device_map={"": local_rank}
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    seedSet(args.seed)
    
    os.makedirs(args.output_dir,exist_ok=True)
    early_stopping_patience = 3
    
    
    model = MatrixFactorization(user_num = user_num, item_num = item_num, embedding_size = args.id_embed_dim).to(device)
    
    model = model.to(f"cuda:{local_rank}")
    
    if args.debug:
        num_batches_val = 10
    else:
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(eval_loader)
        print(f"num_batches_val: {num_batches_val}\n")
    
    to_print = True
    
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
        verify_model_weights(model)
        
    model.eval()
    all_info = []
    golds,preds = [],[]
    
    m = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        for stepv, batchv in enumerate(tqdm(eval_loader)):
            if stepv > num_batches_val:
                break
            torch.cuda.empty_cache()
            with autocast():
                user_ids = torch.tensor(batchv['UserID']).to(device)
                item_ids = torch.tensor(batchv['ItemID']).to(device)
                labels = torch.tensor(batchv['label'],dtype=torch.float16).to(device)
                pos_item = batchv['pos-item']
                
                
                # Forward pass
                logits = model(user_ids, item_ids).squeeze()
                logits /= args.temperature

                if args.softmax:
                    pred_scores = torch.tensor(logits,dtype=torch.float32).softmax(dim=-1)
                else:
                    pred_scores = torch.sigmoid(logits)
                
                _, indices = torch.sort(pred_scores, descending=True)
                
                if args.debug:

                    print("User: ",user_ids)
                    print("Item: ",item_ids)
                    print("Pos items: ",pos_item)
                    print("input user size:",user_ids.shape , ' items shape: ',item_ids.shape)
                    print("labels shape: ",labels.shape)
                    if args.softmax:
                        print("preds before softmax: ",logits.shape, ' ',logits)
                        print("preds after softmax: ",pred_scores)
                    else:
                        print("preds before sigmoid: ",logits.shape, ' ',logits)
                        print("preds after sigmoid: ",pred_scores)
                    
                    print("Sorted with dimension 0: ",_)
                    print("Indices with dimension 0: ",indices)
                    print("="*100)
                else:
                    pass                


            torch.cuda.empty_cache()
            del user_ids
            del batchv
            new_info = {}
            new_info['target_item'] = [pos_item[0]] # same for entire batch
            new_info['gen_item_list'] = [item_ids[_] for _ in indices[:args.batch_size]]
            golds.extend([int(labels[_]) for _ in range(args.batch_size)])
            preds.extend(pred_scores.float().cpu().tolist())
            
            all_info.append(new_info)
                
                
    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info['target_item'][0])]
        pred_dict = {}
        for j in range(len(info['gen_item_list'])):
            try:
                pred_dict[int(info['gen_item_list'][j])] = -(j+1)
            except:
                pass
        ui_scores[i] = pred_dict
    
    
    print("# golds: ",len(golds))
    print("# preds: ",len(preds))
    
    print("ATTACK UI SCORES: ",ui_scores)
    print("ATTACK GT SCORES: ",gt)
    
    
    if not args.debug:
        save_path = os.path.join(args.output_dir,f"MF-BPR-{args.dataset}-preds.pkl")
        save_pickle({'ui_scores':ui_scores,'gt':gt, 'golds':golds, 'preds':preds},save_path)
    
    top = [1,2,3,5,10,20]
    
    print("Recommendation Performance")
    _, Recommendresults = getBasicScores(ui_scores, gt, top)
    print("\nAUC: ",area_curve_metric(golds,preds)) 
    
    print("Fairness Performance")
    FairResults = getFairnessScores(ui_scores, targetItems, top,len(item2id))
    print("Evaluation Complete.")
    print(f'It took {time() - start:.1f}s')
    return {'recommend_results':Recommendresults, 'fair_results':FairResults}
    
    
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
    
    expls = None
    
    eval_loader, eval_dataset = get_eval_dataset_loader(datamaps = datamaps, targetItems = targetItems, user_num = user_num, item_num = item_num, mode='test',batch_size=args.batch_size, dataset = args.dataset, data_path = args.data_path, expls = expls, workers=args.num_workers, shuffle=False)
    
    user_num, item_num = eval_dataset.user_num, eval_dataset.item_num
    
    print("#testing samples: ",len(eval_dataset))
    print("#testing batches: ",len(eval_loader))
    
    
    main(args, eval_loader, user_num, item_num, int(args.local_rank))

