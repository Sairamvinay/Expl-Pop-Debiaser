import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast
from tqdm import tqdm
import pickle
import os
import argparse
import numpy as np
from time import time
from packaging import version
from datetime import timedelta

from data_utils import load_pickle, save_pickle, load_json, readTargetItem
from utils import *
from test_dataset_BPR import get_eval_dataset_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for holistic explainer script")

    parser.add_argument("--output_dir",type=str,default='top-preds/',help='Path to save the model recommendation results')
    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--debug',action='store_true',help='Debug or not?')
    parser.add_argument("--num_workers",type=int,default=4,help='DataLoader Num Workers')
    
    parser.add_argument("--batch_size",type=int,default=100,help='Batch Size for evaluation')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument("--pos_path",type=str,default='top-preds/stage-1-POS-only-EXPLS/',help='Path to load ranked stage 1 results with positive explanations')
    parser.add_argument("--neg_path",type=str,default='top-preds/stage-1-NEG-only-EXPLS/',help='Path to load ranked stage 1 results with negative explanations')
    parser.add_argument("--zero_path",type=str,default='top-preds/stage-1-ZERO-EXPLS/',help='Path to load ranked stage 1 results with zero explanations')
    parser.add_argument("--top_k",type=int,default=10,help='Top-K for the re-ranking for rewarding the ALPHA Search')
    
    
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--num_batches_val', type=int, default=-1, help="Number of batches for validation (-1 for all)")
    
    parser.add_argument("--tau",type=float, default=0.3, help='Pop-Bias constraint for each user')
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')

    args = parser.parse_args()
    return args


def score(u,v,mapping):
    return round(mapping[u][v][0],5),mapping[u][v][1]

def get_top_k(mapping,user, k):
    lst = sorted(mapping[user].items(), key=lambda x:x[1][0],reverse=True)
    if k == -1:
        k = len(lst)
    return [x[0] for x in lst[:k]]

def map_sorted_preds_to_ranked_items(preds_file):
    ui_scores = preds_file['ui_scores']
    preds = preds_file['preds']
    
    assert isinstance(ui_scores, dict)
    assert isinstance(preds, list) or isinstance(preds, np.ndarray)

    mapped_results = {}
    user_ids = list(ui_scores.keys())
    num_candidates = len(next(iter(ui_scores.values())))  # assume uniform candidate size

    for idx, user_id in (enumerate(user_ids)):
        user_pred_scores = np.array(preds[idx * num_candidates : (idx + 1) * num_candidates])
        item_rank_dict = ui_scores[user_id]  # {item_id: -rank}
        
        # Invert to get: {-1: item_id1, -2: item_id2, ..., -100: item_id100}
        rank_to_item = {rank: item for item, rank in item_rank_dict.items()}
        
        # Sort predictions in descending order
        sorted_scores = np.sort(user_pred_scores)[::-1]
        
        # Map highest score to -1, second to -2, ..., lowest to -num_candidates
        user_item_score_map = {}
        for i in range(1, num_candidates + 1):
            rank = -i
            item_id = rank_to_item[rank]
            score = sorted_scores[i - 1]
            user_item_score_map[item_id] = (score,rank)
        
        mapped_results[user_id] = user_item_score_map

    return mapped_results


def grid_search(user, pos_ranks, neg_ranks, zero_ranks, targetItems, groundtruth_item, itemNum, TAU=0.3, num_items = 100, K = 10,debug=False):
    
    
    user_scores = []
    alphas = np.arange(0, 1.01, 0.05)  # Or finer grid
    best_alpha = 0
    best_scores = np.zeros((1,num_items))
    items = get_top_k(pos_ranks,user,-1) # can use any set for grabbing item candidate set
    ground_truth = {}
    ground_truth[user] = [groundtruth_item]
    user_scores = {}
    for alpha in alphas:
        ui_scores = {}
        pred_dict = {}
        for item in items:
            pos_score = score(user,item,pos_ranks)[0]
            neg_score = score(user,item,neg_ranks)[0]
            zero_score = score(user,item,zero_ranks)[0]
            diff = pos_score - neg_score
            scores = (alpha * diff) + ((1 - alpha) * zero_score)
            pred_dict[item] = scores # map item to their score
        
        obj = FairnessMetric({user:pred_dict},targetItems,top = [K], itemNum=itemNum)
        topK_results_all = obj.get_topK_recommendations(K) # {user: list of topK items}
        pr = obj.popularity_rate(topK_results_all)
        
        
        _,res = evaluate_all({user:pred_dict},ground_truth, topk = K,debug=False)
         
        NDCG = res['ndcg']
        HR = res['hit']
        
        if debug:
            print(f"Alpha={alpha}")
            print("topK_results_all (pre): ",topK_results_all)
            topK_results = list(topK_results_all.values())[0]
            print("topK results (pre): ",topK_results)
            print(f"Res: {res}")
            print(f"PR: {pr}")
            print("="*25)
            
        
        
        user_scores[alpha] = {'rec_list':pred_dict,'pr':pr,'ndcg':NDCG,
                              'hit':HR,'res':res}
    
    # Choose best alpha
    sorted_recs = sorted(user_scores.items(), key=lambda x:x[1]['ndcg'],reverse=True)
    max_rec = sorted_recs[0]
    sorted_recs = [rec for rec in sorted_recs if rec[1]['pr'] <= TAU]
    if sorted_recs:
        best_rec = sorted_recs[0]
    else:
        best_rec = max_rec # (0.0, user_scores[0.0])
    
    count = False
    if best_rec[1]['res']['hit'] > 0:
        count = True
    if debug:
        print("user: ",user)
        print(f"best rec alpha: {best_rec[0]}")
        print(f"res: {best_rec[1]['res']}")
        print(f"PR: {best_rec[1]['pr']}")
        top_res = list(best_rec[1]['rec_list'].values())
        print("top_res: ",top_res)
        with_scores = list(sorted(best_rec[1]['rec_list'].items(),key=lambda x:x[1], reverse=True))
        
        
        # print(f"topK with results: {with_scores[:K]}")
        
        for ranking_pos, lst in enumerate(with_scores[:K]):
            print(f"Rank {ranking_pos}: item: {lst}")

        # print(f"topK_results_all: {topK_results_all}")
        print(f"gt: {groundtruth_item}")
        # print(f"SCORED RECS: {sorted_recs}")
        # print("topKResults: ",topK_results)
        print("="*50)
    
    return best_rec,sorted_recs, int(count)


def main(args, new_pos_expls, new_neg_expls, zero_preds, user_num, item_num, local_rank):
    start = time()
    device = torch.device(f"cuda:{local_rank}")
    device_map={"": local_rank}
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    seedSet(args.seed)
    
    new_zero_expls = map_sorted_preds_to_ranked_items(zero_preds)

    early_stopping_patience = 3

    to_print = True
    
    
    all_info = []
    golds,preds = [],[]
    USER_ALPHAS = {}
    
    TAU = args.tau
    K = args.top_k
    
    os.makedirs(os.path.join(args.output_dir,args.dataset,f"Rerank-FindAlpha",f"TAU={TAU}-K={K}"),exist_ok=True)
    
    ui_scores = dict()
    gt = dict()
    total = [int(x) - 1 for x in list(range(1,user_num))] # it is already 1 + #users (check test_dataset_bpr.py)
    print("len(users): ",len(total))
    
    # users = np.random.choice(total,size=len(total),replace=False)
    users = total
    if args.debug:
        num_batches_val = 10
    else:
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(users)
        print(f"num_batches_val: {num_batches_val}\n")
    
    
    success = 0
    
    for stepv, user in tqdm(enumerate(users)):
        
        user = int(user)
        if stepv >= num_batches_val:
            break
        
        gold_item = int(zero_preds['gt'][user][0])
        
        best_rec,user_scores,found = grid_search(user = user, 
                    pos_ranks = new_pos_expls, 
                    neg_ranks = new_neg_expls, 
                    zero_ranks = new_zero_expls, 
                    targetItems = targetItems, 
                    groundtruth_item = gold_item,
                    itemNum = item_num, 
                    TAU=TAU, 
                    num_items = len(zero_preds['ui_scores'][user]), 
                    K = K,
                    debug = args.debug
                    )
        success += found
        USER_ALPHAS[user] = best_rec[0]
        scores_dict = best_rec[1]['rec_list']
        rerank_lst = sorted(scores_dict.items(),key = lambda x:x[1], reverse=True)
        gt[user] = [gold_item]
        pred_dict = {}

        for j in range(len(rerank_lst)):

            item, score = rerank_lst[j]
            pred_dict[item] = -(j + 1)
            label = int(gold_item == item)
            golds.append(label)
            preds.append(score)
        ui_scores[user] = pred_dict
        
    
    print("Success: ",success, ' out of ',num_batches_val)
        
    print("# golds: ",len(golds))
    print("# preds: ",len(preds))
    
    if not args.debug:
        save_path = os.path.join(args.output_dir,args.dataset,f"Rerank-FindAlpha",f"TAU={TAU}-K={K}",f"DEEPFM-{args.dataset}-preds.pkl")
        save_pickle({'ui_scores':ui_scores,'gt':gt, 'golds':golds, 'preds':preds, 'alphas':USER_ALPHAS},save_path)
    
    
    print("ATTACK UI SCORES: ",ui_scores)
    print("ATTACK GT SCORES: ",gt)
    top = [1,2,3,5,10,20]
    
    print("Recommendation Performance")
    _, Recommendresults = getBasicScores(ui_scores, gt, top)
    print("\nAUC: ",area_curve_metric(golds,preds)) 
    
    print("Fairness Performance")
    FairResults = getFairnessScores(ui_scores, targetItems, top, len(item2id))
    print("Evaluation Complete.")
    print(f'It took {time() - start:.1f}s')
    return {'recommend_results':Recommendresults, 'fair_results':FairResults}
    
    
    return

if __name__ == '__main__':
    args = parse_args()
    
    print("Args: ",args)
    
    pos_preds = load_pickle(os.path.join(args.pos_path,f"DEEPFM-{args.dataset}-preds.pkl"))
    neg_preds = load_pickle(os.path.join(args.neg_path,f"DEEPFM-{args.dataset}-preds.pkl"))
    zero_preds = load_pickle(os.path.join(args.zero_path, f"DEEPFM-{args.dataset}-preds.pkl"))
    
    datamaps = load_json(os.path.join(args.data_path, args.dataset, 'datamaps.json'))
    
    item2id = datamaps['item2id']
    
    user_num = len(datamaps['user2id'])
    item_num = len(item2id)
    
    targetItems = readTargetItem(os.path.join(args.data_path, args.dataset, "targetItems.txt"))
    
    targetItems = [int(item2id[item]) for item in targetItems]
    print("# Target Items: ",len(targetItems))
    print("Sample Items: ",list(targetItems)[:5])
    
    expls = None
    
    _, eval_dataset = get_eval_dataset_loader(datamaps = datamaps, targetItems = targetItems, user_num = user_num, item_num = item_num, mode='test',batch_size=args.batch_size, dataset = args.dataset, data_path = args.data_path, expls = expls, workers=args.num_workers, shuffle=False)
    
    user_num, item_num = eval_dataset.user_num, eval_dataset.item_num - 1
    
    print("#testing samples: ",len(eval_dataset))
    print("#testing batches: ",len(_))
    
    del eval_dataset
    del _
    
    
    pos_expl_preds, neg_expl_preds = map_sorted_preds_to_ranked_items(pos_preds), map_sorted_preds_to_ranked_items(neg_preds)
    
    main(args, pos_expl_preds, neg_expl_preds, zero_preds, user_num, item_num, int(args.local_rank))

