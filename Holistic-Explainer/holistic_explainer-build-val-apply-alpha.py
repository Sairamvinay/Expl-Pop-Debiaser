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
from model_utils import DeepFM, get_explanation_embedding
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
    parser.add_argument("--expl_path",type=str,default='generated-expls/',help='Path to load explanations: effective from stage 1')
    
    
    parser.add_argument('--id_embed_dim',type=int,default=64,help='DeepFM ID Embedding Dimension')
    parser.add_argument("--hidden_dim",type=str, default='[256,128,64]', help='DeepFM Hidden Dimension List: provide as \"[256,128,64]\"')
    parser.add_argument("--dropout",type=float,default=0.2,help = 'Dropout for DeepFM layers')
    
    
    parser.add_argument("--checkpoint",type=str, default=None, help = 'Load stage 1 checkpoints')
    parser.add_argument("--pos_temperature",type=float,default=1.0,help = 'Positive Temperature for sigmoid')
    parser.add_argument("--neg_temperature",type=float,default=1.0,help = 'Negative Temperature for sigmoid')
    parser.add_argument("--zero_temperature",type=float,default=1.0,help = 'Zero Temperature for sigmoid')
    
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

def map_sorted_preds_to_ranked_items(preds_file,scores_only=False):
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
            user_item_score_map[item_id] = score if scores_only else (score,rank)
        
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
    
    best_alpha_results = sorted_recs[0] # greedy algorithm to pick max NDCG in case no possible answer
    
    sorted_recs = [rec for rec in sorted_recs if rec[1]['pr'] <= TAU]
    if sorted_recs:
        best_rec = sorted_recs[0]
    else:
        best_rec = best_alpha_results # (0.0, user_scores[0.0])
    
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
        
        for index, lst in enumerate(with_scores[:K]):
            print(f"Rank {index}: item: {lst}")

        # print(f"topK_results_all: {topK_results_all}")
        print(f"gt: {groundtruth_item}")
        # print(f"SCORED RECS: {sorted_recs}")
        # print("topKResults: ",topK_results)
        print("="*50)
    
    return best_rec,sorted_recs, int(count)


def run_validation(args, new_pos_expls, new_neg_expls, zero_preds, user_num, item_num, local_rank):
    start_time = time()
    
    new_zero_expls = map_sorted_preds_to_ranked_items(zero_preds)

    early_stopping_patience = 3

    to_print = True
        
    TAU = args.tau
    K = args.top_k
        
    total = [int(x) - 1 for x in list(range(1,user_num))] # it is already 1 + #users (check test_dataset_bpr.py)
    print("len(users): ",len(total))
    
    users = total
    if args.debug:
        num_batches_val = 100
    else:
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(users)
        print(f"num_batches_val: {num_batches_val}\n")
    
    
    success = 0
    
    user_alphas = {}
    
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
        
        user_alphas[user] = best_rec[0]
        
        success += found
        
        
    print("Success: ",success, ' out of ',num_batches_val)
        
    
    if not args.debug:
        save_path = os.path.join(args.output_dir,args.dataset,f"RerankTest-ValFindAlpha",f"TAU={TAU}-K={K}",f"User-alphas-{args.dataset}.pkl")
        save_pickle(user_alphas,save_path)
    
    
    
    print("Running Alpha Grid Search Complete.")
    print(f'It took {time() - start_time:.1f}s')
    
    return user_alphas


def compute_val_preds(args,eval_loader,user_num, item_num,val_items,test_items,pos_test_preds, neg_test_preds, zero_test_preds,train_expls,local_rank):
    """
    Compute validation preds by recomputing val item scores for POS/NEG/ZERO expls,
    and combining with negatives from the test preds.
    
    Returns:
        pos_val, neg_val, zero_val
        each is dict {'ui_scores':..., 'gt':..., 'golds':..., 'preds':...}
    """

    device = torch.device(f"cuda:{local_rank}")
    device_map={"": local_rank}
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    seedSet(args.seed)
    
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # =============================================================
    # Step 2. Basic Tokenizer Setup
    # =============================================================
    AUTH_TOKEN = "YOUR_HF_TOKEN"
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=AUTH_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left' # ADDED BY ME to avoid error for decoder-only model
    
    encoder = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,use_auth_token=AUTH_TOKEN,device_map=device_map,output_hidden_states=True).eval()
    
    model = DeepFM(user_num = user_num, item_num = item_num + 1, id_embed_dim = args.id_embed_dim, expl_embed_dim = encoder.config.hidden_size, hidden_dims=eval(args.hidden_dim), dropout_rate = args.dropout).to(device)
    
    model = model.to(f"cuda:{local_rank}")
    
    if args.debug:
        num_batches_val = 100
    else:
        num_batches_val = args.num_batches_val if args.num_batches_val > -1 else len(eval_loader)
        print(f"num_batches_val: {num_batches_val}\n")
    
    to_print = True
    
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
        verify_model_weights(model)
    
    model.eval()
    
    # Separate containers for pos/neg/zero
    pos_ui, neg_ui, zero_ui = {}, {}, {}
    pos_gt, neg_gt, zero_gt = {}, {}, {}
    pos_golds, neg_golds, zero_golds = [], [], []
    pos_preds, neg_preds, zero_preds = [], [], []

    with torch.no_grad():
        for stepv, batchv in enumerate(tqdm(eval_loader)):
            if stepv >= num_batches_val:
                break
            torch.cuda.empty_cache()
            with autocast():
                u = int(batchv['UserID'][0])
                val_item = val_items[u]
                test_item = test_items[u]

                # fresh forward passes for val item
                def _get_emb(expl_type):
                    if expl_type == 'pos':
                        return get_explanation_embedding(encoder, tokenizer,
                                                         train_expls[(u,val_item)]['pos-expl'], device)
                    elif expl_type == 'neg':
                        return get_explanation_embedding(encoder, tokenizer,
                                                         train_expls[(u,val_item)]['neg-expl'], device)
                    elif expl_type == 'zero':
                        return torch.zeros((1, encoder.config.hidden_size),
                                           dtype=torch.float16).to(device)

                u_t = torch.tensor([u]).to(device)
                v_t = torch.tensor([val_item]).to(device)

                val_score_pos = torch.sigmoid(model(u_t, v_t, _get_emb('pos'),mild_factor_scale=1/args.pos_temperature)).item()
                val_score_neg = torch.sigmoid(model(u_t, v_t, _get_emb('neg'),mild_factor_scale=1/args.neg_temperature)).item()
                val_score_zero = torch.sigmoid(model(u_t, v_t, _get_emb('zero'),mild_factor_scale=1/args.zero_temperature)).item()

            # reuse negatives
            neg_items = list(zero_test_preds[u].keys())
            if val_item in neg_items:
                neg_items.remove(val_item)
            
            if test_item in neg_items:
                neg_items.remove(test_item)

            # POS dict
            pos_dict = {it: pos_test_preds[u][it] for it in neg_items}
            pos_dict[val_item] = val_score_pos
            pos_sorted = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)
            pos_ui[u] = {it: -(i+1) for i,(it,_) in enumerate(pos_sorted)}
            pos_gt[u] = [val_item]
            for it, sc in pos_sorted:
                pos_preds.append(sc)
                pos_golds.append(int(it == val_item))

            # NEG dict
            neg_dict = {it: neg_test_preds[u][it] for it in neg_items}
            neg_dict[val_item] = val_score_neg
            neg_sorted = sorted(neg_dict.items(), key=lambda x: x[1], reverse=True)
            neg_ui[u] = {it: -(i+1) for i,(it,_) in enumerate(neg_sorted)}
            neg_gt[u] = [val_item]
            for it, sc in neg_sorted:
                neg_preds.append(sc)
                neg_golds.append(int(it == val_item))

            # ZERO dict
            zero_dict = {it: zero_test_preds[u][it] for it in neg_items}
            zero_dict[val_item] = val_score_zero
            zero_sorted = sorted(zero_dict.items(), key=lambda x: x[1], reverse=True)
            zero_ui[u] = {it: -(i+1) for i,(it,_) in enumerate(zero_sorted)}
            zero_gt[u] = [val_item]
            for it, sc in zero_sorted:
                zero_preds.append(sc)
                zero_golds.append(int(it == val_item))

    pos_val = {'ui_scores': pos_ui, 'gt': pos_gt,
               'golds': pos_golds, 'preds': pos_preds}
    neg_val = {'ui_scores': neg_ui, 'gt': neg_gt,
               'golds': neg_golds, 'preds': neg_preds}
    zero_val= {'ui_scores': zero_ui,'gt': zero_gt,
               'golds': zero_golds,'preds': zero_preds}
    
    print("# pos-golds: ",len(pos_golds))
    print("# pos-preds: ",len(pos_preds))
    
    print("# neg-golds: ",len(neg_golds))
    print("# neg-preds: ",len(neg_preds))
    
    print("# zero-golds: ",len(zero_golds))
    print("# zero-preds: ",len(zero_preds))
    
    if not args.debug:
        save_path = os.path.join(args.output_dir,args.dataset,f"RerankTest-ValFindAlpha",f"DEEPFM-{args.dataset}-Val-preds.pkl")
        dict_out = {'pos-val':pos_val,'neg-val':neg_val,'zero_val':zero_val}
        save_pickle(dict_out,save_path)
        print("Saved computed val-preds preds:", save_path)
    
    

    return pos_val, neg_val, zero_val
    
def apply_alphas_on_test(args, TAU, K, targetItems, item2id, pos_preds, neg_preds, zero_preds, test_gt, user_alphas):
    """Apply learned alphas on test set to rerank items."""
    
    ui_scores, golds, preds = {}, [], []

    for user in tqdm(test_gt.keys(), desc="Apply alpha on test"):
        alpha = user_alphas.get(user, 0.0)
        cand_items = list(pos_preds["ui_scores"][user].keys())
        pred_dict = {}
        for item in cand_items:
            pos_score = pos_preds["ui_scores"][user][item]
            neg_score = neg_preds["ui_scores"][user][item]
            zero_score = zero_preds["ui_scores"][user][item]
            final_score = alpha * (pos_score - neg_score) + (1 - alpha) * zero_score
            pred_dict[item] = final_score
        rerank = sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)
        ui_scores[user] = {it: -(i+1) for i,(it,_) in enumerate(rerank)}
        for it, sc in rerank:
            preds.append(sc)
            golds.append(int(it == test_gt[user][0]))

    dict_out = {"ui_scores": ui_scores, "gt": test_gt,
           "golds": golds, "preds": preds}

    print("# golds: ",len(golds))
    print("# preds: ",len(preds))
    
    if not args.debug:
        save_path = os.path.join(args.output_dir,args.dataset,f"RerankTest-ValFindAlpha",f"TAU={TAU}-K={K}",f"DEEPFM-{args.dataset}-preds.pkl")
        save_pickle(dict_out,save_path)
        print("Saved reranked test preds:", save_path)
    
    
    print("ATTACK UI SCORES: ",ui_scores)
    print("ATTACK GT SCORES: ",test_gt)
    top = [1,2,3,5,10,20]
    
    print("Recommendation Performance")
    _, Recommendresults = getBasicScores(ui_scores, test_gt, top)
    print("\nAUC: ",area_curve_metric(golds,preds)) 
    
    print("Fairness Performance")
    FairResults = getFairnessScores(ui_scores, targetItems, top, len(item2id))
    print("Evaluation Complete.")
    
    return {'recommend_results':Recommendresults, 'fair_results':FairResults}
    
    
def main(args,eval_loader,user_num, item_num,val_items,test_items,targetItems,item2id, pos_preds, neg_preds, zero_preds, local_rank):
    
    start = time()
     # --- Step 1: Build validation preds for POS/NEG/ZERO ---
    pos_test_scores, neg_test_scores, zero_test_scores = map_sorted_preds_to_ranked_items(pos_preds,scores_only=True), map_sorted_preds_to_ranked_items(neg_preds,scores_only=True), map_sorted_preds_to_ranked_items(zero_preds,scores_only=True)
    
    
    TAU = args.tau
    K = args.top_k
    
    alpha_path = os.path.join(args.output_dir,args.dataset,f"RerankTest-ValFindAlpha",f"TAU={TAU}-K={K}",f"User-alphas-{args.dataset}.pkl")
    
    if os.path.exists(alpha_path):
        print("Loading from saved Alphas file at :",alpha_path)
        user_alphas = load_pickle(alpha_path)
    else:
        os.makedirs(os.path.join(args.output_dir, args.dataset,f"RerankTest-ValFindAlpha",f"TAU={TAU}-K={K}"),exist_ok=True)
        
        val_pred_path = os.path.join(args.output_dir, args.dataset,f"RerankTest-ValFindAlpha",f"DEEPFM-{args.dataset}-Val-preds.pkl")
        if os.path.exists(val_pred_path):
            print("Loading saved precomputed scores from: ",val_pred_path)
            dict_out = load_pickle(val_pred_path)
            pos_val_preds, neg_val_preds, zero_val_preds = dict_out['pos-val'],dict_out['neg-val'],dict_out['zero_val']
            
        
        else:
            print("Val pred building starts...")

            # compute score for val dataset
            pos_val_preds, neg_val_preds, zero_val_preds = compute_val_preds(args=args,eval_loader=eval_loader,user_num=user_num, item_num=item_num,val_items=val_items, test_items=test_items,pos_test_preds=pos_test_scores, neg_test_preds=neg_test_scores, zero_test_preds=zero_test_scores,train_expls=train_expls, local_rank=local_rank)
        
        # Get the scores mapped to each item in place of the ranks!
        pos_val_expl_preds, neg_val_expl_preds = map_sorted_preds_to_ranked_items(pos_val_preds), map_sorted_preds_to_ranked_items(neg_val_preds)
    
        print("Alpha Search begins...")
        # --- Step 2: Alpha search ---
        user_alphas = run_validation(args, pos_val_expl_preds, neg_val_expl_preds, zero_val_preds, user_num, item_num, local_rank)
    
    # --- Step 3: Apply alphas on test set ---
    _ = apply_alphas_on_test(args, TAU, K, targetItems, item2id, pos_preds, neg_preds, zero_preds, pos_preds["gt"], user_alphas)
    print(f'It took {time() - start:.1f}s')
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
    
    test_expls = load_pickle(os.path.join(args.expl_path, args.dataset,'test.pkl'))
    
    
    eval_loader, eval_dataset = get_eval_dataset_loader(datamaps = datamaps, targetItems = targetItems, user_num = user_num, item_num = item_num, mode='test',batch_size=args.batch_size, dataset = args.dataset, data_path = args.data_path, expls = test_expls, workers=args.num_workers, shuffle=False)
    
    user_num, item_num = eval_dataset.user_num, eval_dataset.item_num - 1
    
    val_items = eval_dataset.get_val_items()
    test_items = eval_dataset.get_test_items()
    train_expls = load_pickle(os.path.join(args.expl_path, args.dataset, 'train.pkl'))
    
    
    print("#testing samples: ",len(eval_dataset))
    print("#testing batches: ",len(eval_loader))
    
    del eval_dataset
    
    main(args,eval_loader,user_num, item_num,val_items,test_items,targetItems,item2id, pos_preds, neg_preds, zero_preds, int(args.local_rank))

