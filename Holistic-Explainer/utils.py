from __future__ import absolute_import, division, print_function

import random
import numpy as np
import os
import torch
from scipy.stats import entropy

import math
import numpy as np
import heapq
import random
from collections import defaultdict, Counter
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import re
import ast

def verify_model_weights(model):
    for name, param in model.named_parameters():
        print(f"{name} → {param.shape} -> {param.data.abs().mean()}")

def extract_response_list_from_decoded_output(text):
    """
    Extracts a valid Python-style list of integers from raw LLM output text (post-decode).
    This version does NOT require '### Response:'.
    """
    match = re.search(r'\[\s*(\d+\s*,\s*)*\d+\s*\]', text)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except Exception:
            return None
    return None

def print_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"🔢 Trainable params: {trainable_params:,}")
    print(f"📦 Total params: {total_params:,}")
    print(f"📉 Percentage: {100 * trainable_params / total_params:.4f}%")


def load_best_val_loss(path):
    """
    Loads the best validation loss from a saved .pt file.
    
    Args:
        path (str): Path to 'best_val_loss.pt'
    
    Returns:
        float: The best validation loss
    """
    try:
        val_loss = torch.load(path) 
        print(f"📈 Loaded best validation loss: {val_loss:.6f}")
        return val_loss
    except Exception as e:
        print(f"❌ Failed to load validation loss from '{path}': {e}")
        return None

def verify_loaded_peft(model):
    print(f"🔍 Model class: {type(model)}")
    sd = get_peft_model_state_dict(model)
    lora_keys = [k for k in sd if 'lora_' in k]
    print(f"🔍 Found {len(lora_keys)} LoRA weights.")
    if lora_keys:
        mean_val = sd[lora_keys[0]].abs().mean().item()
        print(f"🔍 Sample LoRA param '{lora_keys[0]}': mean={mean_val:.6f}")
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🧠 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
        if mean_val < 1e-5:
            print("⚠️ LoRA weights are very small — possibly untrained?")
        else:
            print("✅ LoRA adapter appears correctly loaded and trained.")
        

def unwrap_all_layers(m):
    while hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "module"):  # handles DDP after base_model nesting
        m = m.module
    return m

def unwrap_peft_model(model):
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):  # torch.compile wrapped model
        model = model._orig_mod
    return model

def load_checkpoint(model,url_or_filename):
        """
        Resume from a checkpoint.
        """
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename)
        else:
            raise RuntimeError("checkpoint path is invalid")

        state_dict = checkpoint["model"]
        unwrap_all_layers(model).load_state_dict(state_dict,strict=False)
        
        
        print("Resume checkpoint from {}".format(url_or_filename))
        return model
        
def save_checkpoint(model,output_dir,cur_epoch,compress=True):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = unwrap_all_layers(model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        if compress:
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    # delete parameters that do not require gradient
                    del state_dict[k]
        save_obj = {
            "model": state_dict,
            "epoch": cur_epoch,
        }
        
        if cur_epoch == 'BEST':
            save_to = os.path.join(output_dir, f'checkpoint_epoch_{cur_epoch}.pt')
        else:
            save_to = os.path.join(output_dir, f'checkpoint_epoch_{cur_epoch+1}.pt')

        print("Saving checkpoint at path {} at Epoch {}".format(save_to,cur_epoch))
        torch.save(save_obj, save_to)
        return

def load_model_checkpoint(resume_from_checkpoint, model):
    """
    Load LoRA adapter weights into a PEFT-wrapped model.
    This assumes the checkpoint is a LoRA-only adapter (saved via get_peft_model_state_dict).
    """
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔁 Loading checkpoint from: {resume_from_checkpoint}")

        # Load adapter-only state dict
        adapter_weights = torch.load(resume_from_checkpoint)

        if not any("lora_" in k for k in adapter_weights):
            raise ValueError("❌ No LoRA weights found in the checkpoint!")

        # Unwrap PEFT/DDP model for direct state_dict access
        model = unwrap_peft_model(model)

        # Load adapter weights
        set_peft_model_state_dict(model, adapter_weights)

        print(f"✅ LoRA adapter weights loaded.")
        print(f"  → Loaded {len(adapter_weights)} keys")
        
    else:
        print("No checkpoint path was provided! Training with random initialized weights!!")

    return model

def save_checkpoint_peft(peft_model, path):
    adapter_weights = get_peft_model_state_dict(unwrap_peft_model(peft_model))
    torch.save(adapter_weights,path)

def safe_save_peft_adapter(peft_model, path):
    os.makedirs(path, exist_ok=True)
    
    # Extract only LoRA weights
    
    # adapter_state_dict = {k: v for k, v in peft_model.state_dict().items() if 'lora_' in k}
    
    # Official, robust PEFT method for saving adapter weights
    adapter_state_dict = get_peft_model_state_dict(unwrap_peft_model(peft_model))
    
    if len(adapter_state_dict) == 0:
        raise ValueError("❌ No LoRA weights found in state_dict! Is this a PeftModel?")
    
    torch.save(adapter_state_dict, os.path.join(path, 'adapter_model.bin'))
    
    # Save adapter config
    unwrap_peft_model(peft_model).peft_config['default'].save_pretrained(path)
    
    print(f"✅ Saved {len(adapter_state_dict)} LoRA adapter weights to: {path}")


def area_curve_metric(golds, preds):
    return roc_auc_score(golds, preds)

    
def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FairnessMetric:
    def __init__(self, user_item_scores, targetItems, top, itemNum):
        self.user_recommendations = user_item_scores
        self.popular_items = set(targetItems)
        self.top = top
        self.itemNum = itemNum

    def compute_metrics(self):
        results = {}
        
        for K in self.top:
            topK_recommendations = self.get_topK_recommendations(K)
            item_counts = self.get_item_counts(topK_recommendations)
            
            gini_index = self.gini_index(item_counts)
            popularity_rate = self.popularity_rate(topK_recommendations)
            long_tail_rate = self.long_tail_rate(topK_recommendations)
            kl_divergence = self.kl_divergence(item_counts)
            simpson_diversity = self.simpson_diversity(item_counts)
            
            user_head_coverage = self.user_head_cover(topK_recommendations)
            
            results[K] = {
                "Gini Index": gini_index,
                "Popularity Rate": popularity_rate,
                "Long Tail Rate": long_tail_rate,
                "KL Divergence": kl_divergence,
                "Simpson Diversity": simpson_diversity,
                "UHC":user_head_coverage,
            }
            topk = K
            msg = "\nPR@{}\tLTR@{}\tKLD@{}\tGini@{}\tSDI@{}\tUHC@{}".format(topk, topk, topk, topk, topk,topk)
            msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(popularity_rate, long_tail_rate, kl_divergence, gini_index, simpson_diversity,user_head_coverage)
            print(msg)
        
        return results
    
    def get_topK_recommendations(self, K):
        return {user: sorted(items, key=items.get, reverse=True)[:K] for user, items in self.user_recommendations.items()}
    
    def get_item_counts(self, recommendations):
        item_counts = Counter()
        for recs in recommendations.values():
            item_counts.update(recs)
        return item_counts
    
    def user_head_cover(self, recommendations):
        num_users = len(recommendations)
        if num_users == 0:
            return 0.0
        num_user_head = sum(1 for recs in recommendations.values() if any(item in self.popular_items for item in recs))
        return num_user_head / num_users        
    
    def gini_index(self, item_counts):
        values = np.array(list(item_counts.values()), dtype=np.float32)
        values.sort()
        n = len(values)
        if n == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))
    
    def popularity_rate(self, recommendations):
        total_recommendations = sum(len(items) for items in recommendations.values())
        if total_recommendations == 0:
            return 0.0
        popular_count = sum(1 for recs in recommendations.values() for item in recs if item in self.popular_items)
        return popular_count / total_recommendations
    
    def long_tail_rate(self, recommendations):
        return 1 - self.popularity_rate(recommendations)
    
    def kl_divergence(self, item_counts):
        idealdist = [len(self.popular_items)/self.itemNum, (self.itemNum - len(self.popular_items))/self.itemNum]
        
        popCount = 0
        tailCount = 0
        total = 0
        for item, count in item_counts.items():
            total += count
            if item in self.popular_items:
                popCount += count
            else:
                tailCount += count
        
        truedist = [popCount / max(1,total), tailCount / max(1,total)] # zero division handling
        return entropy(truedist, idealdist)
        
        
    def simpson_diversity(self, item_counts):
        total_recommendations = sum(item_counts.values())
        if total_recommendations == 0:
            return 0.0
        
        popCount, tailCount = 0,0
        for item, count in item_counts.items():
            if item in self.popular_items:
                popCount += count
            else:
                tailCount += count
        
        main_num = (popCount * (popCount - 1)) + (tailCount * (tailCount - 1))
        main_denom = total_recommendations * (total_recommendations - 1)

        return 1 - float(main_num / max(1,main_denom)) # avoid zero division



def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.0
    return np.mean(r[: z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asarray(r, dtype=np.float64)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max

def evaluate_once(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_id>. length of the list is topK.
        groundtruth: list of <item_id>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = []
    for iid in topk_preds:
        if iid in gt_set:
            rel.append(1)
        else:
            rel.append(0)
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        "rel": rel,
    }


def evaluate_all(user_item_scores, groudtruth, topk=10,debug=True):
    """Evaluate all user-items performance.
    Args:
        user_item_scores: dict with key = <item_id>, value = <user_item_score>.
                     Make sure larger score means better recommendation.
        groudtruth: dict with key = <user_id>, value = list of <item_id>.
        topk: int
    Returns:
    """
    avg_prec, avg_recall, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    for uid in user_item_scores:
        # [Important] Use shuffle to break ties!!!
        ui_scores = list(user_item_scores[uid].items())
        np.random.shuffle(ui_scores)  # break ties
        # topk_preds = heapq.nlargest(topk, user_item_scores[uid], key=user_item_scores[uid].get)  # list of k <item_id>
        topk_preds = heapq.nlargest(topk, ui_scores, key=lambda x: x[1]) # list of k tuples
        topk_preds = [x[0] for x in topk_preds]  # list of k <item_id>
        # print(topk_preds, groudtruth[uid])
        result = evaluate_once(topk_preds, groudtruth[uid])
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1

    
    
    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    map_ = mean_average_precision(rs)
    mrr = mean_reciprocal_rank(rs)
    msg = "\nNDCG@{}\tRec@{}\tHits@{}\tPrec@{}\tMAP@{}\tMRR@{}".format(topk, topk, topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec, map_, mrr)
    if debug:
        print(msg)
    res = {
        'ndcg': avg_ndcg,
        'map': map_,
        'recall': avg_recall,
        'precision': avg_prec,
        'mrr': mrr,
        'hit': avg_hit,
    }
    return msg, res

def getBasicScores(ui_scores,gt,top):
    msgs,recs = [],{}
    for k in top:

        msg,rec = evaluate_all(ui_scores, gt, k)
        msgs.append(msg)
        recs[k] = rec


    return msgs, recs

def getFairnessScores(ui_scores,targetItems,top,itemNum):
    return FairnessMetric(ui_scores,targetItems,top,itemNum).compute_metrics()

