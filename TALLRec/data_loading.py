from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import sys
import math
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def compute_iid2pid(item_counts, data_path, split, num_groups=5):
    """
    Computes a mapping from item IDs to popularity group IDs.

    Args:
        item_counts: map itemid to its frequency across dataset.
        num_groups: int, number of popularity groups to split items into.

    Returns:
        iid2pid: dict mapping item ID to group ID (0 = most popular).
    """
    file_path = os.path.join(data_path, split, f'iid2popid-{num_groups}.json')
    if os.path.exists(file_path):
        print("Loading iid2pid from existing file path: ",file_path)
        iid2pid = load_json(file_path)
        iid2pid = {int(k): v for k, v in iid2pid.items()}
        print("Random item maps: ",random.sample(list(iid2pid.items()), 10))
        return iid2pid
    
    # Step 2: Sort items by count descending (most popular first)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Divide into groups
    total_items = len(sorted_items)
    group_size = total_items // num_groups
    iid2pid = {}
    for idx, (item, _) in enumerate(sorted_items):
        group = min(idx // group_size, num_groups - 1)  # avoid overflow in final group
        iid2pid[int(item)] = group
        
    
    save_json({str(k): v for k, v in iid2pid.items()}, file_path)
    print("Written new iid2pid File: ",file_path)
    print("Random item maps: ",random.sample(list(iid2pid.items()), 10))
    return iid2pid


def get_dataset_loader(tokenizer, sample_numbers,cutoff_len = 256,train_on_inputs = True,mode='train', dataset='toys',data_path="../data", sample_type='random',batch_size=16, workers=4, distributed=False,local_rank=0, shuffle=False):
    
    if dataset == 'yelp':
        data_obj = YelpData(tokenizer=tokenizer,sample_numbers=sample_numbers,cutoff_len = cutoff_len,train_on_inputs = train_on_inputs,mode=mode, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank)
    
    else:
        data_obj = AmazonData(tokenizer=tokenizer,sample_numbers=sample_numbers,cutoff_len = cutoff_len,train_on_inputs = train_on_inputs,mode=mode, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank)
    
    
    if distributed:
        sampler = DistributedSampler(data_obj)
    else:
        sampler = None
    
    loader = DataLoader(
        data_obj,
        batch_size=batch_size,
        num_workers=workers, pin_memory=False,
        sampler=sampler,
        shuffle=None if (sampler is not None) else shuffle,
        collate_fn=data_obj.collate_fn,
        drop_last=False)
    print(f"[INFO] Mode: {mode}, Dataset Size: {len(data_obj)}, DataLoader Size: {len(loader)}")
    return loader, data_obj
    
def generate_prompt(data_point):
    if data_point["input"]:
        # noqa: E501
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    
    else:
        # noqa: E501
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response: 
{data_point["output"]}"""
        
     
        

class YelpData(Dataset):
    def __init__(self, tokenizer, sample_numbers,cutoff_len = 256,train_on_inputs = True,mode='train', dataset='toys', data_path = '../data/', sample_type='random',local_rank=0,num_groups=5):
        print("Yelp Dataset Loader Here!")
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        
        self.sample_numbers = sample_numbers
        self.split = dataset
        self.sample_type = sample_type
        print('Dataset: ', self.split)
        self.mode = mode

        self.sequential_data = ReadLineFromFile(os.path.join(self.data_path, self.split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
            
        datamaps = load_json(os.path.join(self.data_path, self.split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        # ADDED MAY 11 for Fairness Inclusion
        if mode == 'train':
            self.iid2pid_dict = compute_iid2pid(item_count, self.data_path, self.split, num_groups=num_groups)
        
        self.user_id2name = load_pickle(os.path.join(self.data_path, self.split, 'user_id2name.pkl'))
                
        self.meta_data = load_pickle(os.path.join(self.data_path, self.split, 'meta_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
        
#         self.user_data = load_pickle(os.path.join(self.data_path, self.split, 'user_data.pkl'))
#         self.user_meta_dict = {}
#         for j, user_meta_item in enumerate(self.user_data):
#             self.user_meta_dict[user_meta_item['user_id']] = j
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        
        if self.mode == 'train':
            self.total_length += len(self.sequential_data) * self.sample_numbers
            for i in range(self.total_length - curr):
                self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
            curr = self.total_length


        elif self.mode == 'val':
            self.total_length += len(self.user2id) * self.sample_numbers
            for i in range(self.total_length - curr):
                self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
            curr = self.total_length

        else:
            raise NotImplementedError
    
    def __len__(self):
        return self.total_length
    
    
    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        # print(f"[DEBUG] Tokenized length: {len(result['input_ids'])}, Model Max Length: {self.tokenizer.model_max_length}")
        return result
    
    
    def get_title(self,target_item):
        return self.meta_data[self.meta_dict[self.id2item[target_item]]].get('name','unknown title')
            
    def __getitem__(self, idx):
        
        out_dict = {}
        
        # <ItemTitleList>, <TargetItemTitle>, {Yes/No}
        # source_text = "A user has given high ratings to the following products: {}. Leverage the information to predict whether the user would enjoy the product titled {}? Answer with \"Yes\" or \"No\"."  
        data_point = {}
        data_point['instruction'] = "A user has given high ratings to the following products. Leverage the information to predict whether the user would enjoy the target product?  Answer with \"Yes\" or \"No\". Respond with exactly one word and no additional text."
        data_point['input'] = "User Preference: {}\nWhether the user will enjoy the target product {}?"
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        user_desc = self.user_id2name[user_id]
        
        if self.mode == 'train':
            end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
            end_index = random.randint(0, len(end_candidates)-1)
            end_pos = end_candidates[end_index]
            start_candidates = [_ for _ in range(1, min(4, end_pos))]
            start_index = random.randint(0, len(start_candidates)-1)
            start_pos = start_candidates[start_index]
            purchase_history = sequence[start_pos:end_pos+1] # sample a history sequence from the full user purchase history
            target_item = sequence[end_pos+1]
            seq_len = len(sequence) - 2
        elif self.mode == 'val':
            purchase_history = sequence[1:-2]
            target_item = sequence[-2]
            seq_len = 1
        
        else:
            raise NotImplementedError
        
        # Ranking style correction made: Mar 6, 2024 for tallrec; Training style is different from Tallrec and modelled more onlines of P5's ranking style data structure: I use all/subset of purchasing history and don't use likes/dislikes concept which is mirroring old rating style prediction idea. 
        purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]
        
        candidate_item_idx = datum_info_idx[2] 
        
        # rand_prob = random.random()
        # if rand_prob > 0.5:
        
        # 50% of the samples are positive / better than the 50% randomization
        if candidate_item_idx < (self.sample_numbers // 2):
            # +ve item
            target_text = "Yes"
            target_item_title = "\"" + self.get_title(target_item) + "\""
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            candidate_samples = []
            candidate_num = 1
            while len(candidate_samples) < candidate_num:
                if self.sample_type == 'random':
                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                else:
                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                candidate_samples.extend(sample_ids)
            candidate_samples = candidate_samples[:candidate_num]

            target_item = candidate_samples[0]
            target_text = "No"
            target_item_title = "\"" + self.get_title(target_item) + "\""
        
        
        data_point['input'] = data_point['input'].format(', '.join(purchase_history_titles), target_item_title)
        data_point['output'] = target_text
        
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
        # return out_dict
        out_dict =  {
        "input_ids": torch.tensor(tokenized_full_prompt["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized_full_prompt["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(tokenized_full_prompt["labels"], dtype=torch.long),
        "data_point":full_prompt,
        "target_item":int(target_item),
        "history_item": [int(item) for item in purchase_history]
        }
#         if not self.print_once:
#             print("OUTDICT: ",out_dict)
#             self.print_once = True
        
        return out_dict
        
    
    
    def collate_fn(self, batch):
        batch_entry = {}

        input_ids = [entry["input_ids"] for entry in batch]
        attention_mask = [entry["attention_mask"] for entry in batch]
        labels = [entry["labels"] for entry in batch]

        # Find max length in batch to pad manually
        max_length = max(len(ids) for ids in input_ids)

        def pad_sequence(seq, max_length):
            pad_len = max_length - len(seq)
            return torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long), seq], dim=0)


        input_ids = torch.stack([pad_sequence(seq, max_length) for seq in input_ids])
        attention_mask = torch.stack([pad_sequence(seq, max_length) for seq in attention_mask])
        labels = torch.stack([pad_sequence(seq, max_length) for seq in labels])
        

        
        batch_entry["input_ids"] = input_ids # .to(device, non_blocking=True)
        batch_entry["attention_mask"] = attention_mask # .to(device, non_blocking=True)
        batch_entry["labels"] = labels # .to(device, non_blocking=True)
        batch_entry['data_point'] = [entry['data_point'] for entry in batch]
        batch_entry['target_items'] = [int(entry["target_item"]) for entry in batch]
        batch_entry['history_items'] = [entry['history_item'] for entry in batch]

        return batch_entry


        




class AmazonData(Dataset):
    def __init__(self, tokenizer, sample_numbers,cutoff_len = 256,train_on_inputs = True,mode='train', dataset='toys', data_path = '../data/', sample_type='random',local_rank=0,num_groups=5):
        print("Amazon Dataset Loader Here!")
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        
        self.sample_numbers = sample_numbers
        self.split = dataset
        self.sample_type = sample_type
        print('Dataset: ', self.split)
        self.mode = mode

        self.sequential_data = ReadLineFromFile(os.path.join(self.data_path, self.split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
            
        datamaps = load_json(os.path.join(self.data_path, self.split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        # ADDED MAY 11 for Fairness Inclusion
        if mode == 'train':
            self.iid2pid_dict = compute_iid2pid(item_count, self.data_path, self.split, num_groups=num_groups)
        
        self.user_id2name = load_pickle(os.path.join(self.data_path, self.split, 'user_id2name.pkl'))
                
        self.meta_data = []
        for meta in parse(os.path.join(self.data_path, self.split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        
        if self.mode == 'train':
            self.total_length += len(self.sequential_data) * self.sample_numbers
            for i in range(self.total_length - curr):
                self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
            curr = self.total_length


        elif self.mode == 'val':
            self.total_length += len(self.user2id) * self.sample_numbers
            for i in range(self.total_length - curr):
                self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
            curr = self.total_length

        else:
            raise NotImplementedError
    
    def __len__(self):
        return self.total_length
    
    
    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        # print(f"[DEBUG] Tokenized length: {len(result['input_ids'])}, Model Max Length: {self.tokenizer.model_max_length}")
        return result
    
    
    def get_title(self,target_item):
        return self.meta_data[self.meta_dict[self.id2item[target_item]]].get('title','unknown title')
            
    def __getitem__(self, idx):
        
        out_dict = {}
        
        # <ItemTitleList>, <TargetItemTitle>, {Yes/No}
        # source_text = "A user has given high ratings to the following products: {}. Leverage the information to predict whether the user would enjoy the product titled {}? Answer with \"Yes\" or \"No\"."  
        data_point = {}
        data_point['instruction'] = "A user has given high ratings to the following products. Leverage the information to predict whether the user would enjoy the target product?  Answer with \"Yes\" or \"No\". Respond with exactly one word and no additional text."
        data_point['input'] = "User Preference: {}\nWhether the user will enjoy the target product {}?"
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        user_desc = self.user_id2name[user_id]
        
        if self.mode == 'train':
            end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
            end_index = random.randint(0, len(end_candidates)-1)
            end_pos = end_candidates[end_index]
            start_candidates = [_ for _ in range(1, min(4, end_pos))]
            start_index = random.randint(0, len(start_candidates)-1)
            start_pos = start_candidates[start_index]
            purchase_history = sequence[start_pos:end_pos+1] # sample a history sequence from the full user purchase history
            target_item = sequence[end_pos+1]
            seq_len = len(sequence) - 2
        elif self.mode == 'val':
            purchase_history = sequence[1:-2]
            target_item = sequence[-2]
            seq_len = 1
        
        else:
            raise NotImplementedError
        
        # Ranking style correction made: Mar 6, 2024 for tallrec; Training style is different from Tallrec and modelled more onlines of P5's ranking style data structure: I use all/subset of purchasing history and don't use likes/dislikes concept which is mirroring old rating style prediction idea. 
        purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]
        
        candidate_item_idx = datum_info_idx[2] 
        
        # rand_prob = random.random()
        # if rand_prob > 0.5:
        
        # 50% of the samples are positive / better than the 50% randomization
        if candidate_item_idx < (self.sample_numbers // 2):
            # +ve item
            target_text = "Yes"
            target_item_title = "\"" + self.get_title(target_item) + "\""
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            candidate_samples = []
            candidate_num = 1
            while len(candidate_samples) < candidate_num:
                if self.sample_type == 'random':
                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                else:
                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                candidate_samples.extend(sample_ids)
            candidate_samples = candidate_samples[:candidate_num]

            target_item = candidate_samples[0]
            target_text = "No"
            target_item_title = "\"" + self.get_title(target_item) + "\""
        
        
        data_point['input'] = data_point['input'].format(', '.join(purchase_history_titles), target_item_title)
        data_point['output'] = target_text
        
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
        # return out_dict
        out_dict =  {
        "input_ids": torch.tensor(tokenized_full_prompt["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized_full_prompt["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(tokenized_full_prompt["labels"], dtype=torch.long),
        "data_point":full_prompt,
        "target_item":int(target_item),
        "history_item": [int(item) for item in purchase_history]
        }
#         if not self.print_once:
#             print("OUTDICT: ",out_dict)
#             self.print_once = True
        
        return out_dict
        
    
    
    def collate_fn(self, batch):
        batch_entry = {}

        input_ids = [entry["input_ids"] for entry in batch]
        attention_mask = [entry["attention_mask"] for entry in batch]
        labels = [entry["labels"] for entry in batch]

        # Find max length in batch to pad manually
        max_length = max(len(ids) for ids in input_ids)

        def pad_sequence(seq, max_length):
            pad_len = max_length - len(seq)
            return torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long), seq], dim=0)


        input_ids = torch.stack([pad_sequence(seq, max_length) for seq in input_ids])
        attention_mask = torch.stack([pad_sequence(seq, max_length) for seq in attention_mask])
        labels = torch.stack([pad_sequence(seq, max_length) for seq in labels])
        

        
        batch_entry["input_ids"] = input_ids # .to(device, non_blocking=True)
        batch_entry["attention_mask"] = attention_mask # .to(device, non_blocking=True)
        batch_entry["labels"] = labels # .to(device, non_blocking=True)
        batch_entry['data_point'] = [entry['data_point'] for entry in batch]
        batch_entry['target_items'] = [int(entry["target_item"]) for entry in batch]
        batch_entry['history_items'] = [entry['history_item'] for entry in batch]        

        return batch_entry

