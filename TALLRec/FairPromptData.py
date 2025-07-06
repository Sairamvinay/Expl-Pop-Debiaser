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

def get_fairprompt_loader(tokenizer, popitems, sample_numbers = 100,dataset='toys',data_path="../data",batch_size=16, workers=4, distributed=False,local_rank=0, shuffle=False):
    
    if dataset == 'yelp':
        data_obj = YelpFairPromptData(tokenizer=tokenizer,popitems = popitems,sample_numbers=sample_numbers, dataset=dataset, data_path=data_path, local_rank=local_rank)
    else:
        data_obj = AmazonFairPromptData(tokenizer=tokenizer,popitems = popitems,sample_numbers=sample_numbers, dataset=dataset, data_path=data_path, local_rank=local_rank)
    
    
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
    print(f"[INFO] Dataset Size: {len(data_obj)}, DataLoader Size: {len(loader)}")
    return loader, data_obj
    


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




def generate_prompt(data_point):
    if data_point["input"]:
        # noqa: E501
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
    
    else:
        # noqa: E501
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response: 
"""
        
class YelpFairPromptData(Dataset):
    def __init__(self, tokenizer, popitems, sample_numbers = 100, dataset='yelp', data_path = '../data/', local_rank=0):
        print("Yelp FairPrompt Dataset Loader Here!")
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        
        self.sample_numbers = sample_numbers
        self.split = dataset
        print('Dataset: ', self.split)

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
        
        self.negative_samples = ReadLineFromFile(os.path.join(self.data_path, self.split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join(self.data_path, self.split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.popitems = popitems
        print("# Target Items: ",len(self.popitems))
        print("Sample Items: ",list(self.popitems)[:5])                
        self.meta_data = load_pickle(os.path.join(self.data_path, self.split, 'meta_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        curr = 0
        self.total_length += len(self.user2id) * self.sample_numbers
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
        curr = self.total_length

        
    def __len__(self):
        return self.total_length
    
    
    def tokenize(self, prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(prompt,return_tensors="pt", padding=True, truncation=True, max_length = 1024)        
        return result
    
    
    def get_title(self,target_item):
        return self.meta_data[self.meta_dict[self.id2item[target_item]]].get('name','unknown title')
    
    def get_pop_id(self, item):
        return int(int(item) in self.popitems)
    
    def __getitem__(self, idx):
        
        out_dict = {}
        
        # <ItemTitleList>, <TargetItemTitle>, {Yes/No}
        
        data_point = {}
        
        fair_prompt= "You are a item-fair recommender. Please try to ensure that each category of items receives fair recommendations."
        
        data_point['instruction'] = f"A user has given high ratings to the following products. Leverage the information to predict whether the user would enjoy the target product? {fair_prompt} Answer with \"Yes\" or \"No\". Respond with exactly one word and no additional text."
        
        data_point['input'] = "User Preference with features: {}\nWhether the user will enjoy the target product {}?"
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        candidate_item_idx = datum_info_idx[2] 
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        
        purchase_history = sequence[1:-1]
        
        history_data = ""
        
        for item in purchase_history:
            title = "\"" + self.get_title(item) + "\""
            popID = self.get_pop_id(item)
            history_data += f"Item id: {item}, Item title: {title}, Item publisher: {popID}, \n" 

        
        if candidate_item_idx == 0:
            # +ve item
            target_text = "Yes"
            target_item = sequence[-1]
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')

            target_item = candidate_samples[candidate_item_idx-1]
            target_text = "No"
       
        
        cand_item_data = f"Item id: {target_item}, Item title: \"{self.get_title(target_item)}\", Item publisher: {self.get_pop_id(target_item)}, \n" 
        
        data_point['input'] = data_point['input'].format(history_data,cand_item_data)
        data_point['original_item'] = sequence[-1] # original item always
        
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        
        # return out_dict
        out_dict =  {
        "input_ids": torch.tensor(tokenized_full_prompt["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized_full_prompt["attention_mask"], dtype=torch.long),
        "data_point":data_point,
        }
        out_dict['target_item']  = target_item
        out_dict['user_id'] = user_id
        out_dict['target_text'] = target_text
        
        if not self.print_once:
            print("OUTDICT: ",out_dict)
            self.print_once = True
        
        
        return out_dict
        
    
    
    def collate_fn(self, batch):
        batch_entry = {}

        input_ids = [entry["input_ids"].squeeze(0) for entry in batch]
        attention_mask = [entry["attention_mask"].squeeze(0) for entry in batch]

        # Find max length in batch to pad manually
        max_length = max(len(ids) for ids in input_ids)

        def pad_sequence(seq, max_length):
            pad_len = max_length - len(seq)
            return torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long), seq], dim=0)


        input_ids = torch.stack([pad_sequence(seq, max_length) for seq in input_ids])
        attention_mask = torch.stack([pad_sequence(seq, max_length) for seq in attention_mask])
        

        batch_entry["input_ids"] = input_ids
        batch_entry["attention_mask"] = attention_mask
        batch_entry['original_item'] = [entry['data_point']['original_item'] for entry in batch]
        batch_entry['item_ids'] = [entry["target_item"] for entry in batch]
        batch_entry["user_ids"] = [entry["user_id"] for entry in batch]
        batch_entry["target_text"] = [entry["target_text"] for entry in batch]
        batch_entry['data_point'] = [entry['data_point'] for entry in batch]

        return batch_entry
     
        
class AmazonFairPromptData(Dataset):
    def __init__(self, tokenizer,popitems, sample_numbers = 100, dataset='toys', data_path = '../data/', local_rank=0):
        print("Amazon FairPrompt Dataset Loader Here!")
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        self.sample_numbers = sample_numbers
        self.split = dataset
        print('Dataset: ', self.split)

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
        
        self.negative_samples = ReadLineFromFile(os.path.join(self.data_path, self.split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join(self.data_path, self.split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        self.popitems = popitems
        print("# Target Items: ",len(self.popitems))
        print("Sample Items: ",list(self.popitems)[:5])
                        
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
        self.total_length += len(self.user2id) * self.sample_numbers
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
        curr = self.total_length

        
    def __len__(self):
        return self.total_length
    
    
    def tokenize(self, prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(prompt,return_tensors="pt", padding=True, truncation=True, max_length = 1024)        
        return result
    
    
    def get_title(self,target_item):
        return self.meta_data[self.meta_dict[self.id2item[target_item]]].get('title','unknown title')
    
    def get_pop_id(self, item):
        return int(int(item) in self.popitems)
    
    def __getitem__(self, idx):
        
        out_dict = {}
        
        # <ItemTitleList>, <TargetItemTitle>, {Yes/No}
        
        data_point = {}
        
        fair_prompt= "You are a item-fair recommender. Please try to ensure that each category of items receives fair recommendations."
        
        data_point['instruction'] = f"A user has given high ratings to the following products. Leverage the information to predict whether the user would enjoy the target product? {fair_prompt} Answer with \"Yes\" or \"No\". Respond with exactly one word and no additional text."
        
        data_point['input'] = "User Preference with features: {}\nWhether the user will enjoy the target product {}?"
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        candidate_item_idx = datum_info_idx[2] 
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        
        purchase_history = sequence[1:-1]
        
        history_data = ""
        
        for item in purchase_history:
            title = "\"" + self.get_title(item) + "\""
            popID = self.get_pop_id(item)
            history_data += f"Item id: {item}, Item title: {title}, Item publisher: {popID}, \n" 

        
        if candidate_item_idx == 0:
            # +ve item
            target_text = "Yes"
            target_item = sequence[-1]
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')

            target_item = candidate_samples[candidate_item_idx-1]
            target_text = "No"
       
        
        cand_item_data = f"Item id: {target_item}, Item title: \"{self.get_title(target_item)}\", Item publisher: {self.get_pop_id(target_item)}, \n" 
        
        data_point['input'] = data_point['input'].format(history_data,cand_item_data)
        data_point['original_item'] = sequence[-1] # original item always
        
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        
        # return out_dict
        out_dict =  {
        "input_ids": torch.tensor(tokenized_full_prompt["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(tokenized_full_prompt["attention_mask"], dtype=torch.long),
        "data_point":data_point,
        }
        out_dict['target_item']  = target_item
        out_dict['user_id'] = user_id
        out_dict['target_text'] = target_text
        
        if not self.print_once:
            print("OUTDICT: ",out_dict)
            self.print_once = True
        
        
        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}

        input_ids = [entry["input_ids"].squeeze(0) for entry in batch]
        attention_mask = [entry["attention_mask"].squeeze(0) for entry in batch]

        # Find max length in batch to pad manually
        max_length = max(len(ids) for ids in input_ids)
        
#         print("Max length: ",max_length)
#         print("Seq lens: ",[len(x) for x in input_ids])

        def pad_sequence(seq, max_length):
            pad_len = max_length - len(seq)
            return torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long), seq], dim=0)


        input_ids = torch.stack([pad_sequence(seq, max_length) for seq in input_ids])
        attention_mask = torch.stack([pad_sequence(seq, max_length) for seq in attention_mask])
        

        batch_entry["input_ids"] = input_ids
        batch_entry["attention_mask"] = attention_mask
        batch_entry['original_item'] = [entry['data_point']['original_item'] for entry in batch]
        batch_entry['item_ids'] = [entry["target_item"] for entry in batch]
        batch_entry["user_ids"] = [entry["user_id"] for entry in batch]
        batch_entry["target_text"] = [entry["target_text"] for entry in batch]
        batch_entry['data_point'] = [entry['data_point'] for entry in batch]
        return batch_entry

