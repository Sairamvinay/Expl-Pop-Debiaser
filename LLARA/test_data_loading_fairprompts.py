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
# import pandas as pd
import torch
import numpy as np
import html
import re
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

POP_MAP = {1:1, 0:0}

def load_prompt(prompt_path):
    if os.path.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        prompt_list = [p.strip() for p in raw_prompts]
        print('Load {} training prompts'.format(len(prompt_list)))
        print('Prompt Example \n{}'.format(prompt_list))
    else:
        prompt_list = []
    
    return prompt_list
        
def clean_text(text):
    # Convert HTML entities like &amp; and &nbsp;
    text = html.unescape(text)
    # Replace non-breaking spaces (\xa0) with regular spaces
    text = text.replace('\xa0', ' ')
    text = re.sub(u"\\<.*?\\>", "", text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text


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

    
def pad_history(seq,max_length):
    padlen = max(max_length - len(seq),0)
    if padlen > 0:
        return seq + [0 for _ in range(padlen)]
    return seq



class TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 llm_tokenizer=None,
                 train=False,
                 maxlen=784,
                 max_steps=1):
        
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.max_step = max_steps
        self.cur_step = 1
        self.maxlen = maxlen

    def __call__(self, batch):
        
        self.llm_tokenizer.padding_side = 'left'
        instruction = self.prompt_list[0] # random.choice(self.prompt_list)
        inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
#         if p < thresh_hold or not self.train:
        for i, sample in enumerate(batch):
            input_text=inputs_text[i]
            if '[HistoryHere]' in input_text:
                insert_prompt=", ".join([seq_title +' [HistoryEmb]' for seq_title in sample['InteractedItemTitles']])
                input_text=input_text.replace('[HistoryHere]',insert_prompt)
            if '[ItemHere]' in input_text:
                insert_prompt= sample['TargetItemTitle'] +' [ItemEmb]'
                input_text=input_text.replace('[ItemHere]',insert_prompt)    
            inputs_text[i]=input_text
#         flag = False
#         else:
#         for i, sample in enumerate(batch):
#             input_text=inputs_text[i]
#             if '[HistoryHere]' in input_text:
#                 insert_prompt=", ".join([seq_title + ' [PH]' for seq_title in sample['InteractedItemTitles']])
#                 input_text=input_text.replace('[HistoryHere]',insert_prompt)
#             if '[ItemHere]' in input_text:
#                 insert_prompt = sample['TargetItemTitle'] + " [PH]"
#                 input_text=input_text.replace('[ItemHere]',insert_prompt)    
#             inputs_text[i]=input_text
#         flag = True
        self.cur_step += 1
        
        targets_text = [sample['label'] for sample in batch]
        
        MAXLEN = max([entry['InteractedNum'] for entry in batch])

            
        batch_tokens = self.llm_tokenizer(
            inputs_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length = self.maxlen,
            add_special_tokens=True,
            return_attention_mask=True)

        new_batch={"tokens":batch_tokens,
                   'user_id':[sample['UserID'] for sample in batch],
                   'inputs_text': inputs_text,
                   "TargetItemID":np.array([entry['TargetItemID'] for entry in batch]),
                   "seq":torch.stack([torch.tensor(pad_history(sample['InteractedItemIDs'],MAXLEN)) for sample in batch], dim=0),
                   "len_seq":torch.stack([torch.tensor(sample['InteractedNum']) for sample in batch], dim=0),
                   "item_id": torch.stack([torch.tensor(sample['TargetItemID']) for sample in batch], dim=0),
                   "original_item": [sample['original_item'] for sample in batch],
                   'targets_text':targets_text,
                   }
        return new_batch    
                 



def get_dataset_object(popitems, sample_numbers, dataset='toys',data_path="../data", sample_type='random',local_rank=0):
    
    if dataset == 'yelp':
        data_obj = YelpFairData(popitems = popitems, batch_size=sample_numbers, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank)
    
    else:
        data_obj = AmazonFairData(popitems = popitems, batch_size=sample_numbers, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank)
    
    return data_obj
    
def get_fairprompts_loader(data_obj, tokenizer, prompt_path, mode='test',batch_size=16,workers=4,distributed=False,max_epochs=1,shuffle=False,maxlen=784):
    if distributed:
        sampler = DistributedSampler(data_obj)
    else:
        sampler = None
    
    prompt_list = load_prompt(prompt_path)    
    
    collater = TrainCollater(prompt_list=prompt_list,
                             llm_tokenizer = tokenizer,
                             train=False,
                             maxlen=maxlen
                            )
    
    loader = DataLoader(
        data_obj,
        batch_size=batch_size,
        num_workers=workers, pin_memory=False,
        sampler=sampler,
        shuffle=None if (sampler is not None) else shuffle,
        collate_fn= collater,
        drop_last=False)
    print(f"[INFO] Mode: {mode}, Dataset Size: {len(data_obj)}, DataLoader Size: {len(loader)}")
    return loader
    

class YelpFairData(Dataset):
    def __init__(self, popitems, dataset='yelp',batch_size=100, data_path = '../data/', sample_type='random',local_rank=0):
        print("Yelp Fair Dataset Loader Here!")
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        self.MAX_HISTORY_LEN = 5
        
        self.sample_numbers = batch_size
        self.split = dataset
        self.sample_type = sample_type
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
        '''
        class needs attribute
        user_num ; item_num
        '''
        self.user_num = len(self.user2id)
        self.item_num = len(self.item2id)
        
        
        self.user_id2name = load_pickle(os.path.join(self.data_path, self.split, 'user_id2name.pkl'))
                
        self.meta_data = load_pickle(os.path.join(self.data_path, self.split, 'meta_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
        
        self.popitems = [int(self.item2id[item]) for item in popitems]

        print("# Target Items: ",len(self.popitems))
        print("Sample Items: ",list(self.popitems)[:5])

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
    
    def get_title(self,target_item):
        return clean_text(self.meta_data[self.meta_dict[self.id2item[target_item]]].get('name','unknown title'))[:200]
    
    def get_pop_id(self, item):
        return POP_MAP[int(int(item) in self.popitems)]
    
    def __getitem__(self, idx):
        
        out_dict = {}
        
        data_point = {}
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        user_desc = self.user_id2name[user_id]
        
        purchase_history = sequence[1:-1]            
        
        purchase_history = np.random.choice(purchase_history,min(self.MAX_HISTORY_LEN,len(purchase_history)), replace=False)
        purchase_history = purchase_history.tolist()

        # purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]
        history_data = []
        for item in purchase_history:
            title = "\"" + self.get_title(item) + "\""
            popID = self.get_pop_id(item)
            history_data.append(f"Item title: {title}, Item publisher: {popID}") 
        
        
        candidate_item_idx = datum_info_idx[2] 
        
        
        if candidate_item_idx == 0:
            # +ve item
            target_text = "Yes"
            target_label = 1
            target_item = sequence[-1]
            target_item_title = "\"" + self.get_title(target_item) + "\""
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')

            target_item = candidate_samples[candidate_item_idx-1]
            target_text = "No"
            target_item_title = "\"" + self.get_title(target_item) + "\""
            target_label = 0
        
        '''
        sample needs following input keys
            UserID
            TargetItemID
            InteractedItemIDs_pad
            InteractedNum
            InteractedItemTitles
            TargetItemTitle
            label:
        
        '''
        cand_item_data = f"Item title: {target_item_title}, Item publisher: {self.get_pop_id(target_item)}" 
        
        
        out_dict['UserID'] = int(user_id)
        out_dict['TargetItemID'] = int(target_item)
        out_dict['InteractedNum'] = len(purchase_history)
        out_dict['InteractedItemIDs'] = [int(item) for item in purchase_history]
        out_dict['original_item'] = int(sequence[-1])
        out_dict['InteractedItemTitles'] = history_data
        out_dict['TargetItemTitle'] = cand_item_data
        
        out_dict['label'] = target_text
        
        
        return out_dict
    

class AmazonFairData(Dataset):
    def __init__(self, popitems, dataset='toys', data_path = '../data/',batch_size=100, sample_type='random',local_rank=0):
        print("Amazon Fair Dataset Loader Here!")
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        self.MAX_HISTORY_LEN = 5
        self.split = dataset
        
        self.sample_numbers = batch_size
        self.sample_type = sample_type
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
        
            
        datamaps = load_json(os.path.join(self.data_path, self.split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        '''
        class needs attribute
        user_num ; item_num
        '''
        self.user_num = len(self.user2id)
        self.item_num = len(self.item2id)
        self.negative_samples = ReadLineFromFile(os.path.join(self.data_path, self.split, 'negative_samples.txt'))
        
        
        self.user_id2name = load_pickle(os.path.join(self.data_path, self.split, 'user_id2name.pkl'))
                
        self.meta_data = []
        for meta in parse(os.path.join(self.data_path, self.split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
        
        self.popitems = [int(self.item2id[item]) for item in popitems]

        print("# Target Items: ",len(self.popitems))
        print("Sample Items: ",list(self.popitems)[:5])
        
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
    
    def get_title(self,target_item):
        return clean_text(self.meta_data[self.meta_dict[self.id2item[target_item]]].get('title','unknown title'))[:200]
    
    def get_pop_id(self, item):
        return POP_MAP[int(int(item) in self.popitems)]
    
    def __getitem__(self, idx):
        
        out_dict = {}
        
        data_point = {}
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        user_desc = self.user_id2name[user_id]
        
        purchase_history = sequence[1:-1]            
        
        purchase_history = np.random.choice(purchase_history,min(self.MAX_HISTORY_LEN,len(purchase_history)), replace=False)
        purchase_history = purchase_history.tolist()

        # purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]
        history_data = []
        for item in purchase_history:
            title = "\"" + self.get_title(item) + "\""
            popID = self.get_pop_id(item)
            history_data.append(f"Item title: {title}, Item publisher: {popID}") 
        
        
        candidate_item_idx = datum_info_idx[2] 
        
        
        if candidate_item_idx == 0:
            # +ve item
            target_text = "Yes"
            target_label = 1
            target_item = sequence[-1]
            target_item_title = "\"" + self.get_title(target_item) + "\""
            
        else:
            # -ve item
            user_seq = self.user_items[user_id]
            assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')

            target_item = candidate_samples[candidate_item_idx-1]
            target_text = "No"
            target_item_title = "\"" + self.get_title(target_item) + "\""
            target_label = 0
        
        '''
        sample needs following input keys
            UserID
            TargetItemID
            InteractedItemIDs_pad
            InteractedNum
            InteractedItemTitles
            TargetItemTitle
            label:
        
        '''
        cand_item_data = f"Item title: {target_item_title}, Item publisher: {self.get_pop_id(target_item)}" 
        
        
        out_dict['UserID'] = int(user_id)
        out_dict['TargetItemID'] = int(target_item)
        out_dict['InteractedNum'] = len(purchase_history)
        out_dict['InteractedItemIDs'] = [int(item) for item in purchase_history]
        out_dict['original_item'] = int(sequence[-1])
        out_dict['InteractedItemTitles'] = history_data # purchase_history_titles
        out_dict['TargetItemTitle'] = cand_item_data
        
        out_dict['label'] = target_text
        
        
        return out_dict
    
    
    