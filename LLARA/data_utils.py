from torch.utils.data import Dataset
from collections import defaultdict
import json
import gzip
import random
import pickle
import pandas as pd
import numpy as np
import html
import re
import os


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

        
class YelpData(Dataset):
    def __init__(self, sample_numbers,mode='train', dataset='toys', data_path = '../data/', sample_type='random',local_rank=0,num_groups=5):
        print("Yelp Dataset Loader Here!")
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        self.MAX_HISTORY_LEN = 5
        
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
        '''
        class needs attribute
        user_num ; item_num
        '''
        self.user_num = len(self.user2id)
        self.item_num = len(self.item2id)
        
        # ADDED MAY 11 for Fairness Inclusion
        if mode == 'train':
            self.iid2pid_dict = compute_iid2pid(item_count, self.data_path, self.split, num_groups=num_groups)
        
        self.user_id2name = load_pickle(os.path.join(self.data_path, self.split, 'user_id2name.pkl'))
                
        self.meta_data = load_pickle(os.path.join(self.data_path, self.split, 'meta_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
            
        # self.user_data = load_pickle(os.path.join(self.data_path, self.split, 'user_data.pkl'))
        # self.user_meta_dict = {}
        # for j, user_meta_item in enumerate(self.user_data):
        #    self.user_meta_dict[user_meta_item['user_id']] = j
        
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
    
    def get_title(self,target_item):
        return clean_text(self.meta_data[self.meta_dict[self.id2item[target_item]]].get('name','unknown title'))[:200]
            
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
        
        purchase_history = np.random.choice(purchase_history,min(self.MAX_HISTORY_LEN,len(purchase_history)), replace=False)
        purchase_history = purchase_history.tolist()

        purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]
        
        candidate_item_idx = datum_info_idx[2] 
        
        
        # 50% of the samples are positive / better than the 50% randomization
        if candidate_item_idx < (self.sample_numbers // 2):
            # +ve item
            target_text = "Yes"
            target_label = 1
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
            target_label = 0
            target_text = "No"
            target_item_title = "\"" + self.get_title(target_item) + "\""
        
        '''
        sample needs following input keys
            UserID
            TargetItemID: item_id in LLARA
            InteractedItemIDs: seq in LLARA
            InteractedNum: len_seq in LLARA
            InteractedItemTitles: seq_name in LLARA
            TargetItemTitle: item_name in LLARA
            label: (there is correct answer, but we have yes/no values)
        
        '''
            
        out_dict['UserID'] = int(user_id)
        out_dict['TargetItemID'] = int(target_item)
        out_dict['InteractedNum'] = len(purchase_history)
        out_dict['InteractedItemIDs'] = [int(item) for item in purchase_history]
        
        out_dict['InteractedItemTitles'] = purchase_history_titles
        out_dict['TargetItemTitle'] = target_item_title
        
        out_dict['label'] = target_text
        
        
        return out_dict
    


class AmazonData(Dataset):
    def __init__(self, sample_numbers,mode='train', dataset='toys', data_path = '../data/', sample_type='random',local_rank=0,num_groups=5):
        print("Amazon Dataset Loader Here!")
        self.data_path = data_path
        self.local_rank=local_rank
        self.print_once=False
        
        self.MAX_HISTORY_LEN = 5
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
        
        '''
        class needs attribute
        user_num ; item_num
        '''
        self.user_num = len(self.user2id)
        self.item_num = len(self.item2id)
        
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
    
    def get_title(self,target_item):
        return clean_text(self.meta_data[self.meta_dict[self.id2item[target_item]]].get('title','unknown title'))[:200]
            
    def __getitem__(self, idx):
        
        out_dict = {}
        
        
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
        
        
        purchase_history = np.random.choice(purchase_history,min(self.MAX_HISTORY_LEN,len(purchase_history)), replace=False)
        purchase_history = purchase_history.tolist()
        purchase_history_titles = ["\"" + self.get_title(item) + "\"" for item in purchase_history]

        candidate_item_idx = datum_info_idx[2] 
        
        
        # 50% of the samples are positive / better than the 50% randomization
        if candidate_item_idx < (self.sample_numbers // 2):
            # +ve item
            target_text = "Yes"
            target_label = 1
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
            target_label = 0
        
        '''
        sample needs following input keys
            UserID
            TargetItemID: item_id in LLARA
            InteractedItemIDs: seq in LLARA
            InteractedNum: len_seq in LLARA
            InteractedItemTitles: seq_name in LLARA
            TargetItemTitle: item_name in LLARA
            label: (there is correct answer, but we have yes/no values)
        
        '''
        
            
        out_dict['UserID'] = int(user_id)
        out_dict['TargetItemID'] = int(target_item)
        out_dict['InteractedNum'] = len(purchase_history)
        out_dict['InteractedItemIDs'] = [int(item) for item in purchase_history]
        
        out_dict['InteractedItemTitles'] = purchase_history_titles
        out_dict['TargetItemTitle'] = target_item_title
        
        out_dict['label'] = target_text
        
        return out_dict
    