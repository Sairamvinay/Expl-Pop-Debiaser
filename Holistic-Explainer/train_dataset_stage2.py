import os
import random
import numpy as np
from collections import Counter,defaultdict
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler


def get_dataset_loader(expls, targetItems, user_num, item_num, mode = 'train',batch_size=16,workers=4, shuffle=False, distributed=False):
    
    data_obj = ExplStage2Dataset(expls = expls, targetItems = targetItems, user_num = user_num, item_num=item_num) 
    
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

class ExplStage2Dataset(Dataset):
    def __init__(self, expls, targetItems, user_num, item_num):
        
        self.expls = expls
        self.keys = list(expls.keys())
        self.targetItems = set(targetItems)
        self.user_num = user_num + 1
        self.item_num = item_num + 1
        
        self.index = 0
        self.idx_list = []
        self.compute_records()
        
        self.print_info = False
    
    def isPop(self, item):
        return 1 if int(item) in self.targetItems else 0
    
    def compute_records(self):
        self.records = defaultdict(list)
        self.idx_list = []
        for (user, item) in self.keys:
            sample = self.expls[(user,item)] # a dictionary with keys: pos-expl ; neg-expl; target_item (title of item); label
            self.records[user].append({'pos-expl':sample['pos-expl'],'neg-expl':sample['neg-expl'], 'label':sample['label'],'item':item, "pop-label":self.isPop(item)})
        
        self.users = list(self.records.keys())
        for user in range(len(self.users)):
            for idx in range(len(self.records[user])):
                self.idx_list.append((user,idx))
            
        
        print("#self.records: ",len(self.records))
        return
    
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        
        user, sample_idx = self.idx_list[idx]
        out_dict= {}
        out_dict['UserID'] = user
        samples = self.records[user][sample_idx]
        out_dict['ItemID'] = samples['item']
        out_dict['pos-expl'] = samples['pos-expl']
        out_dict['neg-expl'] = samples['neg-expl']
        out_dict['label'] = samples['label']
        out_dict['pop-label'] = samples['pop-label']
        
        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['UserID'] = np.array([entry['UserID'] for entry in batch])
        batch_entry['ItemID'] = np.array([entry['ItemID'] for entry in batch])
        
        batch_entry['pos-expl'] = [entry['pos-expl'] for entry in batch]
        batch_entry['neg-expl'] = [entry['neg-expl'] for entry in batch]
        
        batch_entry['label'] = np.array([entry['label'] for entry in batch])
        batch_entry['pop-label'] = np.array([entry['pop-label'] for entry in batch])
        
        if not self.print_info:
            print("batch: ",batch_entry)
            self.print_info = True
        
        return batch_entry




