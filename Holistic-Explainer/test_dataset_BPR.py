import os
import random
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler

def get_eval_dataset_loader(datamaps, targetItems, user_num, item_num, expls = None, data_path = '../data/', dataset = 'beauty', mode='test',batch_size=100,workers=0, shuffle=False):
    
    data_obj = SimpleBPRTestDataset(targetItems = targetItems, user_num = user_num, item_num=item_num, datamaps = datamaps, data_path = data_path, dataset = dataset, expls = expls) 
    
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


class SimpleBPRTestDataset(Dataset):
    def __init__(self, datamaps, targetItems, user_num, item_num, data_path = '../data/', dataset = 'beauty', expls = None):
        self.sequential_data = ReadLineFromFile(os.path.join(data_path, dataset, 'sequential_data.txt'))
        self.negative_samples = ReadLineFromFile(os.path.join(data_path, dataset, 'negative_samples.txt'))
        
        self.user2id = datamaps['user2id']
        self.sample_numbers = 100
        self.expls = expls
        
        self.user_counts = defaultdict(int)
        self.item_counts = defaultdict(int)
        
        self.val_items = defaultdict(int)
        self.test_items = defaultdict(int)
        for line in self.sequential_data:
            user, items = line.strip().split(' ',1)
            items = items.split(' ')
            items = [int(x) for x in items]
            self.test_items[int(user) - 1] = items[-1] # last chronological item
            items = items[:-1] # exclude test items
            user = int(user)
            self.user_counts[user] = len(items)
            self.val_items[int(user) - 1] = items[-1] # Val item if excluding the test items
            for item in items:
                self.item_counts[item] += 1
        
        
        self.targetItems = set(targetItems)
        self.user_num = user_num + 1
        self.item_num = item_num + 1
        self.datum_info = []
        self.total_length = 0
        self.compute_datum_info()

    def __len__(self):
        return self.total_length
    
    def compute_datum_info(self):
        curr = 0
        self.total_length += len(self.user2id) * self.sample_numbers
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, i // self.sample_numbers, i % self.sample_numbers))
        curr = self.total_length
    
    def get_user_coeff(self, user):
        # +1 since we adjust for embedding in __getitem__
        return (self.user_counts[user + 1] / max(self.user_counts.values()))
    
    def get_item_coeff(self, item):
        return (self.item_counts[item] / max(self.item_counts.values()))
    
    def get_val_items(self):
        return self.val_items
    
    def get_test_items(self):
        return self.test_items

    def __getitem__(self, idx):
        datum_info_idx = self.datum_info[idx]
        datum_idx = datum_info_idx[1]
        candidate_item_idx = datum_info_idx[2] 
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        
        user = int(sequence[0])
        assert user == int(self.negative_samples[user - 1].split(' ', 1)[0])
        candidate_samples = self.negative_samples[user - 1].split(' ', 1)[1].split(' ')
        candidate_samples = [int(_) for _ in candidate_samples]
        items = [int(sequence[-1])] + candidate_samples
        
        item = int(items[candidate_item_idx])
        
        label = int(item == int(sequence[-1]))
        pop_label = 1 if int(item) in self.targetItems else 0
        
        out_dict = {}
        out_dict['UserID'] = int(user) - 1
        out_dict['ItemID'] = int(item)
        out_dict['label'] = int(label)
        out_dict['pop-label'] = int(pop_label)
        out_dict['pos-item'] = int(sequence[-1])
        
        if self.expls:
            key = (int(user) - 1 , int(item))
            out_dict['pos-expl'] = self.expls[key]['pos-expl']
            out_dict['neg-expl'] = self.expls[key]['neg-expl']

        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['UserID'] = np.array([entry['UserID'] for entry in batch])
        batch_entry['ItemID'] = np.array([entry['ItemID'] for entry in batch])
        
        batch_entry['label'] = np.array([entry['label'] for entry in batch])
        batch_entry['pop-label'] = np.array([entry['pop-label'] for entry in batch])
        
        batch_entry['pos-item'] = np.array([entry['pos-item'] for entry in batch])
        if self.expls:
            batch_entry['pos-expl'] = [entry['pos-expl'] for entry in batch]
            batch_entry['neg-expl'] = [entry['neg-expl'] for entry in batch]
        
        return batch_entry
