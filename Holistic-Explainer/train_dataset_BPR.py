import os
import random
import numpy as np
from collections import Counter,defaultdict
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler


def get_BPRdataset_loader(dataset, data_path, mode='train',batch_size=16,workers=4, shuffle=False, distributed=False):
    
    data_obj = SimpleBPRDataset(dataset = dataset, data_path = data_path, mode=mode) 
    
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


class SimpleBPRDataset:
    def __init__(self, dataset,data_path = '../data',mode='train'):
        self.data_path = data_path
        self.dataset = dataset        
        self.mode = mode

        # ====================================================================
        # Step 0: Load some basic information: Seq data, titles, meta info
        # ====================================================================
        self.sequential_data = ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'sequential_data.txt'))
        
        if dataset == 'yelp':
            self.meta_data = load_pickle(os.path.join(self.data_path, self.dataset, 'meta_data.pkl'))
            self.meta_dict = {}
            for i, meta_item in enumerate(self.meta_data):
                self.meta_dict[meta_item['business_id']] = i
        
        else:
            self.meta_data = []
            for meta in parse(os.path.join(self.data_path, self.dataset, 'meta.json.gz')):
                self.meta_data.append(meta)
            self.meta_dict = {}
            for i, meta_item in enumerate(self.meta_data):
                self.meta_dict[meta_item['asin']] = i        
        
        
        datamaps = load_json(os.path.join(self.data_path, self.dataset, "datamaps.json"))
        
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.id2item = datamaps['id2item']
        
        self.user_num = len(self.user2id)
        self.item_num = len(self.item2id)
        
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
        
        self.train_negative = load_pickle(os.path.join(self.data_path, self.dataset, 'train-negatives.pkl'))
        
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        
        if self.mode == 'train':
            self.total_length = 0
            curr = 0
            for i,user in enumerate(list(self.user_items.keys())):
                num_items = len(self.user_items[user]) - 2 # exclude val and test
                for j in range(num_items):
                    self.datum_info.append((curr,int(user)-1,j))
                    self.total_length += 1
                    curr += 1
            
        elif self.mode == 'val':
            self.total_length = len(self.user2id)
            curr = 0
            for i,user in enumerate(list(self.user_items.keys())):
                self.datum_info.append((curr,int(user)-1,len(self.user_items[user]) - 2))
                curr += 1

        else:
            raise NotImplementedError

    def get_title(self, target_item):
        if self.dataset == 'yelp':
            return clean_text(self.meta_data[self.meta_dict[self.id2item[str(target_item)]]].get('name','unknown title'))
        
        else:
            return clean_text(self.meta_data[self.meta_dict[self.id2item[str(target_item)]]].get('title','unknown title'))
    
    def __len__(self):
        return self.total_length
    
    def generate_item_profile(self,item):
        return "\"" + self.get_title(item) + "\""
    
    def __getitem__(self, idx):
        
        out_dict = {}
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        item_idx = datum_info_idx[2]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        
        assert int(user_id) - 1 == datum_idx
        assert user_id == self.train_negative[int(user_id)-1][0]
        
        pos_items = self.user_items[user_id][:-1]
        neg_items = self.train_negative[int(user_id) - 1][1:]
        
        pos_items = [int(x) for x in pos_items]
        neg_items = [int(x) for x in neg_items]
        
        pos_item = np.random.choice(pos_items, 1, replace=False)
        neg_item = np.random.choice(neg_items, 1, replace=False)
        
        pos_item = pos_item.tolist()[0]
        neg_item = neg_item.tolist()[0]
        
        out_dict['UserID'] = int(user_id) - 1
        out_dict['PosItem'] = pos_item
        out_dict['NegItem'] = neg_item
        
        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['UserID'] = np.array([entry['UserID'] for entry in batch])
        batch_entry['PosItem'] = np.array([entry['PosItem'] for entry in batch])
        batch_entry['NegItem'] = np.array([entry['NegItem'] for entry in batch])
        return batch_entry