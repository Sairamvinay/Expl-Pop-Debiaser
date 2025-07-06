import os
import random
import numpy as np
from collections import Counter,defaultdict
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler


def get_dataset_loader(expls, targetItems, user_num, item_num, mode = 'train',batch_size=16,workers=4, shuffle=False, distributed=False):
    
    data_obj = ExplBPRDataset(expls = expls, targetItems = targetItems, user_num = user_num, item_num=item_num) 
    
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


class ExplBPRDataset(Dataset):
    def __init__(self, expls, targetItems, user_num, item_num):
        
        self.expls = expls
        self.keys = list(expls.keys())
        self.targetItems = set(targetItems)
        self.user_num = user_num + 1
        self.item_num = item_num + 1
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        user, item = self.keys[idx]
        record = self.expls[(user, item)]
        pos_text = record['pos-expl']
        neg_text = record['neg-expl']
        label = record['label']
        pop_label = 1 if int(item) in self.targetItems else 0
        
        out_dict = {}
        out_dict['UserID'] = int(user)
        out_dict['ItemID'] = int(item)
        out_dict['pos-expl'] = pos_text
        out_dict['neg-expl'] = neg_text
        out_dict['label'] = int(label)
        out_dict['pop-label'] = int(pop_label)

        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['UserID'] = np.array([entry['UserID'] for entry in batch])
        batch_entry['ItemID'] = np.array([entry['ItemID'] for entry in batch])
        
        batch_entry['pos-expl'] = [entry['pos-expl'] for entry in batch]
        batch_entry['neg-expl'] = [entry['neg-expl'] for entry in batch]
        
        batch_entry['label'] = np.array([entry['label'] for entry in batch])
        batch_entry['pop-label'] = np.array([entry['pop-label'] for entry in batch])
        
        return batch_entry



PROMPT_MAPS = {'beauty':build_explain_prompt_beauty,'clothing':build_explain_prompt_clothing,'yelp':build_explain_prompt_yelp_general}

THRESHOLDS = {'beauty':4, 'clothing':4,'yelp':4}
MAX_PREFS  = {'beauty':4, 'clothing':4,'yelp':0}


class ExplGenTrainData:
    def __init__(self, dataset,data_path = '../data',mode='train'):
        self.data_path = data_path
        self.dataset = dataset
        self.sample_type = 'random'
        self.MAX_HISTORY_LEN = 5
        
        
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
        
        self.build_expl_inputs = PROMPT_MAPS[self.dataset]
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
                num_items = len(self.user_items[user]) - 1 # exclude test
                for j in range(num_items * 2):
                    self.datum_info.append((curr,int(user)-1,j))
                    self.total_length += 1
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
    
    
    def generate_expl_dataset(self):
        expl_dataset = {}
        pos_sample_count = 0
        for idx in tqdm(range(self.total_length)):
            record = self.single_item(idx)
            user,item = record['user_id'],record['item_id']
            label = record['label']
            expl_dataset[(user,item)] = record
            pos_sample_count += label
        
        print("#Num Pos: ",pos_sample_count)
        print("#Total Samples:", self.total_length)
        return expl_dataset
    
    def single_item(self, idx):
        
        out_dict = {}
        
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        datum_idx = datum_info_idx[1]
        
        sequential_datum = self.sequential_data[datum_idx]
        sequence = sequential_datum.split()
        user_id = sequence[0]
        candidate_item_idx = datum_info_idx[2] 
        
        
        pos_items = self.user_items[user_id][:-1]
        neg_items = self.train_negative[int(user_id) - 1][1:]
        
        assert int(user_id) - 1 == datum_idx
        assert user_id == self.train_negative[int(user_id)-1][0]
        
        pos_items = [int(x) for x in pos_items]
        neg_items = [int(x) for x in neg_items]
        
        if self.mode == 'train':
            purchase_history = list(pos_items)
        
        else:
            raise NotImplementedError
        
        
        if candidate_item_idx < len(pos_items):
            # +ve item
            target_item = pos_items[candidate_item_idx]
            target_text = "Yes"
            target_label = 1
            target_item_title = self.generate_item_profile(target_item)
            
            purchase_history = [item for item in purchase_history if item != target_item]
            
        else:
            # -ve item
            
            target_item = neg_items[(candidate_item_idx % len(pos_items))]
            target_label = 0
            target_text = "No"
            target_item_title = self.generate_item_profile(target_item)
    
        purchase_history = np.random.choice(purchase_history,min(self.MAX_HISTORY_LEN,len(purchase_history)), replace=False)
        purchase_history = purchase_history.tolist()
        purchase_history_titles = [self.generate_item_profile(item) for item in purchase_history]
        
        pos_prompt, neg_prompt = self.build_expl_inputs(purchase_history=', '.join(purchase_history_titles), target_item_profile=target_item_title)
        
        out_dict = {"pos_prompt":pos_prompt,"neg_prompt":neg_prompt,'target_item':target_item_title,
                    "label":target_label, 'user_id':int(user_id) - 1,'item_id':int(target_item)}
        
        return out_dict
        
        
        
        
        