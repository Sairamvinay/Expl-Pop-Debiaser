import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler

# ==========================
# 7. Evaluation Data: 

'''
Note: JUN 17

Test.pkl for all files have ids start from 0 to user_num - 1
Train.pkl and Val.pkl for all files have ids from 1 to user_num
'''

# ==========================

def get_eval_dataset_loader(datamaps, expls, targetItems, user_num, item_num, data_path = '../data/', dataset = 'beauty', mode='test',batch_size=100,workers=0, shuffle=False):
    
    data_obj = ExplTestDataset(expls = expls, targetItems = targetItems, user_num = user_num, item_num=item_num, datamaps = datamaps, data_path = data_path, dataset = dataset) 
    
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


class ExplTestDataset(Dataset):
    def __init__(self, datamaps, expls, targetItems, user_num, item_num, data_path = '../data/', dataset = 'beauty'):
        self.sequential_data = ReadLineFromFile(os.path.join(data_path, dataset, 'sequential_data.txt'))
        self.negative_samples = ReadLineFromFile(os.path.join(data_path, dataset, 'negative_samples.txt'))
        
        self.user2id = datamaps['user2id']
        self.sample_numbers = 100
        
        self.keys = list(expls.keys())
        self.expls = expls
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
        
        record = self.expls[(user - 1, item)] # user - 1 for all test.pkl files.
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
        out_dict['pos-item'] = int(sequence[-1])

        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['UserID'] = np.array([entry['UserID'] for entry in batch])
        batch_entry['ItemID'] = np.array([entry['ItemID'] for entry in batch])
        
        batch_entry['pos-expl'] = [entry['pos-expl'] for entry in batch]
        batch_entry['neg-expl'] = [entry['neg-expl'] for entry in batch]
        
        batch_entry['label'] = np.array([entry['label'] for entry in batch])
        batch_entry['pop-label'] = np.array([entry['pop-label'] for entry in batch])
        
        batch_entry['pos-item'] = np.array([entry['pos-item'] for entry in batch])
        
        return batch_entry

PROMPT_MAPS = {'beauty':build_explain_prompt_beauty,'clothing':build_explain_prompt_clothing,'yelp':build_explain_prompt_yelp_general}

THRESHOLDS = {'beauty':4, 'clothing':4,'yelp':4}
MAX_PREFS  = {'beauty':4, 'clothing':4,'yelp':0}


class ExplGenData:
    def __init__(self, topK_path, dataset, model_name = 'TallRec-Clean',data_path = '../data',max_pref= 4):
        self.data_path = data_path
        self.dataset = dataset
        self.max_pref = max_pref # max. number of liked/disliked items to represent
        self.topK_path = topK_path
        self.model_name = model_name

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
        
        new_path = os.path.join(self.data_path, self.dataset, "liked-disliked-items.pkl")
        if os.path.exists(new_path):
            self.likes_dislikes = load_pickle(new_path)
            print("Loaded Likes/Dislikes:",len(self.likes_dislikes))
        else:
            self.review_data = load_pickle(os.path.join(self.data_path, self.dataset,"review_splits.pkl"))
            self.review_data = self.review_data['train'] + self.review_data['val']
            self.likes_dislikes = self.compute_preference_lists()
            save_pickle(self.likes_dislikes, new_path)
        
        
        # ==========================
        # Step 1: Load the R^K data
        # ==========================
        
        file_path = os.path.join(self.topK_path, f"{self.model_name}-{self.dataset}-preds.pkl")
        print("Loading data from: ",file_path)
        self.topK_recommendations = self.get_topK_recommendations(file_path)
        self.build_expl_inputs = PROMPT_MAPS[self.dataset]
        self.positive_items = self.get_positive_items()
        print("len(positive items):",len(self.positive_items))
        
    
    def get_positive_items(self):
        # Take all users chosen from the topK since we align these together
        return {user:self.get_original(user) for user in self.topK_recommendations}
    
    def get_original(self, user):
        sequential_datum = self.sequential_data[user]
        sequence = sequential_datum.split()
        sequence = [int(_) for _ in sequence]
        assert user == int(sequence[0]) - 1
        test_item = int(sequence[-1]) # only test data that's why and convert to str since we match
        
        return test_item
    
    
    def compute_preference_lists(self):
        rating_info = {}
        for datum in tqdm(self.review_data):
            user_desc = datum['reviewerID']
            item_name = datum['asin']
            user_id = self.user2id[user_desc]
            item_id = self.item2id[item_name]
            rating = datum['overall']
            if user_id not in rating_info:
                rating_info[user_id] = {"liked":[],"disliked":[]}
            if rating <= THRESHOLDS[self.dataset]:
                rating_info[user_id]['disliked'].append(item_id)
            else:
                rating_info[user_id]['liked'].append(item_id)
        
        print("len(rating_info):",len(rating_info))
        return rating_info

    def get_title(self, target_item):
        if self.dataset == 'yelp':
            return clean_text(self.meta_data[self.meta_dict[self.id2item[str(target_item)]]].get('name','unknown title'))
        
        else:
            return clean_text(self.meta_data[self.meta_dict[self.id2item[str(target_item)]]].get('title','unknown title'))
    
    def get_desc(self,item):
        if self.dataset == 'yelp':
            pass
        else:
            return clean_text(self.meta_data[self.meta_dict[self.id2item[str(target_item)]]].get('description',''))
    
    def generate_item_desc(self,item):
        return self.get_desc(item)
    
    
    def generate_item_profile(self,item):
        return "\"" + self.get_title(item) + "\""
    
    
    def get_topK_recommendations(self,file_path):
        user_recommendations = load_pickle(file_path)
        user_recommendations = user_recommendations['ui_scores']
        
        return {user: sorted(items, key=items.get, reverse=True) for user, items in user_recommendations.items()}
    
    def get_feedback(self, user, item):
        sequential_datum = self.sequential_data[user]
        sequence = sequential_datum.split()
        sequence = [int(_) for _ in sequence]
        assert user == int(sequence[0]) - 1
        test_item = int(sequence[-1]) # only test data that's why
        
        return int(test_item == int(item)) # bool saying true/false for +ve item
    
    def generate_expl_dataset(self):

        # ==========================
        # Step 2: Format into HF based dataset (source: https://github.com/huggingface/trl/blob/v0.6.0/examples/notebooks/gpt2-sentiment-control.ipynb)
        # ==========================

        expl_dataset = {}
        pos_sample_count = 0
        for user in tqdm(self.topK_recommendations):
            liked_items = self.likes_dislikes[str(user + 1)]['liked'] # likes/dislikes starts with 1
            pos_item = self.positive_items[user]
            # items = self.topK_recommendations[user] if int(pos_item) in self.topK_recommendations[user] else self.topK_recommendations[user][:-1] + [int(pos_item)] # Forcibly include pos_item into TOPK by replacing with the last item (Kth).
            # items = self.topK_recommendations[user] if int(pos_item) in self.topK_recommendations[user] else self.topK_recommendations[user] + [int(pos_item)]
            items = self.topK_recommendations[user]
            for item in items:
                
                item_profile = self.generate_item_profile(item)
                label = self.get_feedback(user,item)
                pos_sample_count += label
                random.shuffle(liked_items)
                
                liked_items = liked_items[:self.max_pref]                
                liked_profiles = ', '.join([self.generate_item_profile(x) for x in liked_items])
                
                pos_prompt, neg_prompt = self.build_expl_inputs(liked_profiles, item_profile)
                expl_dataset[(user,item)] = {"pos_prompt":pos_prompt,"neg_prompt":neg_prompt,'target_item':item_profile,"label":label}
        
        print("#Num Pos: ",pos_sample_count)
        assert pos_sample_count == len(self.topK_recommendations) # align one GT item per user
        return expl_dataset

    
