from torch.utils.data import DataLoader, Sampler
import random
import numpy as np
from multiprocessing import Pool
import torch
from torch.utils.data.distributed import DistributedSampler

from data_utils import AmazonData, YelpData, load_prompt


def get_dataset_object(sample_numbers,mode='train', dataset='toys',data_path="../data", sample_type='random',local_rank=0, num_groups = 5):
    
    if dataset == 'yelp':
        data_obj = YelpData(sample_numbers=sample_numbers,mode=mode, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank,num_groups = num_groups)
    
    else:
        data_obj = AmazonData(sample_numbers=sample_numbers,mode=mode, dataset=dataset, data_path=data_path, sample_type=sample_type,local_rank=local_rank,num_groups = num_groups)
    
    return data_obj
    
def get_dataset_loader(data_obj, tokenizer, prompt_path, mode='train',batch_size=16,workers=4,distributed=False,max_epochs=1,shuffle=False,fair_reweight=False):
    if distributed:
        sampler = DistributedSampler(data_obj)
    else:
        sampler = None
    
    prompt_list = load_prompt(prompt_path)    
    if workers == 0:
        workers = 1
    
    if mode == 'train':
        max_steps = max_epochs*(len(data_obj)//batch_size)//workers
        print(f"Max STEPS: {max_steps}")
        collater = TrainCollater(prompt_list=prompt_list,
                                 llm_tokenizer = tokenizer,
                                 train = True,
                                 max_steps=max_steps,
                                 fair_reweight=fair_reweight,
                                )
    else:
        collater = TrainCollater(prompt_list=prompt_list,
                                 llm_tokenizer = tokenizer,
                                 train=False,
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
                 terminator="",
                 fair_reweight=False,
                 max_steps=1):
        
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_steps
        self.fair_reweight = fair_reweight
        self.cur_step = 1

    def __call__(self, batch):
        
        self.llm_tokenizer.padding_side = 'left'
        instruction = random.choice(self.prompt_list)
        inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p < thresh_hold or not self.train:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title +' [HistoryEmb]' for seq_title in sample['InteractedItemTitles']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[ItemHere]' in input_text:
                    insert_prompt= sample['TargetItemTitle'] +' [ItemEmb]'
                    input_text=input_text.replace('[ItemHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title + ' [PH]' for seq_title in sample['InteractedItemTitles']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[ItemHere]' in input_text:
                    insert_prompt = sample['TargetItemTitle'] + " [PH]"
                    input_text=input_text.replace('[ItemHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1
        
        targets_text = [sample['label'] for sample in batch]
        
        MAXLEN = max([entry['InteractedNum'] for entry in batch])

        if self.train:
            targets_text=[target_text+self.terminator for target_text in targets_text]
            # inputs_pair = [p + ' ' + t for p, t in zip(inputs_text, targets_text)] # CORRECTED JULY 25: add + and space in place of pair up
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            
            
            
            new_batch={"tokens":batch_tokens,
                       'user_id':[sample['UserID'] for sample in batch],
                       'inputs_text': inputs_text,
                       "seq":torch.stack([torch.tensor(pad_history(sample['InteractedItemIDs'],MAXLEN)) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['InteractedNum']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['TargetItemID']) for sample in batch], dim=0),
                       "flag":flag,
                       'targets_text':targets_text
                       }
            if self.fair_reweight:
                new_batch["TargetItemID"] = np.array([entry['TargetItemID'] for entry in batch])
                
            
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            
            new_batch={"tokens":batch_tokens,
                       'user_id':[sample['UserID'] for sample in batch],
                       'inputs_text': inputs_text,
                       "TargetItemID":np.array([entry['TargetItemID'] for entry in batch]),
                       "seq":torch.stack([torch.tensor(pad_history(sample['InteractedItemIDs'],MAXLEN)) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['InteractedNum']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['TargetItemID']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       'targets_text':targets_text,
                       }
        return new_batch
                 
