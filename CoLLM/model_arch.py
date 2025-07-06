import logging
import random
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os
import warnings
import numpy as np

from transformers import LlamaTokenizer, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict, get_peft_model_state_dict, PeftModel
from peft.utils.config import PeftConfig
from rec_base_models import MatrixFactorization

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

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def unwrap_peft_model(model):
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):  # torch.compile wrapped model
        model = model._orig_mod
    return model

def verify_loaded_peft(model):
    print(f"🔍 Model class: {type(model)}")
    sd = get_peft_model_state_dict(model)
    lora_keys = [k for k in sd if 'lora_' in k]
    print(f"🔍 Found {len(lora_keys)} LoRA weights.")
    if lora_keys:
        mean_val = sd[lora_keys[0]].abs().mean().item()
        print(f"🔍 Sample LoRA param '{lora_keys[0]}': mean={mean_val:.6f}")
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🧠 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
        if mean_val < 1e-5:
            print("⚠️ LoRA weights are very small — possibly untrained?")
        else:
            print("✅ LoRA adapter appears correctly loaded and trained.")
        

def load_model_checkpoint(resume_from_checkpoint, model,device_map):
    """
    Load LoRA adapter weights into a PEFT-wrapped model.
    This assumes the checkpoint is a LoRA-only adapter (saved via get_peft_model_state_dict).
    """
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔁 Loading LORA checkpoint from: {resume_from_checkpoint}")

        model = PeftModel.from_pretrained(
            model,
            resume_from_checkpoint,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    
        verify_loaded_peft(model)
    
        cfg = PeftConfig.from_pretrained(resume_from_checkpoint)
        print(f"✅ Adapter loaded: task={cfg.task_type}, cfg: {cfg}")


        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")
        
    else:
        print("No checkpoint path was provided! Training with random initialized weights!!")

    return model

def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_>=0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_



class CoLLM(nn.Module):
    
    
    def __init__(
        self,
        rec_model="MF",
        user_num=22363,
        item_num=12101,
        embedding_size=64,
        freeze_rec=True,
        llama_model="meta-llama/Llama-3.2-1B",
        prompt_path=None,
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj","v_proj"],
        lora_dropout=0.2,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False,
        llama_ckpt=None,
        rec_ckpt = None,
        stage = 1, # 1 for loading llama_proj and rec_encoder ; 2 for loading the llama_model_lora ; -1 for loading all
        fair_reweight=False,
    ):
        super().__init__()
        AUTH_TOKEN = "YOUR_HF_TOKEN"

        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.device_8bit = device_8bit

        print("runing MiniGPT4Rec_v2 ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        rec_config = {"user_num":user_num, "item_num":item_num, "embedding_size":embedding_size}
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, 'fp16')
        # try:
        
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train.__get__(self.rec_encoder, self.rec_encoder.__class__)
            logging.info("freeze rec encoder")
            print("freeze rec encoder")

        print('Loading Rec_model Done')

        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model,use_auth_token=AUTH_TOKEN,use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        
        ADDED = False
        if self.llama_tokenizer.unk_token is None:
            self.llama_tokenizer.unk_token = "<unk>"
        if "<unk>" not in self.llama_tokenizer.get_vocab():
            self.llama_tokenizer.add_tokens(["<unk>"])
            ADDED = True
            
        if self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': self.device_8bit},
                token=AUTH_TOKEN
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                token=AUTH_TOKEN
            )
        
        if ADDED:
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        
        print("UNKNOWN: ",self.llama_tokenizer.unk_token)
        print("UNKNOWN TOKEN ID:",self.llama_tokenizer.unk_token_id)
        
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        
#         for name, module in self.llama_model.named_modules():
#             if "proj" in name:
#                 print(name)
        print('Loading LLAMA Done')

        self.use_lora = use_lora
        self.fair_reweight= fair_reweight
        
        
        if use_lora:
            print("Setting Lora")
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            ) 
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
        
        
        if freeze_lora:
            print("freeze lora...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False

       
        # for normal 
        
        if self.rec_encoder is not None:
            print("type:", type(proj_mid), proj_mid)
            self.llama_proj = nn.Sequential(
                nn.Linear(embedding_size, embedding_size*int(proj_mid)),  # ml100=>5
                nn.ReLU(),
                nn.Linear(embedding_size*int(proj_mid), self.llama_model.config.hidden_size * self.proj_token_num),
            )
        
        else:
            self.llama_proj = None

        
        if freeze_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train.__get__(self.llama_proj, self.llama_proj.__class__)
            logging.info("!!!! freeze llama_proj...")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt=False
        self.print_labels = False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode=False
            self.prompt_list_p = None
            self.print_debug = False
        else:
            self.prompt_list = []
            self.prompt_list_p = None
            
        
        if stage == 1:
            if rec_ckpt:
                self.load_checkpoint(rec_ckpt, stage=stage)
        
        elif stage == 2:
            if llama_ckpt:
                self.load_checkpoint(llama_ckpt, stage=stage)
        
        # during evaluation stage
        else:
            if llama_ckpt:
                self.load_checkpoint(llama_ckpt, stage=2)
            
            if rec_ckpt:
                self.load_checkpoint(rec_ckpt, stage=1)
        
        
        ans_type = 'v2'
        self.set_answer_type(mode=ans_type)
        self.print_prompt()
        print("#Params Size: ",self.show_n_params())
    
    def load_checkpoint(self, url_or_filename, stage = 1):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if os.path.exists(url_or_filename):
            pass
        else:
            raise RuntimeError(f"checkpoint url or path {url_or_filename} is invalid")

        
        if stage == 1:
            checkpoint = torch.load(url_or_filename)
            # load the pretrained MF and Projection weights if existent
            msg = self.rec_encoder.load_state_dict(checkpoint['rec_encoder'], strict=False)
            print("Missing keys after loading recommender encoder: ",msg.missing_keys)
            msg = self.llama_proj.load_state_dict(checkpoint['llama_proj'], strict=False)
            print("Missing keys after loading llama projector: ",msg.missing_keys)
            
        elif stage == 2:
            # load the LoRA model weights:
            self.llama_model_lora = load_model_checkpoint(url_or_filename, self.llama_model, device_map={'': self.device_8bit})
        
        else:
            pass
        
        print("load checkpoint from %s" % url_or_filename)

        return
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def show_n_params(self, return_str=True):
        
        print(f"{'Layer':<60} {'Shape':<30} {'Requires Grad'}")
        print("-" * 100)
        for name, param in self.named_parameters():
            print(f"{name:<60} {str(param.shape):<30} {param.requires_grad}")

        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot
    
    
    
    def after_evaluation(self, **kwargs):
        pass

    
    def init_rec_encoder(self,rec_model, config, precision):
        if rec_model == "MF":
            print("### rec_encoder:", "MF")
            rec_model = MatrixFactorization(**config)
        
        else:
            rec_model = None
            warnings.warn(" the input rec_model is not MF, we won't utilize the rec_encoder directly.")
            # raise NotImplementedError("the current version olny supports the following models: MF,...")
        return rec_model

    
    
    def to_be_trained(self):
        if self.use_lora:
            return True
        # return True # have lora module, will be trained anyway
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        return False
    
    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode
    
    
    def set_answer_type(self,mode):
        if mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']

            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:",pos_ans_id, "neg ids:", neg_ans_id)
            
        else:
            raise NotImplementedError("not implement this types of answers")
    
    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))


    
    # Takes in a sample from the dataset (batch of data probably): and then processes it accordingly
    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = self.device # sample['UserID'].device

        with autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            
            
            user_embeds = self.rec_encoder.user_encoder(torch.tensor(sample['UserID']).to(device), all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(torch.tensor(sample['TargetItemID']).to(device), all_items=all_item_embeds).unsqueeze(-2)
            
            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order)==3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'], all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)

                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds,dim=1)              
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad']==self.rec_encoder.padding_index, 0, idx_flag) # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones([idx_flag.shape[0],1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag,dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)

                #adding consitence loss
                
                 

                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:,0],idx_nopad[:,1]].reshape(-1, hidden_size),
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                }
        sample_atts_llama = None

        return sample_embeds_llama, sample_atts_llama

    

    # takes samples and embeddings and then ensures user_id, itemtitlelist and targetitemid and title are all fixed with unknown and then their corresponding embedding from rec_encoder(). this was derived from encode_recdata_v2()
    def recprompt_wrap_v2(self, embeddings, ori_samples, prompt): # used for stage 2
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token #"<unk>"
            unk_ = ".".join([unk_]*self.proj_token_num)
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<TargetItemID>", unk_)


            prompt_list = []
            
            
            for k in range(batch_size):
                prompt_ = prompt + ""
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_]*ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                prompt_list.append(prompt_)
            
            if not self.has_print_prompt:
                print("prompt example:", prompt_list[0])
                self.has_print_prompt = True
            
            # print(prompt_list[0])
            
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device) # .to(ori_samples['UserID'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            
            if not self.has_pri_decode:
                print("#######prompt decoded example:",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                print("prompt tokens tokenized: ",prompts_tokens.input_ids)
                self.has_pri_decode = True
                
            
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
            
            # ADDED: JUN 8 to ensure we avoid truncation issue
            
            # Extract all left indices
            present = replaced_idx[:, 0].tolist()
            missing = [i for i in range(batch_size) if i not in present]
            
            # Assume the last index to use (e.g., self.max_txt_len - 1)
            last_index =  self.max_txt_len - 1
            
            # Construct missing rows
            missing_rows = torch.tensor([[i, last_index] for i in missing], device=replaced_idx.device)

            # Concatenate and sort by the left index
            replaced_idx = torch.cat([replaced_idx, missing_rows], dim=0)
            replaced_idx = replaced_idx[replaced_idx[:, 0].argsort()]
            replaced_idx = torch.tensor(sorted(replaced_idx.tolist(), key=lambda x: (x[0], x[1])))
            replaced_idx = torch.tensor(replaced_idx, dtype=torch.long).to(self.device)
            
            
            if not self.print_debug:
                print("replaced idx: ",replaced_idx)
                self.print_debug = True
            
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori  and "<ItemIDList>" in prompt_ori and  "<TargetItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = embeddings['merged_embs']
            elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
                try:
                    prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = torch.cat([embeddings['User_emb'], embeddings['TargetItem_emb']],dim=-2).reshape(-1,embeddings['User_emb'].shape[-1])
                except Exception as e:
                    print("Error :",e)
                    print("batch info: ",ori_samples)
                    print("replaced idx: ",replaced_idx)
                    print("prompt example:", prompt_list[:])
                    print("#######prompt decoded example:",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[:])))
                    exit(1)

            else:
                pass 
            return prompt_embeds, prompts_tokens.attention_mask, prompt_list
            
            



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
    
    def forward(self,samples):
        if self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("None-template version has not been implementned...")  


   
    
    def prompt_based_encode_v2(self,prompt, samples):
        id_orders = get_ids_order(prompt)
        embeddings_encode, _ = self.encode_recdata_v2(samples,ids_order=id_orders)
        sample_embeds, atts_samples, prompt_tokens = self.recprompt_wrap_v2(embeddings_encode, samples, prompt)
        return sample_embeds, atts_samples, prompt_tokens
        

    def prompt_with_p(self,p):
        if self.prompt_list_p is None:
            prompt_list_p= []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]]*p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p


    def forward_v2(self, samples):
        user_selective_prompts = False
        if self.prompt_list:
            
            prompt = random.choice(self.prompt_with_p([5,5,5,1])) #[1,5,3,1]  #[2,5,3,1]
            sample_embeds, atts_samples, prompt_texts = self.prompt_based_encode_v2(prompt,samples)
            

        self.llama_tokenizer.padding_side = "right"
        device = self.device #samples_encode['User_emb'].device

        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]} # yes/no mapping

        text = [ans_[int(t)] for t in samples["label"]] 

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        

        # new loss, just focus on the target pos and neg tokens 
        pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
        # logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        # loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(samples['label']).float().to(device))
        
        if not self.print_labels:
            print("targets: ",targets)
            self.print_labels = True
        
        if self.fair_reweight:
            return {"logits":outputs.get('logits'),'labels':targets}
        else:
            loss = outputs.loss
            return {"loss": loss}
    
    def predict_samples(self, samples):
        user_selective_prompts = False
        if self.prompt_list:
            prompt = self.prompt_list[0]
            sample_embeds, atts_samples, _ = self.prompt_based_encode_v2(prompt,samples)
                

        self.llama_tokenizer.padding_side = "right"

        device = self.device # samples['UserID'].device

        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]

        ans_ = {1:pos_ans, 0:neg_ans}


        text = [ ans_[int(t)]  for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with autocast():
            
            if not self.use_lora:

                beam_outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
            else:
                beam_outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
        
        outputs = {"logits":beam_outputs.logits[:,-t_posi,:]}
        
        return outputs
        
    
    def predict_samples_generate(self, samples, generation_kwargs = {}):
        user_selective_prompts = False
        if self.prompt_list:
            prompt = self.prompt_list[0]
            sample_embeds, atts_samples, _ = self.prompt_based_encode_v2(prompt,samples)
                

        self.llama_tokenizer.padding_side = "right"

        device = self.device # samples['UserID'].device

        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]

        ans_ = {1:pos_ans, 0:neg_ans}


        text = [ ans_[int(t)]  for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with autocast():
            
            if not self.use_lora:
                beam_outputs = self.llama_model.generate(inputs_embeds = inputs_embeds,
                                                         attention_mask = attention_mask,
                                                         return_dict_in_generate=True,
                                                         output_scores=True,
                                                         **generation_kwargs
                                                        )

            else:
                beam_outputs = self.llama_model_lora.generate(inputs_embeds = inputs_embeds,
                                                         attention_mask = attention_mask,
                                                         return_dict_in_generate=True,
                                                         output_scores=True,
                                                         **generation_kwargs
                                                        )
        
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
        
        
        return beam_outputs

    def generate_for_samples_v2(self, samples,return_all=False):
        user_selective_prompts = False
        if self.prompt_list:
            prompt = self.prompt_list[0]
            sample_embeds, atts_samples, _ = self.prompt_based_encode_v2(prompt,samples)
                

        self.llama_tokenizer.padding_side = "right"

        device = self.device # samples['UserID'].device

        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]

        ans_ = {1:pos_ans, 0:neg_ans}


        text = [ ans_[int(t)]  for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)


        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]

        logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        # loss = nn.functional.binary_cross_entropy_with_logits(logits_, torch.tensor(samples['label']).float().to(device))
        loss = outputs.loss

        if return_all:
            return outputs, logits_

        return {"loss": loss, 'logits':logits_}
    

    def generate_for_samples(self,samples):
        if self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        else:
            raise NotImplementedError("Not implement the default version")     


    
