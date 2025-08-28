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
        for key in lora_keys:
            mean_val = sd[key].abs().mean().item()
            print(f"🔍 Sample LoRA param '{key}': mean={mean_val:.6f} requires_grad={sd[key].requires_grad} ",end=' | ')
            if mean_val < 1e-5:
                print("⚠️ LoRA weights are very small — possibly untrained?")
            else:
                print("✅ LoRA adapter appears correctly loaded and trained.")
        
        return
        
def verify_model_weights(model):
    for name, param in model.named_parameters():
        print(f"{name} → {param.shape} -> {param.data.abs().mean()}")

        
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



class LLARA(nn.Module):
    
    
    def __init__(
        self,
        rec_model="MF",
        user_num=22363,
        item_num=12101,
        embedding_size=64,
        freeze_rec=True,
        llama_model="meta-llama/Llama-3.2-1B",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj","v_proj"],
        lora_dropout=0.2,
        freeze_lora=False,
        freeze_proj=False,
        llama_ckpt=None,
        rec_ckpt = None,
        proj_ckpt = None,
        fair_reweight=False,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.device_8bit = device_8bit
        
        AUTH_TOKEN = "YOUR_HF_TOKEN"

        print("running LLARA ...... ")

        print('Loading Rec_model')
        rec_config = {"user_num":user_num, "item_num":item_num, "embedding_size":embedding_size}
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, 'fp16')
        
        if rec_ckpt:
            self.load_checkpoint(rec_ckpt, stage=1)
        
        
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train.__get__(self.rec_encoder, self.rec_encoder.__class__)
            logging.info("freeze rec encoder")
            print("freeze rec encoder")
        
        else:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = True
            self.rec_encoder = self.rec_encoder.train()
            print("train rec encoder")

        print('Loading Rec_model Done')
        
        ## LLAMA LOADING START
        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model,use_auth_token=AUTH_TOKEN,use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[ItemEmb]']})
        
        self.his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        self.item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        
        
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
        
        
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        

        print('Loading LLAMA Done')
        
        use_lora=True

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
        
        # during evaluation stage / for additional tuning
        if llama_ckpt:
            self.load_checkpoint(llama_ckpt, stage=2)

        
        if freeze_lora:
            print("freeze lora...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False
            
            self.llama_model_lora = self.llama_model_lora.eval()
            # Check trainable parameters
            trainable = sum(p.numel() for p in self.llama_model_lora.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llama_model_lora.parameters())
            print(f"🧠 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
        
        else:
            print("train lora...")
#             for name, param in self.llama_model_lora.named_parameters():
#                 param.requires_grad = True
            
            self.llama_model_lora = self.llama_model_lora.train()
            verify_loaded_peft(self.llama_model_lora)
            
            # Check trainable parameters
            trainable = sum(p.numel() for p in self.llama_model_lora.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llama_model_lora.parameters())
            print(f"🧠 Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
        
        print("Loading LLAMA Done")
        
        ## LLAMA LOADING END
    
        # MADE CHANGES TO ADAPT acc to LLARA
        ## LLAMAPROJ STARTS
        print("Loading Projector")
        self.projector = nn.Sequential(
            nn.Linear(embedding_size, self.llama_model.config.hidden_size), 
            nn.GELU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
        )
        
        if proj_ckpt:
            self.load_checkpoint(proj_ckpt,stage=3)
                
        if freeze_proj:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
            self.projector = self.projector.eval()
            self.projector.train = disabled_train.__get__(self.projector, self.projector.__class__)
            logging.info("!!!! freeze projector...")
            print('Loading PROJECTOR (EVAL CONDITION) Done')
        
        else:
            print("Training PROJECTOR (FINETUNING CONDITION)")
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
            
            self.projector = self.projector.train()
        
        ## LLAMAPROJ END
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        
        
        self.print_prompt=False
        self.print_labels=False
        self.has_pri_decode = False
        self.print_debug = False
        
        ans_type = 'v2'
        self.set_answer_type(mode=ans_type)
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

        # load rec encoder
        if stage == 1:
            checkpoint = torch.load(url_or_filename)
            # load the pretrained MF and Projection weights if existent
            msg = self.rec_encoder.load_state_dict(checkpoint['model'], strict=False)
            print("Missing keys after loading recommender encoder: ",msg.missing_keys)
            print('='*50)
            verify_model_weights(self.rec_encoder)
            
        # load llama lora weights
        elif stage == 2:
            # load the LoRA model weights:
            self.llama_model_lora = load_model_checkpoint(url_or_filename, self.llama_model, device_map={'': self.device_8bit})
        
        # load llama proj weights
        elif stage == 3:
            checkpoint = torch.load(url_or_filename)
            msg = self.projector.load_state_dict(checkpoint['projector'], strict=False)
            print("Missing keys after loading llama projector: ",msg.missing_keys)
            verify_model_weights(self.projector)
            
        else:
            pass
        
        print("load checkpoint from %s" % url_or_filename)

        return
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def show_n_params(self, return_str=True):
        
        print(f"{'Layer':<60} {'Shape':<30} {'Value':<20} {'Requires Grad'}")
        print("-" * 120)
        for name, param in self.named_parameters():
            mean_val = round(param.data.float().abs().mean().item(), 6)            
            print(f"{name:<60} {str(param.shape):<30} {str(mean_val):<20} {param.requires_grad}")

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
    
    def encode_items(self, seq):
        item_rec_embs = self.rec_encoder.item_encoder(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
        
    def embed_tokens(self, token_ids):
        if self.use_lora:
            embeds = self.llama_model_lora.base_model.embed_tokens(token_ids)
        
        else:
            embeds = self.llama_model.embed_tokens(token_ids)
        
        return embeds
    
    def wrap_emb(self, batch):
        device = self.device
        if self.use_lora:
            input_embeds = self.llama_model_lora.get_input_embeddings()(batch["tokens"].input_ids.to(device))
            
        else:
            input_embeds = self.llama_model.get_input_embeddings()(batch['tokens'].input_ids.to(device))
            
        his_item_embeds = self.encode_items(batch["seq"].to(device))
        item_embeds = self.encode_items(batch["item_id"].to(device))
        
        
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == self.his_token_id).nonzero().shape[0]>0:
                idx_tensor = (batch["tokens"].input_ids[i] == self.his_token_id).nonzero().view(-1)
                
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx] = item_emb
                    
            if (batch["tokens"].input_ids[i] == self.item_token_id).nonzero().shape[0]>0:
                idx = (batch["tokens"].input_ids[i] == self.item_token_id).nonzero().item()
                input_embeds[i,idx] = item_embeds[i]
                

        return input_embeds
    
    def forward(self,batch):
        if self.run_mode_ == 'v2':
            return self.forward_v2(batch)
        else:
            raise NotImplementedError("None-template version has not been implemented...")  


    def forward_v2(self, batch):
        if not self.print_prompt:
            print("BATCH info below:")
            print(batch)
            self.print_prompt = True
        
        if not self.has_pri_decode:
            print("#######prompt decoded example:",' '.join(self.llama_tokenizer.batch_decode(batch["tokens"].input_ids[0])))
            print("prompt tokens tokenized: ",batch["tokens"].input_ids)
            self.has_pri_decode = True
        
        if not self.print_debug:
            idx_tensor = (batch["tokens"].input_ids[0] == self.his_token_id).nonzero().view(-1)            
            idx = (batch["tokens"].input_ids[0] == self.item_token_id).nonzero()
            
            if (idx is not None) and (idx_tensor is not None) and idx_tensor.numel() > 0:
                print("History Embed Index: ",idx_tensor)
                print("Target item Embed Index: ",idx)
                self.print_debug = True
            
            del idx_tensor
            del idx
            
            
        
        '''
        Create inputs_embeds ; attention_mask; targets; 
        '''
        device = self.device
        sample_embeds = self.wrap_emb(batch)
        sample_embeds = sample_embeds.to(device)
        atts_samples = batch["tokens"].attention_mask
        atts_samples = atts_samples.to(device)
        
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            batch['targets_text'],
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
        
#         if batch["flag"]:
#             for name, param in self.projector.named_parameters():
#                 param.requires_grad = False
#         else:
#             for name, param in self.projector.named_parameters():
#                 param.requires_grad = True
        
       

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
        

        if not self.print_labels:
            print("targets: ",targets)
            self.print_labels = True
        
        if self.fair_reweight:
            return {"logits":outputs.get('logits'),'labels':targets}
        else:
            loss = outputs.loss
            return {"loss": loss}
    
    def predict_samples(self, batch):
        
        
        '''
        Create inputs_embeds ; attention_mask; targets; 
        '''
        device = self.device
        sample_embeds = self.wrap_emb(batch)
        sample_embeds = sample_embeds.to(device)
        atts_samples = batch["tokens"].attention_mask
        atts_samples = atts_samples.to(device)
        
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            batch['targets_text'],
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
        
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
        
        outputs = {"logits":beam_outputs.logits[:,-t_posi,:]}
        
        return outputs
        
    

    def generate_for_samples_v2(self, batch,return_all=False):
        
        
        '''
        Create inputs_embeds ; attention_mask; targets; 
        '''
        device = self.device
        sample_embeds = self.wrap_emb(batch)
        sample_embeds = sample_embeds.to(device)
        atts_samples = batch["tokens"].attention_mask
        atts_samples = atts_samples.to(device)
        
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            batch['targets_text'],
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
        
        
        '''
        Create input_embeds; attention_mask ; targets; 
        '''

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
        
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]

        logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        loss = outputs.loss

        if return_all:
            return outputs, logits_

        return {"loss": loss, 'logits':logits_}
    

    def generate_for_samples(self,batch):
        if self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(batch)
        else:
            raise NotImplementedError("Not implement the default version")     


    
