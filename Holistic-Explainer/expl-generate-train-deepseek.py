import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from utils import seedSet
from data_utils import load_pickle,save_pickle,clean_text
from train_dataset import ExplGenTrainData
import random
from tqdm import tqdm
import wandb
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for RL training script")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    
    parser.add_argument('--output_dir', type=str, default="generated-expls-deepseek/", help='Directory to save generated explanations')
    
    parser.add_argument('--maxlen', type=int, default=50, help='Maximum sequence length')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=999, help='Seed for random number generation')
    parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')

    parser.add_argument('--debug', action='store_true', help='Debugging flag')
    parser.add_argument('--log_wandb', action='store_true', help='log_wandb flag')    
    
    args = parser.parse_args()
    return args



def main(args):
    # ==========================
    # 1. Load LLM + Tokenizer + LoRA
    # ==========================
    
    start = time()
    device = torch.device(f"cuda:{args.local_rank}")
    seedSet(args.seed)
    
    if args.log_wandb:
        wandb.init(project='explanation-generation-deepseek',config=vars(args),reinit=True)

    # =============================================================
    # Step 1. Basic Reward Model (RM) Setup
    # =============================================================
    
    # model_name = "deepseek-ai/deepseek-llm-7b-base"
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    
    AUTH_TOKEN = "YOUR_HF_TOKEN"
    
    # =============================================================
    # Step 2. Basic Tokenizer Setup
    # =============================================================
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=AUTH_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left' # ADDED BY ME to avoid error for decoder-only model
    
    generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.pad_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": args.maxlen, # TRL SAYS 32 | specify how many tokens you want to generate at most
}

    print("GEN KWARGS: ",generation_kwargs)
    
    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id = tokenizer.convert_tokens_to_ids("No")
    
    
    # ====================================================
    # Step 3: Load the Expl Training Data and Format Accordingly
    # ====================================================
    train_data_obj = ExplGenTrainData(dataset = args.dataset, data_path = args.data_path, mode = 'train')
    train_expl_prompt_data = train_data_obj.generate_expl_dataset()
    
    
    del train_data_obj
    
    print("Total number of samples: ",len(train_expl_prompt_data))
    
    
    # -------------------
    # 4. Initialize Frozen Explanation Generator
    # -------------------
    gen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,use_auth_token=AUTH_TOKEN)
    gen_model.to(device)
    gen_model.eval()
    
    print("Expected #keys: ",len(train_expl_prompt_data.keys()))
    # Steps: 
    # 1) Generate +ve Explanations from frozen LLM    
    # 2) Generate -ve Explanations from frozen LLM
    # -------------------
    # 5. Generate Explanations
    # -------------------
    if args.debug:
        NUM_SAMPLES = 500
    else:
        NUM_SAMPLES = len(train_expl_prompt_data.keys())
    
    keys = list(train_expl_prompt_data.keys())[:NUM_SAMPLES]
    random.shuffle(keys)
    
    num_batches = math.ceil(len(keys) / args.batch_size)
    print(f"#Samples: {NUM_SAMPLES}")
    print(f"#Batches: {num_batches}")
    
    print("="*50)
    print("TRAINING")
    
    save_path = os.path.join(args.output_dir,args.dataset,f"train.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        explanations = load_pickle(save_path)
        print(f"Loaded Train Explanations from {save_path}: {len(explanations)}")
    else:
        explanations = {}
        print("Generating Training Explanations")

        for i in tqdm(range(0, len(keys), args.batch_size), desc="Generating explanations"):
            torch.cuda.empty_cache()
            curr_keys = keys[i:min(i + args.batch_size,len(keys))]
            pos_batch_prompts = [train_expl_prompt_data[key]['pos_prompt'] for key in curr_keys]
            neg_batch_prompts = [train_expl_prompt_data[key]['neg_prompt'] for key in curr_keys]
            
            # Positive Explanations
            pos_enc = tokenizer(pos_batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            pos_enc = {k: v.to(device) for k, v in pos_enc.items()}

            with torch.no_grad():
                positive_outputs = gen_model.generate(**pos_enc, **generation_kwargs)
            
            # Negative Explanations
            neg_enc = tokenizer(neg_batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            neg_enc = {k: v.to(device) for k, v in neg_enc.items()}

            with torch.no_grad():
                negative_outputs = gen_model.generate(**neg_enc, **generation_kwargs)
            
            # slice out newly generated tokens beyond the prompt
            for key, pos_inp_ids,neg_inp_ids, pos_out_ids, neg_out_ids in zip(curr_keys,pos_enc['input_ids'],neg_enc['input_ids'], positive_outputs,negative_outputs):
                pos_gen_ids = pos_out_ids[len(pos_inp_ids):]
                pos_text = tokenizer.decode(pos_gen_ids, skip_special_tokens=True)
                
                neg_gen_ids = neg_out_ids[len(neg_inp_ids):]
                neg_text = tokenizer.decode(neg_gen_ids, skip_special_tokens=True)
                explanations[key] = {'pos-expl':clean_text(pos_text),'neg-expl':clean_text(neg_text),'label':train_expl_prompt_data[key]['label'],'target_item':train_expl_prompt_data[key]['target_item']}

            if args.debug:
                print("Pos/Neg Expl Prompt: ",pos_batch_prompts[0],neg_batch_prompts[0])
                print("Pos Query shape:", [len(x) for x in pos_enc['input_ids']])
                print("Neg Query shape:", [len(x) for x in neg_enc['input_ids']])
                print("Pos/Neg Explanations: ",[(explanations[k]['pos-expl'],explanations[k]['neg-expl']) for k in curr_keys])
                print('='*50)
            if args.log_wandb:
                table = wandb.Table(columns=["user","item","target_item","pos-explanation","neg-explanation","label"])
                for k in curr_keys[:10]:
                    table.add_data(
                        k[0],
                        k[1],
                        explanations[k]['target_item'],
                        explanations[k]['pos-expl'],
                        explanations[k]['neg-expl'],
                        explanations[k]['label'],
                    )
                
                wandb.log({"explanations": table}, step=i)
            
            del pos_enc
            del neg_enc
            del pos_batch_prompts
            del neg_batch_prompts
            del positive_outputs
            del negative_outputs
            del curr_keys
            torch.cuda.empty_cache()
        
        if not args.debug:
            save_pickle(explanations, save_path)
            print(f"Saved Train Explanations to {save_path}: {len(explanations)}")
        
    del train_expl_prompt_data
    
    # =============================================================
    # Log Explanations to W&B
    # =============================================================
    if args.log_wandb:
        wandb.finish()
        
    
    
    del gen_model
    
        
    print("Generation Complete. Model saved.")
    print(f'It took {time() - start:.1f}s')



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    args = parse_args()
    print("ARGS: ",args)
    main(args)
