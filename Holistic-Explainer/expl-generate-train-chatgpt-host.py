import torch
import argparse
import os
import openai
from time import time
from utils import seedSet
import json
from data_utils import load_pickle,save_pickle,clean_text
from train_dataset import ExplGenTrainData
import random
from tqdm import tqdm
import wandb
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for CHATGPT expl generation script")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    
    parser.add_argument('--output_dir', type=str, default="generated-expls-chatgpt/", help='Directory to save generated explanations')
    
    parser.add_argument('--maxlen', type=int, default=50, help='Maximum sequence length')
    
    parser.add_argument('--seed', type=int, default=999, help='Seed for random number generation')
    # parser.add_argument("--local-rank",default=-1,type=int, help='local-rank (GPU)')
    parser.add_argument("--start",default=0,type=int,help='Starting index')
    parser.add_argument("--end",default=-1,type=int,help='End index')
    parser.add_argument("--temperature",type=float,default=0.8,help='Temperature for GPT text generation')
    parser.add_argument("--frequency_penalty",type=float,default=0.5,help='Frequency Penalty for GPT to avoid repetitive words')
    parser.add_argument('--debug', action='store_true', help='Debugging flag')    
    
    args = parser.parse_args()
    return args



def main(args):
    # ==========================
    # 1. Load LLM + Tokenizer + LoRA
    # ==========================
    
    start = time()
    # device = torch.device(f"cuda:{args.local_rank}")
    seedSet(args.seed)
    
    
    API_KEY = "YOUR_KEY"  # YOUR OPENAI API_KEY to be added
    client = openai.OpenAI(api_key=API_KEY)
    MODEL_NAME = "gpt-4.1-mini"
    
    # ====================================================
    # Step 3: Load the Expl Training Data and Format Accordingly
    # ====================================================
    train_data_obj = ExplGenTrainData(dataset = args.dataset, data_path = args.data_path, mode = 'train',chatgpt=True,maxlen=args.maxlen)
    train_expl_prompt_data = train_data_obj.generate_expl_dataset()
    
    del train_data_obj
    
    print("Total number of samples: ",len(train_expl_prompt_data))
    
    
    # -------------------
    # 4. Initialize Frozen Explanation Generator
    # -------------------
    
    print("Expected #keys: ",len(train_expl_prompt_data.keys()))

    path_name = os.path.join(args.output_dir,args.dataset, f"train-keys.pkl")
    if os.path.exists(path_name):
        keys = load_pickle(path_name)
        print(f"Loaded from {path_name}")
    
    else:
        keys = list(train_expl_prompt_data.keys())[:]
        os.makedirs(os.path.join(f"{args.output_dir}", args.dataset), exist_ok=True)
        random.shuffle(keys)
        print("Generated after running data")
        save_pickle(keys,os.path.join(args.output_dir,args.dataset, f"train-keys.pkl"))
        
    # Steps: 
    # 1) Generate +ve Explanations from frozen LLM    
    # 2) Generate -ve Explanations from frozen LLM
    # -------------------
    # 5. Generate Explanations
    # -------------------
    
    
    print("="*50)
    print("Curating Samples for API requests")
    
    tasks = []
    
    start = args.start
    end = len(keys) if args.end == -1 else args.end
    NUM_SAMPLES = end - start
    print(f"#Samples: {NUM_SAMPLES}")
    
    

    for i in tqdm(range(start, end), desc="Generating explanations"):
        torch.cuda.empty_cache()
        key = keys[i]
        pos_batch_prompts = train_expl_prompt_data[key]['pos_prompt']
        system_prompt_pos, description_pos = pos_batch_prompts[0], pos_batch_prompts[1]
        
        neg_batch_prompts = train_expl_prompt_data[key]['neg_prompt']
        system_prompt_neg, description_neg = neg_batch_prompts[0], neg_batch_prompts[1]
        
        user_id, item_id = int(key[0]),int(key[1])
        pos_task = {"custom_id":f"task-{user_id}-{item_id}-pos",
                "method":"POST",
                "url":"/v1/chat/completions",
                "body":{
                    "model":MODEL_NAME,
                    "temperature":args.temperature,
                    "frequency_penalty":args.frequency_penalty,
                    "messages":[{"role": "system","content": system_prompt_pos},{"role": "user","content": description_pos}],
                    "max_tokens":args.maxlen,
                }
            }

        tasks.append(pos_task)
        neg_task = {"custom_id":f"task-{user_id}-{item_id}-neg",
                "method":"POST",
                "url":"/v1/chat/completions",
                "body":{
                    "model":MODEL_NAME,
                    "temperature":args.temperature,
                    "frequency_penalty":args.frequency_penalty,
                    "messages":[{"role": "system","content": system_prompt_neg},{"role": "user","content": description_neg}],
                    "max_tokens":args.maxlen,
                }
            }

        tasks.append(neg_task)
        if args.debug:
            print("System Expl Prompts: ",pos_batch_prompts[0],neg_batch_prompts[0])
            print("Pos Explanation Prompt: ",description_pos)
            print("Neg Explanation Prompt: ",description_neg)
        
    
    print("Number of tasks: ",len(tasks))
    file_name = os.path.join(f"{args.output_dir}", args.dataset,f'batch_{args.dataset}-start-{start}-end-{end}.jsonl')
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    
    batch_file = client.files.create(file=open(file_name, "rb"),purpose="batch")
    
    print("Batch File: ",batch_file)
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print("BATCH JOB: after creation:",batch_job)

    del train_expl_prompt_data
    
    print("Hosting Complete")
    print(f'It took {time() - start:.1f}s')
    return



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    args = parse_args()
    print("ARGS: ",args)
    main(args)
