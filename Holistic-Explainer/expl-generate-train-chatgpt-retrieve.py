import torch
import argparse
import os
import csv
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
    
    parser.add_argument('--debug', action='store_true', help='Debugging flag')    
    
    args = parser.parse_args()
    return args


def process_batch(args,start,end,batch_job_id,client):
    batch_job = client.batches.retrieve(batch_job_id)
    print("BATCH JOB: ",batch_job)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    result_file_name = os.path.join(f"{args.output_dir}", args.dataset,"output-files", f'batch_results_{args.dataset}-start-{start}-end-{end}.jsonl')
    if os.path.exists(result_file_name):
        pass
    else:
        with open(result_file_name, 'wb') as file:
            file.write(result)
    
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results

def main(args):
    # ==========================
    # 1. Load LLM + Tokenizer + LoRA
    # ==========================
    
    start_time = time()
    # device = torch.device(f"cuda:{args.local_rank}")
    seedSet(args.seed)
    
    os.makedirs(os.path.join(f"{args.output_dir}", args.dataset,"output-files"), exist_ok=True)
    
    API_KEY = "YOUR_API_KEY"  # YOUR OPENAI API_KEY to be added
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
    
    print("="*50)
    print("Retrieving Samples from served API requests")
    
    # Open the CSV file
    with open("batches-retrieve.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        results = []
        for idx, row in tqdm(enumerate(reader)):
            print(f"Row {idx}: batch_id={row['batch_id']}, start={row['start']}, end={row['end']}, is_missing={row['is_missing']}")
            start = row['start']
            end = row['end']
            batch_id = row['batch_id']
            result = process_batch(args=args,start=start,end=end,batch_job_id=batch_id,client=client)
            results += result
    
    print("Len(results): ",len(results))
    
    save_path = os.path.join(args.output_dir,args.dataset,f"train.pkl")
    if os.path.exists(save_path):
        explanations = load_pickle(save_path)
        print(f"loading explanations: length: {len(explanations)}")
    else:
        explanations = {}
    
    for res in tqdm(results[:], desc="Retrieving explanations"):
        task_id = res['custom_id']
        vals = task_id.split('-')
        user_id, item_id, expl_type = int(vals[1]),int(vals[2]), str(vals[3])
        result = res['response']['body']['choices'][0]['message']['content']
        
        result = clean_text(result)
        
        key = (user_id, item_id)
        if key not in explanations:
            explanations[key] = {'pos-expl':'','neg-expl':'','label':train_expl_prompt_data[key]['label'],'target_item':train_expl_prompt_data[key]['target_item']}
        
        new_key = f"{expl_type}-expl"
        explanations[key][new_key] = result
        
        if args.debug:
            print("User ID, Item ID: ",user_id, ",", item_id)
            print(f"{expl_type.upper()} Explanation Result: {explanations[key]}")

    
    del train_expl_prompt_data
    
    
    save_pickle(explanations, save_path)
    print(f"Saved Train Explanations to {save_path}: {len(explanations)}")
    
    print("Retrieving Complete")
    print(f'It took {time() - start_time:.1f}s')
    return



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    args = parse_args()
    print("ARGS: ",args)
    main(args)
