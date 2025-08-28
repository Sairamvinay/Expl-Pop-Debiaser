import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from packaging import version
from time import time
from datetime import timedelta
import inspect
import argparse

from model_arch import LLARA
from utils import seedSet,save_pickle,readTargetItem,getBasicScores,getFairnessScores,area_curve_metric
from test_data_loading import get_dataset_loader,get_dataset_object

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Argument parser for evaluation script")

    parser.add_argument('--rec_ckpt', type=str, default=None, help='Checkpoint to load recommender and projector model weights alone: give just the model checkpoint .pt file snap/llama-beauty/checkpoint_epoch_7.pt')
    parser.add_argument("--llama_ckpt", type=str, default=None, help="Checkpoint to load LLaMa model weights alone: give just the directory")
    parser.add_argument("--proj_ckpt", type=str, default=None, help="Checkpoint to load Projector model weights alone")
    
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--maxlen', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel dataloader workers')
    parser.add_argument("--gpu", type=int, default=0, help='GPU ID')
    parser.add_argument("--model_name",type=str,default='LLARA-Clean',help='Name of Model to save the results: Such as LLARA-Clean')
    parser.add_argument("--output_dir",type=str,default='../top-preds/',help='Path to save the predictions')
    parser.add_argument("--rec_dim",type=int,default=64,help="Recommender Embedding Size")
    parser.add_argument("--lora_r",type=int,default=8,help="Lora R Value")
    parser.add_argument("--lora_alpha",type=int,default=16,help="Lora Alpha Value")
    parser.add_argument("--prompt_path",type=str, default="prompts/llara_amazon.txt",help='Path to load the prompt styles: e.g.: "prompts/llara_amazon.txt"')
    
    parser.add_argument("--temperature", type=float, default=0.7, help='Temperature for LLM output generation')
    parser.add_argument("--top_p", type=float, default=0.9, help='Top_p for LLM output generation')
    parser.add_argument("--top_k", type=int, default=50, help='Top_k for LLM output generation')
    
    parser.add_argument('--debug',action='store_true', help='Flag to debug')

    args = parser.parse_args()
    return args


def evaluate_llara(seed=999,
                   gpu = 0,
                   dataset = 'beauty',
                   model_name = 'LLARA',
                   data_path = '../data/',
                   rec_dim = 64,
                   prompt_path = 'prompts/llara_amazon.txt',
                   maxlen = 512,
                   output_dir = '../top-preds/',
                   lora_r=16,
                   num_workers=4,
                   lora_alpha=16,
                   lora_dropout = 0.0,
                   llama_ckpt = 'snap/llama-beauty/',
                   rec_ckpt = '../MF/snap/beauty/checkpoint_epoch_BEST.pt',
                   proj_ckpt = "snap/llama-beauty/PROJ/PROJ_checkpoint_epoch_BEST.pt",
                   debug=False,
                   temperature=1,
                   top_p = 0.95,
                   top_k=50,
                  ):
    
    
    seedSet(seed) 
    x, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("Arguments passed to evaluate_llara():")
    for arg in x:
        print(f"\t {arg} = {values[arg]}")
    
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)  # Assign unique GPU to each rank
    
    base_model = "meta-llama/Llama-3.2-1B"
    
    device_map={"": gpu}
    
    # Step 0: Eval setup basics
    start = time()
    
    # Step 1: Data Creation
    
    batch_size = 100
      
    
    eval_dataset = get_dataset_object(sample_numbers = batch_size, dataset=dataset, data_path=data_path, local_rank = gpu)
    
    path = os.path.join(data_path, dataset, "targetItems.txt")
    popitems = readTargetItem(path)
    print("Sample Items: ",list(popitems)[:5])
    
    popitems = [int(eval_dataset.item2id[item]) for item in popitems]

    print("# Target Items: ",len(popitems))
    print("Sample Items: ",list(popitems)[:5])
    
    # Step 2: Model Creation
    model = LLARA(
        rec_model="MF",
        user_num=eval_dataset.user_num + 1,
        item_num=eval_dataset.item_num + 1,
        embedding_size=rec_dim,
        freeze_rec=True,
        freeze_lora=True,
        freeze_proj=True,
        llama_model=base_model,
        max_txt_len=maxlen,
        end_sym='\n',
        low_resource=True,  # use 8 bit
        device_8bit= gpu,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=["q_proj","v_proj"],
        lora_dropout=lora_dropout,
        llama_ckpt=llama_ckpt,
        rec_ckpt=rec_ckpt,
        proj_ckpt=proj_ckpt,
    )

    model = model.to(f"cuda:{gpu}")
    
    model.set_mode('v2')
    
    
    eval_loader = get_dataset_loader(data_obj = eval_dataset, tokenizer = model.llama_tokenizer, prompt_path = prompt_path,mode='test',batch_size = batch_size, workers= num_workers, shuffle = False, distributed=False,maxlen=args.maxlen)
    
    yes_token = model.llama_tokenizer("Yes",add_special_tokens=False).input_ids[0]
    no_token = model.llama_tokenizer("No",add_special_tokens=False).input_ids[0]
    
    print("Yes Token ID:", yes_token, "No Token ID:", no_token)
    
    # Step 4: Evaluation loop
    model.eval()
    
    num_batches_eval = len(eval_loader)
    
    all_info = []
    golds,preds = [],[]
    
    
    # generation_kwargs = {"max_new_tokens":1,"temperature":temperature,"top_k":top_k, "top_p":top_p,"do_sample":True, "pad_token_id":model.llama_tokenizer.pad_token_id}

    with torch.no_grad():
        for stepv, batch in enumerate(tqdm(eval_loader)):
            
            if debug:
                if stepv >= 100:
                    break
            
            # beam_outputs = model.predict_samples_generate(batch,generation_kwargs)
            # scores = beam_outputs.scores[0].softmax(dim=1)
            
            beam_outputs = model.predict_samples(batch)
            beam_outputs['logits'] /= temperature
            scores =  beam_outputs['logits'].softmax(dim=1)
            gen_yes_probs = torch.tensor(scores[:,[yes_token, no_token]], dtype=torch.float32).softmax(dim=-1)
            logits = gen_yes_probs[:,0]
            _, indices = torch.sort(logits, descending=True)

            
            if debug or stepv < 5:
                # s = beam_outputs.sequences
                # output = model.llama_tokenizer.batch_decode(s, skip_special_tokens=True)
                # output = [_.split('\n#Answer: Yes or No')[-1] for _ in output]
                print("STEP ",stepv)
                print("batch: ",batch)
                # print("Generated Sequences with dimension 0 (Decoded):", output)
                print("Scores shape: ",scores.shape)
                print("Beam output scores: for Yes/No (no softmax): ",scores[:, [yes_token, no_token]])
                print("After softmax dimension 0 one dimension: ",gen_yes_probs)                
                print("Sorted with dimension 0: ",_)
                print("Indices with dimension 0: ",indices)
                print('='*100)

            del gen_yes_probs
            del scores
            del beam_outputs
            torch.cuda.empty_cache()
            new_info = {}
            new_info['target_item'] = [batch['original_item'][0]] # same for entire batch
            new_info['gen_item_list'] = [batch['TargetItemID'][_] for _ in indices[:batch_size]]
            golds.extend([int(batch['TargetItemID'][_]==new_info['target_item'][0]) for _ in range(batch_size)])
            preds.extend(logits.cpu().tolist())
            
            all_info.append(new_info)
        

    # Clear unused memory
    torch.cuda.empty_cache()
    
    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info['target_item'][0])]
        pred_dict = {}
        for j in range(len(info['gen_item_list'])):
            try:
                pred_dict[int(info['gen_item_list'][j])] = -(j+1)
            except:
                pass
        ui_scores[i] = pred_dict
    
    
    print("# golds: ",len(golds))
    print("# preds: ",len(preds))
    
    print("ATTACK UI SCORES: ",ui_scores)
    print("ATTACK GT SCORES: ",gt)
    
    if not debug:
        save_path = os.path.join(output_dir,f"{model_name}-{dataset}-preds.pkl")
        save_pickle({'ui_scores':ui_scores,'gt':gt, 'golds':golds, 'preds':preds},save_path)
    
    top = [1,2,3,5,10,20]
    
    print("Recommendation Performance")
    _, Recommendresults = getBasicScores(ui_scores, gt, top)
    print("\nAUC: ",area_curve_metric(golds,preds)) 
    
    print("Fairness Performance")
    FairResults = getFairnessScores(ui_scores, popitems, top,len(eval_dataset.item2id))
    print(f'It took {time() - start:.1f}s')
    return {'recommend_results':Recommendresults, 'fair_results':FairResults}
    
    
    print("Evaluation Complete.")
    print(f'It took {time() - start:.1f}s')
    model.set_mode(None)
    
    return


if __name__ == "__main__":
    args = parse_eval_args()

    print("Starting Evaluation with args:", args)
    
    evaluate_results = evaluate_llara(seed=args.seed,
                   gpu = args.gpu,
                   data_path=args.data_path,
                   dataset=args.dataset,
                   model_name = args.model_name,
                   rec_dim = args.rec_dim,
                   prompt_path = args.prompt_path,
                   maxlen = args.maxlen,
                   output_dir = args.output_dir,
                   lora_r=args.lora_r,
                   num_workers=args.num_workers,
                   lora_alpha=args.lora_alpha,
                   lora_dropout = 0.0,
                   llama_ckpt = args.llama_ckpt,
                   rec_ckpt = args.rec_ckpt,
                   proj_ckpt = args.proj_ckpt,
                   debug = args.debug,
                   temperature= args.temperature,
                   top_p = args.top_p,
                   top_k = args.top_k,
                  )
    
    print("Total Results: ",evaluate_results)
    
    print("Evaluation Complete")

