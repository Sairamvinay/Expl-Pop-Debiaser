import os
import sys
import torch
from peft import PeftModel
from peft.utils.config import PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig,logging
from test_data_loading import get_dataset_loader
import argparse
from time import time
import random
import numpy as np
from utils import *
import inspect

logging.set_verbosity_error()  # Only show errors, suppress warnings/info

'''
# Notes As per MAR 17: 

we get a tuple of two tensors for .scores attribute: we use always the second tensor (it is probably post-processed after logits).

we use only m(beam_outputs.scores[1][:, [yes_token, no_token]])[:, 0].cpu()

Using scores[0] gives extremely small values and softmax will probably yield 50% for both tokens (yes/no).


ALWAYS use TEMPERATURE in GENERATION CONFIG: else we always get maximum focus on token 128000 (Start of sentence). use temperature something around 0.3-0.8. 

If not, we always focus on the first token and no randommness at all!

yes: 9891
no: 2201
Yes: 9642
No: 2822

'''


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Argument parser for evaluation script")

    parser.add_argument('--base_model', type=str, required=True, help='Base model path or name: eg: meta-llama/Llama-3.2-1B')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint directory')
    parser.add_argument('--seed',type=int,default=999,help='Seed value')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset: beauty/yelp/clothing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--maxlen', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel dataloader workers')
    parser.add_argument("--gpu", type=int, default=0, help='GPU ID')
    parser.add_argument("--temperature", type=float, default=0.7, help='Temperature for LLM output generation')
    parser.add_argument("--top_p", type=float, default=0.9, help='Top_p for LLM output generation')
    parser.add_argument("--top_k", type=int, default=50, help='Top_k for LLM output generation')
    parser.add_argument("--model_name",type=str,default='TallRec-Clean',help='Name of Model to save the results: Such as TallRec-Clean')
    parser.add_argument("--output_dir",type=str,default='outputs/',help='Path to save the predictions')

    args = parser.parse_args()
    return args



def evaluate_model(item2id, base_model, checkpoint_dir, data_path, dataset, batch_size, maxlen, gpu, load_8bit=False,num_workers=1,num_beams=1,model_name='TallRec-Clean',output_dir='outputs',seed = 999,temperature=0.7,top_p=0.9,top_k=50):
    seedSet(seed)
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("Arguments passed to evaluate_model():")
    for arg in args:
        if arg != 'item2id':
            print(f"\t {arg} = {values[arg]}")
    
    start = time()
    
    path = os.path.join(data_path, dataset, "targetItems.txt")
    popitems = readTargetItem(path)
    print("Sample Items: ",list(popitems)[:5])
    
    popitems = [int(item2id[item]) for item in popitems]

    print("# Target Items: ",len(popitems))
    print("Sample Items: ",list(popitems)[:5])
    
    device_map={"": gpu}
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    model = PeftModel.from_pretrained(
            model,
            checkpoint_dir,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    
    verify_loaded_peft(model)
    
    cfg = PeftConfig.from_pretrained(checkpoint_dir)
    print(f"✅ Adapter loaded: task={cfg.task_type}, cfg: {cfg}")


    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name}")
            

    tokenizer.padding_side = "left"  # Allow batched inference
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compile the model!")
        model = torch.compile(model)
    
    # Load validation/test data
    eval_loader, _ = get_dataset_loader(
        tokenizer, 
        sample_numbers=batch_size,  # Load test set since 100 is #candidate items
        dataset=dataset, 
        data_path=data_path, 
        batch_size=batch_size, 
        workers=num_workers,
        local_rank=gpu
    )
    yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token = tokenizer.encode("No", add_special_tokens=False)[0]
    
    
    print("Yes Token ID:", yes_token, "No Token ID:", no_token)
    total_loss = 0
    num_batches_eval = len(eval_loader)
    
    all_info = []
    golds,preds = [],[]
    
    dim = 0 # this gives the actual model outputs: 1 gives already normalized: not very well distributed

    m = torch.nn.Softmax(dim=1)

    generation_config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
    
    debug = False

    with torch.no_grad():
        for step,batch in tqdm(enumerate(eval_loader), desc="Evaluating"):

            torch.cuda.empty_cache()
            beam_outputs = model.generate(
                input_ids=batch['input_ids'].to(gpu), 
                attention_mask=batch['attention_mask'].to(gpu),
                max_new_tokens=maxlen, # setting to 1 yields scores to return only one value instead of 2 value tuple.
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
                use_cache=True,
            )
            
            if dim == 1:

                print("Beam output scores dimension 1: for Yes/No (no softmax): ",beam_outputs.scores[1][:, [yes_token, no_token]])
                # 9642 is Yes, 2822 is No for LLaMA 1B
                gen_yes_probs = m(beam_outputs.scores[1][:, [yes_token, no_token]])[:, 0].cpu()
                print("After softmax dimension 1 one dimension: ",gen_yes_probs)
                vals, indices = torch.sort(gen_yes_probs, descending=True)
                print("Sorted with dimension 1: ",vals)
                print("Indices with dimension 1: ",indices)
                s = beam_outputs.sequences

                output = tokenizer.batch_decode(s, skip_special_tokens=True)
                output = [_.split('Response:\n')[-1] for _ in output]

                print("Generated Sequences with dimension 1 (Decoded):", output)
            else:
                
                scores = beam_outputs.scores[0].softmax(dim=-1)
                gen_yes_probs = torch.tensor(scores[:,[yes_token, no_token]], dtype=torch.float32).softmax(dim=-1)
                logits = gen_yes_probs[:,0]
                _, indices = torch.sort(logits, descending=True)
                
                if debug:
                    s = beam_outputs.sequences
                    output = tokenizer.batch_decode(s, skip_special_tokens=True)
                    output = [_.split('Response:\n')[-1] for _ in output]
                    print("STEP ",step)
                    print("Generated Sequences with dimension 0 (Decoded):", output)
                    print("Beam output scores dimension 0: for Yes/No (no softmax): ",beam_outputs.scores[0][:, [yes_token, no_token]])
                    print("After softmax dimension 0 one dimension: ",gen_yes_probs)                
                    print("Sorted with dimension 0: ",_)
                    print("Indices with dimension 0: ",indices)
                    print('='*100)
                else:
                    del gen_yes_probs
                    del scores
                    del beam_outputs
                    torch.cuda.empty_cache()
                
                
            new_info = {}
            new_info['target_item'] = [batch['original_item'][0]] # same for entire batch
            new_info['gen_item_list'] = [batch['item_ids'][_] for _ in indices[:batch_size]]
            golds.extend([int(batch['item_ids'][_]==new_info['target_item'][0]) for _ in range(batch_size)])
            preds.extend(logits.cpu().tolist())
            
            all_info.append(new_info)  
            

        
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
    
    
    save_path = os.path.join(output_dir,f"{model_name}-{dataset}-preds.pkl")
    save_pickle({'ui_scores':ui_scores,'gt':gt, 'golds':golds, 'preds':preds},save_path)
    top = [1,2,3,5,10,20]
    
    print("Recommendation Performance")
    _, Recommendresults = getBasicScores(ui_scores, gt, top)
    print("\nAUC: ",area_curve_metric(golds,preds)) 
    
    print("Fairness Performance")
    FairResults = getFairnessScores(ui_scores, popitems, top,len(item2id))
    print(f'It took {time() - start:.1f}s')
    return {'recommend_results':Recommendresults, 'fair_results':FairResults}



if __name__ == "__main__":
    args = parse_eval_args()

    print("Starting Evaluation with args:", args)
    
    datamaps = load_json(f"{args.data_path}{args.dataset}/datamaps.json")
    

    evaluate_results = evaluate_model(
        item2id=datamaps['item2id'],
        base_model=args.base_model,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_path,
        data_path=args.data_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        top_k=args.top_k,
        maxlen=args.maxlen,
        gpu=args.gpu,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    
    print("Total Results: ",evaluate_results)
    
    print("Evaluation Complete")
