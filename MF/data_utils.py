import pickle
import os
import json
import html
import gzip
import re
import sys


def build_explain_prompt_beauty(purchase_history, target_item_profile):
    
    return generate_prompt({"instruction": "Given the profiles of the purchasing history of this consumer, can you provide a reason for why this consumer purchased the current product? Answer with one sentence with the following format: \"The consumer purchased this product because ...\"","input": f"Profiles of Purchasing History: {purchase_history}\n\nCurrent Product Profile: {target_item_profile}\n"}), generate_prompt({"instruction": "Given the profiles of the purchasing history of this consumer, can you provide a reason for why this consumer did not purchase the current product? Answer with one sentence with the following format: \"The consumer did not purchase this product because ...\"","input": f"Profiles of Purchasing History: {purchase_history}\n\nCurrent Product Profile: {target_item_profile}\n"})

# APR 21: Clothing domain prompt (for fashion, apparel, accessories)
def build_explain_prompt_clothing(purchase_history, target_item_profile):
    return generate_prompt({"instruction": "Given the profiles of the purchasing history of this consumer, can you provide a reason for why this consumer purchased the current product? Answer with one sentence with the following format: \"The consumer purchased this product because ...\"","input": f"Profiles of Purchasing History: {purchase_history}\n\nCurrent Product Profile: {target_item_profile}\n"}), generate_prompt({"instruction": "Given the profiles of the purchasing history of this consumer, can you provide a reason for why this consumer did not purchase the current product? Answer with one sentence with the following format: \"The consumer did not purchase this product because ...\"","input": f"Profiles of Purchasing History: {purchase_history}\n\nCurrent Product Profile: {target_item_profile}\n"})


# APR 21: General-purpose Yelp business prompt (for all types of services or businesses)
def build_explain_prompt_yelp_general(purchase_history, target_item_profile):
    return generate_prompt({"instruction": "Given the profiles of the visit history of this customer, can you provide a reason for why this customer visited the current business? Answer with one sentence with the following format: \"The customer visited this business because ...\"","input": f"Profiles of Visited History: {purchase_history}\n\nCurrent Business Profile: {target_item_profile}\n"}), generate_prompt({"instruction": "Given the profiles of the visit history of this customer, can you provide a reason for why this customer did not visit the current business? Answer with one sentence with the following format: \"The customer did not visit this business because ...\"","input": f"Profiles of Visited History: {purchase_history}\n\nCurrent Business Profile: {target_item_profile}\n"})


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_prompt(data_point):
    if data_point["input"]:
        # noqa: E501
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
    
    else:
        # noqa: E501
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        
def clean_text(text):
    # Convert HTML entities like &amp; and &nbsp;
    text = html.unescape(text)
    # Replace non-breaking spaces (\xa0) with regular spaces
    text = text.replace('\xa0', ' ')
    text = re.sub(u"\\<.*?\\>", "", text)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
        
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def readTargetItem(path):
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            line = f.read()
            targetItem = [i.replace("'", "") for i in line.split(", ")]
        
        return targetItem
    else:
        print(f"Target File Missing! Create and save inside {path}!")
        sys.exit(1)