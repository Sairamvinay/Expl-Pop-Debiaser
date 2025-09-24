# TALLRec + FairPrompt -  Evaluation Script for Beauty and Yelp
# can replace beauty with sports. Can use Clean condition checkpoint_path

# Beauty
python3 evaluate_fair_prompts.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-beauty --data_path ../data/ --model_name TallRec-FAIR_PROMPTS --dataset beauty --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 1 >> outputs/beauty/beauty-FAIR-PROMPTS-EVAL.txt

# Yelp
python3 evaluate_fair_prompts.py --base_model meta-llama/Llama-3.2-1B --checkpoint_path snap/llama-yelp/ --data_path ../data/ --model_name TallRec-FAIR_PROMPTS --dataset yelp --output_dir ../top-preds/ --batch_size 100 --maxlen 1 --seed 999 --num_workers 0 --gpu 0 >> outputs/yelp/yelp-FAIR-PROMPTS-EVAL.txt
