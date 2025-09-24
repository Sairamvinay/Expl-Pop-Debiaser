# CoLLM Evaluation Script for Beauty and Yelp
# can replace beauty with sports. Fix the rec_dim, llama_ckpt and lora_r and lora_alpha accordingly

# Beauty
python3 -u evaluate-rec.py --data_path ../data/ --model_name CoLLM-Clean --dataset beauty --batch_size 100 --maxlen 784 --prompt_path prompts/collm_amazon.txt --rec_dim 256 --lora_r 16 --lora_alpha 16 --project_mid 10 --seed 999 --num_workers 2 --llama_ckpt ../TALLRec/snap/llama-beauty/  --rec_ckpt snap/llama-beauty/stage-2/checkpoint_epoch_BEST.pt --output_dir ../top-preds/ --gpu 0 >> outputs/beauty/beauty-EVAL.txt

# YELP
python3 -u evaluate-rec.py --data_path ../data/ --model_name CoLLM-Clean --dataset yelp --batch_size 100 --maxlen 784 --prompt_path prompts/collm_yelp.txt --rec_dim 256 --lora_r 32 --lora_alpha 16 --project_mid 10 --seed 999 --num_workers 2 --llama_ckpt ../TALLRec/snap/llama-yelp/  --rec_ckpt snap/llama-yelp/stage-2/checkpoint_epoch_BEST.pt --output_dir ../top-preds/ --gpu 1 >> outputs/yelp/yelp-EVAL.txt