# LLaRa Evaluation Script for Beauty and Yelp
# can replace beauty with sports. Fix the rec_dim, llama_ckpt, rec_ckpt and lora_r and lora_alpha accordingly


# BEAUTY
python3 -u evaluate-rec.py --data_path ../data/ --model_name LLARA-Clean --dataset beauty --batch_size 100 --maxlen 784 --prompt_path prompts/llara_amazon.txt --rec_dim 128 --lora_r 16 --lora_alpha 16 --seed 999 --num_workers 2 --llama_ckpt snap/llama-beauty  --rec_ckpt ../MF/snap/beauty/checkpoint_epoch_BEST.pt --proj_ckpt snap/llama-beauty/PROJ/PROJ_checkpoint_epoch_BEST.pt --output_dir ../top-preds/ --gpu 3 --temperature 1 >> outputs/beauty/beauty-EVAL.txt

# YELP
python3 -u evaluate-rec.py --data_path ../data/ --model_name LLARA-Clean --dataset yelp --batch_size 100 --maxlen 784 --prompt_path prompts/llara_yelp.txt --rec_dim 128 --lora_r 32 --lora_alpha 16 --seed 999 --num_workers 2 --llama_ckpt snap/llama-yelp  --rec_ckpt ../MF/snap/yelp/checkpoint_epoch_BEST.pt --proj_ckpt snap/llama-yelp/PROJ/PROJ_checkpoint_epoch_BEST.pt --output_dir ../top-preds/ --gpu 3 --temperature 1.0 >> outputs/yelp/yelp-EVAL.txt
