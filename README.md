# LLM-Pop-Bias

## Data

Steps to reproduce
1) Download processed data from [here](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view). Credits: (P5)[https://github.com/jeykigung/P5] 
2) Place in `data/` folder.
3) Create target items as per `notebook-data-creation/beauty-targetItem-10%-MAY29.ipynb` and related notebooks for other datasets. Save the resulting `targetItems.txt` file inside `data/$DATASET$` directory.
4) Create negative items for BPR training for our method using `notebook-data-creation/beauty-negativeBPRCandidates.ipynb` and related notebooks for other datasets.  Save the resulting `train-negatives.pkl` it inside `data/$DATASET$` directory.
5) For generating explanations, please follow `Holistic-Explainer/shell/gen-expl-runs.txt`. Remember to use the HF token for accessing LlaMa models from this step onwards (1B / 7B-HF)

## Baselines
Please follow `TALLRec/` ([source](https://github.com/SAI990323/TALLRec)) and `CoLLM/` ([source](https://github.com/zyang1580/CoLLM)) for further code.

## Our method

Please check `Holistic-Explainer/shell` for running scripts.
