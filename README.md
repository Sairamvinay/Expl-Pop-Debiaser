# LLM-Pop-Bias

## Paper Accepted at WISE 26
- Sairamvinay Vijayaraghavan, Lei Li, Prasant Mohapatra. [Leveraging Holistic Explanations to Mitigate Popularity Bias]. International Web Information Systems Engineering conference (WISE26), 2026 [Accepted].
  
## Data

Steps to reproduce
1) Download processed data from [here](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view). Credits: [P5](https://github.com/jeykigung/P5) 
2) Place in `data/` folder.
3) Create target items as per `notebook-data-creation/beauty-targetItem-10%-MAY29.ipynb` and related notebooks for other datasets. Save the resulting `targetItems.txt` file inside `data/$DATASET$` directory.
4) Create negative items for BPR training for our method using `notebook-data-creation/beauty-negativeBPRCandidates.ipynb` and related notebooks for other datasets.  Save the resulting `train-negatives.pkl` it inside `data/$DATASET$` directory.
5) For generating explanations, please follow `Holistic-Explainer/shell/gen-expl-runs.txt`. Remember to use the HF token for accessing LlaMa models from this step onwards (1B / 7B-HF)

## Baselines
Please follow `TALLRec/` ([source](https://github.com/SAI990323/TALLRec)), `CoLLM/` ([source](https://github.com/zyang1580/CoLLM)) and `LLARA` ([source](https://github.com/ljy0ustc/LLaRA)) for original instructions. 
Anyways, please refer `shell` inside each directory for complete instructions (it is almost identical in all the baselines). Run tuning first and then training and finally evaluation scripts. 
a) `tuning.sh`: for tuning Vanilla

b) `tuning-fairness-reweight.sh`: for tuning FAIR-IPS variants

c) `beauty-run.sh`: for training vanilla models

d) `beauty-fairness-reweight.sh`: for training FAIR-IPS models

e) `eval.sh`: evaluate vanilla models

f) `eval-IPS.sh`: evaluate Fair-IPS models

g) `eval-fair-prompts.sh`: evaluate Fair-Prompt models

## Our method: Holistic-Explainer (Expl-Debias)
For this section, please refer to subdirectories inside `Holistic-Explainer/`

Please check `shell` for running scripts.

a) Check `gen-expl-runs.sh` for LLaMa based explanation generation.

b) (Optional) See `chatgpt-run.sh` and `chatgpt-run-test.sh` for hosting ChatGPT completions. Then `chatgpt-retrieve.sh` for retrieving requests from OpenAI API.

c) Tuning BPR Stage 1: See `tuning-BPR-simple.sh`

d) Train Stage 1: Use found HP and use into `beauty-run.sh` (only first command) 

e) Tuning Contrastive Explanation Stage 2: Use checkpoint from d) into `tuning-BPR-contrexpl.sh`

f) Train Stage 2: See `beauty-run.sh` (only second command) 

g) Evaluate both Stages: see `eval.sh`

For generating and working through 5.3 check `beauty-diff-encoders.sh` and `beauty-diff-generators.sh` and the corresponding `eval-beauty-diff-encoders.sh` for evaluation.
For evaluating different generators: use  `eval.sh` but change the "--expl_path" parameter

Similarly, for the ablation studies, check `analysis-studies` for Section 5.4. 
Also, Section 5.2 work related to positive/negation explanation ablation is found inside `ablation-pos-neg-expls/` directory which also has the diagrams/table data.

We also attached the explanation fidelity evaluation (Section 4.6) in `explanation_eval/` while we also provide our Appendix work regarding internal representations in `embed_study/`
