# FIPO
FIPO: Free-form Instruction-oriented Prompt Optimization with Preference Dataset and Modular Fine-tuning Schema

## Environment
We provide [core_requirement.txt](core_requirement.txt) for your convenience.

## Model Weights
The initial models we used are [Tulu2 models](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101). Below are the model weights of our fine-tuned version. We follow same [AI2 ImpACT Low-risk license](https://allenai.org/impact-license) consistent with Tulu2.

| Name | Share Link |
| --- | --- |
| IPL-IPO-70B-Full | TBD |
| IPL-IPO-70B-Lora | TBD |

## Workflow
### Dataset Diversification
```
Run `python3 codes/generate_new_data.py`.
```

### FIPO fine-tuning
```
Run `bash scripts/sft_train.sh TYPE` for fine-tuning with SFT strategy. TYPE can be 1 or 0, and 1 is for full fine-tuning and 0 is for lora fine-tuning.
Run `bash scripts/dpo_train.sh TYPE METHOD` for fine-tuning with DPO/IPO strategy. METHOD can be `sigmoid` or `ipo`, in which `sigmoid` is for DPO and `ipo` is for IPO.
Run `bash scripts/ipl_train.sh TYPE METHOD` for fine-tuning with IPL strategy.
```

### FIPO evaluation
```
Run `bash scripts/test_inference.sh` for evaluation.
```

## Acknowledgement
We thank [AI2](https://allenai.org) for providing the Tulu2 models, [TRL](https://github.com/huggingface/trl/tree/main) for straightforward applications of DPO/IPO, and [VLLM](https://github.com/vllm-project/vllm) for simple yet efficient LLM batch inference.

## Citation
```
TBD
```
