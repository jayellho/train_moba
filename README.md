# Whisper Singlish Fine-Tuning

This repository provides scripts and instructions to hyperparameter-tune and fine-tune OpenAI’s Whisper (large-v3-turbo) model on Singlish conversational speech.

## Datasets:
- Please ensure they go into the same folder structure as this repository - they are already arranged accordingly.
- Access here: 
https://gatech.box.com/s/k7tbfki3meo10w87c7z6gtv6pi3qsdk3


## Models:
- These are finetuned models for your reference if needed.
- Access here: 
https://gatech.box.com/s/k7tbfki3meo10w87c7z6gtv6pi3qsdk3
- To perform your own finetuning, download the Whisper models here onto your machine: https://huggingface.co/openai/whisper-large-v3-turbo
- Place the model folder and all its contents into the `finetune/models` folder in this repository.


## Prerequisites

- **OS:** Linux / Ubuntu  
- **Package Manager:** Conda  
- **Language:** Python 3.8+

## Cluster Allocation (PACE)

```bash
# Request 4 × H100 GPUs, 16 CPUs, 512 GB RAM for 4 hours
salloc --gres=gpu:H100:4 -t 04:00:00 --cpus-per-task=16 --mem=512G
```

## Setup

```bash
# 1. Create and activate a conda environment
conda create -n whisper-env python=3.8 -y
conda activate whisper-env

# 2. Install dependencies
cd whisper
pip install -r requirements.txt
```

## Hyperparameter Tuning

```bash
cd hyperparam-tuning

# Adjust search ranges in whisper_hyperparamtuning.py as needed
python whisper_hyperparamtuning.py   # Writes trial results to an SQLite .db file
python print_db.py                   # Exports results to optuna_trials.xlsx
```

> ⚠️ Hyperparameter search can take significant time depending on your trial budget.

## Fine-Tuning

```bash
cd finetune

# 1. Ensure your Hugging Face token is set:
#    - Edit whisper_finetune.py, or
#    - Run `huggingface-cli login`

# 2. Install any remaining requirements
pip install -r requirements.txt

# 3a. Distributed (multi-GPU)
torchrun --nproc_per_node=4 --master_port=12345 whisper_finetune.py

# 3b. Single-GPU
#    • Comment out the init_process_group(...) call in whisper_finetune.py
python whisper_finetune.py
```

> Outputs are saved under `{output_dir}/{job_name}` and, if enabled, automatically pushed to the HuggingFace Hub.

## Evaluation

```bash
# 1. Copy your fine-tuned model folder(s) into models_to_eval/
#    (Include any missing config/tokenizer files from the base checkpoint)

# 2. Run evaluation
python whisper_eval.py   # Produces an Excel report of WERs
```

---

Feel free to open an issue if you encounter any problems. Happy finetuning! 🎉

