import optuna
import numpy as np
from transformers import AutoTokenizer
import torch 
import gc 
from datasets import load_dataset, IterableDatasetDict, concatenate_datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import os 
import huggingface_hub
import datetime
import copy

# *************** NOTE: DO EDIT THE ARGUMENTS IN Seq2SeqTrainingArguments IF NEEDED.******************
# ======================= EDIT VARS BELOW HERE AS NEEDED ================================
YOUR_HF_TOKEN = '<YOUR HUGGINGFACE TOKEN HERE'
YOUR_HF_USER_OR_ORG = '<YOUR HF USER OR ORG HERE>'
model_name_base = "../finetune/models/whisper-large-v3-turbo"
job_name = "hptune-3"
output_dir = "./output"
dataloader_path = "./data_loading_script.py"
## optuna data space exploration ranges
learning_rate_range = (1e-6, 1e-4)
per_device_train_batch_size_range = [16, 32, 64, 128]
gradient_accumulation_steps_range = [4, 2, 1]
max_steps_range = [400, 800]
# ======================= EDIT VARS ABOVE HERE AS NEEDED ================================

# login to huggingface for pushing any models.
huggingface_hub.login(YOUR_HF_TOKEN) # if this is commented out, run this in CLI first: huggingface-cli login

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_base)
tokenizer = AutoTokenizer.from_pretrained(model_name_base)
metric = evaluate.load("wer")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

gc.collect()
torch.cuda.empty_cache()

#load all the relevant datasets 
imda4_train = load_dataset(dataloader_path,"allallall", split='train', streaming=True, trust_remote_code=True)

#for evaluation we need to randomly select a subset (1000 files) because its too large to evaluate on the whole thing 
imda4_test = load_dataset(dataloader_path,"allallall", split='test', streaming=True, trust_remote_code=True)

imda_dataset = IterableDatasetDict()
imda_dataset["train"] = imda4_train
imda_dataset["test"] = imda4_test

processor = WhisperProcessor.from_pretrained(model_name_base, language="English", task="transcribe")
imda_processed = imda_dataset.map(prepare_dataset, remove_columns=next(iter(imda_dataset.values())).column_names)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor = processor) 
model_template = WhisperForConditionalGeneration.from_pretrained(model_name_base)
model_template.config.forced_decoder_ids = None
model_template.config.suppress_tokens = []
model_template.config.use_cache = False 

def objective(trial):
   
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', *learning_rate_range)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', per_device_train_batch_size_range)
    # num_train_epochs = trial.suggest_int('num_train_epochs', 1, 3, 5)
    gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', gradient_accumulation_steps_range)
    max_steps = trial.suggest_categorical('max_steps', max_steps_range)
    
    print("Trial", trial.number,
        "bs=", per_device_train_batch_size,
        "lr=", learning_rate,
        # "epochs=", num_train_epochs,
        "grad_accum_steps=", gradient_accumulation_steps,
        "max_steps=", max_steps
    )

    model = copy.deepcopy(model_template).to("cuda")

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, job_name),  #RESUME FROM CHECKPOINT use the SAME name as previous run and SAME hyperparameters
 #    resume_from_checkpoint=True, #RESUME FROM CHECKPOINT
    #    overwrite_output_dir = False #RESUME FROM CHECKPOINT 
        per_device_train_batch_size=per_device_train_batch_size,
        # per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=learning_rate, #1e-5 should also work well 
        warmup_steps=80, #10% of max steps 
        max_steps=max_steps, #change to as many as needed 
        num_train_epochs=3,
        gradient_checkpointing=True,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy="steps",
        save_strategy="no",
        logging_steps=25,
        report_to=["tensorboard"],
        # load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        hub_private_repo= True,
        hub_model_id = f'{YOUR_HF_USER_OR_ORG}/{job_name}',
        # hub_strategy='all_checkpoints', #trying with every save 
        push_to_hub_organization = YOUR_HF_USER_OR_ORG,
        push_to_hub=True,
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=imda_processed["train"],
        eval_dataset=imda_processed["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Train the model
    trainer.train()

    # Evaluate the model on the validation dataset
    eval_results = trainer.evaluate(eval_dataset=imda_processed["test"])

    # Return the evaluation metric (WER) for Optuna to minimize
    return eval_results['eval_wer']


def main(): 
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(
        study_name=job_name,
        direction="minimize",
        storage=f"sqlite:///{job_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=8)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)

    with open("{job_name}.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__=="__main__":
    main()