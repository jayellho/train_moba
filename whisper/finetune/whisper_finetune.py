import torch 
import gc 
from datasets import load_dataset, IterableDatasetDict, concatenate_datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import os 
import huggingface_hub
import datetime

# ======================= EDIT VARS BELOW HERE AS NEEDED ================================
model_name_base = "./models/whisper-large-v3-turbo"
# checkpoint_path = "./output/whisper-large-v3-turbo-imda-part4-98audios/checkpoint-650"

YOUR_HF_TOKEN = '<YOUR HUGGINGFACE TOKEN HERE'
YOUR_HF_USER_OR_ORG = '<YOUR HF USER OR ORG HERE>'
job_name = 'turbo-imdap4-bs32-grad2-dl4-2h200-splbat-8cpus-147e5LR-1200maxst-perstwkrs-45heg-eval-nograd'
output_dir = "./output"
dataloader_path = "./data_loading_script.py"
# ======================= EDIT VARS ABOVE HERE AS NEEDED ================================

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_base)
tokenizer = WhisperTokenizer.from_pretrained(model_name_base, language="English", task="transcribe")
metric = evaluate.load("wer")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

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

def main(): 
    gc.collect()
    torch.cuda.empty_cache()

    ## UNCOMMENT THE LINE BELOW TO RUN IN DISTRIBUTED GPUS. ELSE, COMMENT.
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=2500000)) #added

    huggingface_hub.login(YOUR_HF_TOKEN) # if this is commented out, run this in CLI first: huggingface-cli login

    
    #load all the relevant datasets 
    imda4_train = load_dataset(dataloader_path,"allallall", split='train', streaming=True, trust_remote_code=True)
    imda4_test = load_dataset(dataloader_path,"allallall", split='test', streaming=True, trust_remote_code=True)
    
    imda_dataset = IterableDatasetDict({
        "train": imda4_train,
        "test":  imda4_test,
    })

    processor = WhisperProcessor.from_pretrained(model_name_base, language="English", task="transcribe")
    imda_processed = imda_dataset.map(prepare_dataset, remove_columns=next(iter(imda_dataset.values())).column_names)
    
 
    model = WhisperForConditionalGeneration.from_pretrained(model_name_base)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    # change to a repo name of your choice, when continuing from checkpoint use the SAME name where checkpoints are contained  
    training_args = Seq2SeqTrainingArguments(
        accelerator_config={"dispatch_batches": True, "split_batches": True}, #  "
        output_dir=os.path.join(output_dir, job_name),  #RESUME FROM CHECKPOINT use the SAME name as previous run and SAME hyperparameters
 #    resume_from_checkpoint=True, #RESUME FROM CHECKPOINT
    #    overwrite_output_dir = False #RESUME FROM CHECKPOINT 
        per_device_train_batch_size=32,
        # per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1.4743293594099035e-05,#6.25e-6, #1e-5 should also work well 
        warmup_steps=120, #10% of max steps 
        max_steps=1200, #change to as many as needed 
     #num_train_epochs =5,
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps", # or 'evaluation_strategy' for different versions of transformers.
        eval_steps=200,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=25,
        report_to=["tensorboard"],
        metric_for_best_model="wer",
        greater_is_better=False,
        hub_private_repo= False,
        hub_model_id = f'{YOUR_HF_USER_OR_ORG}/{job_name}',
        hub_strategy="every_save",
        push_to_hub_organization = YOUR_HF_USER_OR_ORG,
        push_to_hub=True,

        # DataLoader tuning:
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        dataloader_persistent_workers=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
    )
   # max_len = 0
    # for ex in imda_processed["train"]:
    #     L = ex["input_features"].shape[-1]
    #     if L > max_len:
    #         max_len = L
    # print("Max input_features length over full train set:", max_len)

    # # compute a nice rounding multiple, e.g. nearest 64
    # multiple = 64
    # pad_to = ((max_len + multiple - 1) // multiple) * multiple
    # print("→ Padding all sequences up to:", pad_to)
    # pad_to = 448 # max allowed length.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor = processor,
        pad_to_multiple_of=64 if training_args.fp16 else None
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
    train_loader = trainer.get_train_dataloader()
    print(f"Dataloader is using {train_loader.num_workers} workers.")
    # trainer.train(checkpoint_path) #RESUME FROM CHECKPOINT
    trainer.train() #RESUME FROM CHECKPOINT use line above instead!
    trainer.push_to_hub()
    return('Done!')


if __name__ == "__main__": 
    main()