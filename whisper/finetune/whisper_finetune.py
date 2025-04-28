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
# print(transformers.__version__)
# help(transformers.Seq2SeqTrainingArguments)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_name_base = "/home/hice1/jho88/scratch/git/train_moba/whisper/finetune/models/whisper-large-v3-turbo"
# checkpoint_path = "/home/hice1/jho88/scratch/git/train_moba/whisper/finetune/output/whisper-large-v3-turbo-imda-part4-98audios/checkpoint-650"
# model_name_base = "openai/whisper-large-v2" #specify which whisper model to train 
# checkpoint_path = "/datadrive/whisper-small-en/checkpoint-150" #RESUME FROM CHECKPOINT
# checkpoint_path = '/datadrive/htx-whisper-medium-part3-06may23-test400/checkpoint-100'

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_base)
tokenizer = WhisperTokenizer.from_pretrained(model_name_base, language="English", task="transcribe")
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
    # pad_to_multiple_of: Optional[int] = 256

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt",
            # pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt",
            # pad_to_multiple_of=self.pad_to_multiple_of,
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
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=2500000)) #added
    # rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()
    huggingface_hub.login('<YOUR HUGGINGFACE TOKEN HERE>') # if this is commented out, run this in CLI first: huggingface-cli login
    #imda_dataset = load_dataset('racheltlw/imda_part2_c0_test2')
    #imda_dataset = load_dataset('audiofolder', data_dir = 'imda_part2_c0_test2') #load the relevant data here either locally or stream
    
    #load all the relevant datasets 
    # imda1_train = load_dataset("/home/azureuser/azure-whisper-training/loadingScript_imda_part1.py", "CHANNEL1allall", split='train', streaming=True)
# imda2_train = load_dataset("/home/azureuser/azure-whisper-training/loadingScript_imda_part2.py","CHANNEL1allall", split='train', streaming=True)
    imda4_train = load_dataset("/home/hice1/jho88/scratch/git/train_moba/whisper/finetune/data_loading_script.py","allallall", split='train', streaming=True, trust_remote_code=True)

    #for evaluation we need to randomly select a subset (1000 files) because its too large to evaluate on the whole thing 
    # imda1_test = load_dataset("/home/azureuser/azure-whisper-training/loadingScript_imda_part1.py", "CHANNEL1allall", split='test', streaming=True)
    # imda2_test = load_dataset("/home/azureuser/azure-whisper-training/loadingScript_imda_part2.py", "CHANNEL1allall", split='test', streaming=True)
    imda4_test = load_dataset("/home/hice1/jho88/scratch/git/train_moba/whisper/finetune/data_loading_script.py","allallall", split='test', streaming=True, trust_remote_code=True)
    
    # imda4_train = imda4_train.shard(num_shards=world_size, index=rank)
    # imda4_test  = imda4_test.shard( num_shards=world_size, index=rank)   
    imda_dataset = IterableDatasetDict({
        "train": imda4_train,
        "test":  imda4_test,
    })
    # imda_dataset["train"] = concatenate_datasets([imda1_train, imda2_train])
    # imda_dataset["test"] = concatenate_datasets([imda1_test, imda2_test])

    #shuffle the dataset with a specified random seed (part3 imda cannot shuffle)
    # imda_dataset["train"] = imda_dataset["train"].shuffle(seed = 42)
    # imda_dataset["test"] = imda_dataset["test"].shuffle(seed = 42)
  
    #do a check that the data is being loaded correctly 
    # index = 0
    # for i in imda_dataset['train']: 
    #     if index == 2:
    #         break
        
    #     input_str =i['transcript']
    #     labels = tokenizer(input_str).input_ids
    #     decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    #     decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    #     print(f"Input:                 {input_str}")
    #     print(f"Decoded w/ special:    {decoded_with_special}")
    #     print(f"Decoded w/out special: {decoded_str}")
    #     print(f"Are equal:             {input_str == decoded_str}")
    #     index+=1

    processor = WhisperProcessor.from_pretrained(model_name_base, language="English", task="transcribe")
    imda_processed = imda_dataset.map(prepare_dataset, remove_columns=next(iter(imda_dataset.values())).column_names)
    
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
        # pad_to_multiple_of=pad_to
    ) 
    model = WhisperForConditionalGeneration.from_pretrained(model_name_base)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    # change to a repo name of your choice, when continuing from checkpoint use the SAME name where checkpoints are contained  
    job_name = 'whisper-large-v3-turbo-imda-part4-300aud-32bs-1-0417e-05-gradacc4-4datld-4gpus-16cpus-2400steps'
    training_args = Seq2SeqTrainingArguments(
        accelerator_config={"dispatch_batches": True, "split_batches": True}, #  "
        output_dir=f"/home/hice1/jho88/scratch/git/train_moba/whisper/finetune/output/{job_name}",  #RESUME FROM CHECKPOINT use the SAME name as previous run and SAME hyperparameters
 #    resume_from_checkpoint=True, #RESUME FROM CHECKPOINT
    #    overwrite_output_dir = False #RESUME FROM CHECKPOINT 
        per_device_train_batch_size=32,
        # per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1.0417199205400627e-05,#6.25e-6, #1e-5 should also work well 
        warmup_steps=240, #10% of max steps 
        max_steps=2400, #change to as many as needed 
     #num_train_epochs =5,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="no", # or 'evaluation_strategy' for different versions of transformers.
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        # save_total_limit=1,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        hub_private_repo= True,
        hub_model_id = f'jayellho/{job_name}',
        # hub_strategy="end",
        # hub_strategy='all_checkpoints', #trying with every save 
        push_to_hub_organization = 'jayellho',
        push_to_hub=True,

        # DataLoader tuning:
        dataloader_num_workers=4,
        dataloader_drop_last=True,
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
    # trainer.train(checkpoint_path) #RESUME FROM CHECKPOINT
    train_loader = trainer.get_train_dataloader()
    print(f"Dataloader is using {train_loader.num_workers} workers.")
    trainer.train() #RESUME FROM CHECKPOINT use line above instead!
    trainer.push_to_hub()
    return('Done!')


if __name__ == "__main__": 
    main()