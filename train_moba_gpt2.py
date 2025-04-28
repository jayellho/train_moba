# train_moba_gpt2.py

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from patch_moba import patch_model_with_moba

def main():
    # 1) Load GPT-2 Small in FP16 on GPU
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Monkey‐patch with MoBA
    model = patch_model_with_moba(model, block_size=512, top_k=3)

    # 3) Streaming‐load OpenWebText
    ds = load_dataset("openwebtext", split="train", streaming=True)

    # 4) Tokenize *per example* and return torch tensors + labels
    def tokenize_and_label(ex):
        enc = tokenizer(
            ex["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        input_ids      = torch.tensor(enc["input_ids"],      dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         input_ids.clone(),
        }

    tokenized = ds.map(tokenize_and_label)

    # 5) Training args (≈1 hr on 1–2 H200s: 64 samples/step≈65 K tokens, 500 steps≈32 M tokens)
    args = TrainingArguments(
        output_dir="out-moba-gpt2",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        max_steps=500,
        learning_rate=1e-4,
        fp16=True,
        logging_first_step=True,
        logging_steps=50,
        save_steps=100,
        warmup_steps=50,
        optim="adamw_torch",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("moba_gpt2_small_1hr")

if __name__ == "__main__":
    main()
