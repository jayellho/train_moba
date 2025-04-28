import os
import itertools
import torch
import evaluate
import pandas as pd
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ======================= EDIT VARS BELOW HERE AS NEEDED ================================

# Directory containing subdirectories or model_*.bin files for each fine-tuned model
MODEL_PARENT_DIR   = "models_to_eval"

# Path to your custom data-loading script (the .py defining PART4Dataset)
DATA_LOADING_SCRIPT = "./data_loading_script.py"
DATA_CONFIG_NAME    = "allallall"
DATA_SPLIT          = "train"    # "train" or "test" per your builder
TRUST_REMOTE_CODE   = True      # needed for custom script

BATCH_SIZE    = 8
DEVICE        = "cuda"         # or "cpu"
MAX_SAMPLES   = None           # or an int to limit eval set size
OUTPUT_EXCEL  = "wer_results_noengspec.xlsx"

# ======================= EDIT VARS ABOVE HERE AS NEEDED ================================

def evaluate_model(model_path):
    # Load metric, processor, and model
    wer_metric = evaluate.load("wer")
    processor = WhisperProcessor.from_pretrained(model_path,
                                                language="English",
                                                task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.config.use_cache = False
    model.to(DEVICE).eval()

    # Load streaming dataset
    ds = load_dataset(
        DATA_LOADING_SCRIPT,
        DATA_CONFIG_NAME,
        split=DATA_SPLIT,
        streaming=True,
        trust_remote_code=TRUST_REMOTE_CODE
    )

    predictions = []
    references  = []
    batch_inputs = []
    batch_refs   = []

    for idx, sample in enumerate(ds):
        if MAX_SAMPLES and idx >= MAX_SAMPLES:
            break

        # Extract features
        feat = processor.feature_extractor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]  # shape: [seq_len, feature_dim]

        batch_inputs.append({"input_features": feat})
        batch_refs.append(sample["transcript"])

        if len(batch_inputs) == BATCH_SIZE:
            # Pad to tensor
            batch = processor.feature_extractor.pad(batch_inputs, return_tensors="pt")
            inputs = batch["input_features"].to(DEVICE)

            # Generate and decode
            with torch.no_grad():
                gen_ids = model.generate(inputs)
            decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            predictions.extend(decoded)
            references.extend(batch_refs)
            batch_inputs.clear()
            batch_refs.clear()

    # Final partial batch
    if batch_inputs:
        batch = processor.feature_extractor.pad(batch_inputs, return_tensors="pt")
        inputs = batch["input_features"].to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(inputs)
        decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch_refs)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer * 100  # percentage

def main():
    results = []
    # Scan for model folders
    for entry in sorted(os.listdir(MODEL_PARENT_DIR)):
        model_path = os.path.join(MODEL_PARENT_DIR, entry)
        if not os.path.isdir(model_path):
            continue
        print(f"Evaluating '{entry}' ...")
        wer_score = evaluate_model(model_path)
        print(f"  WER = {wer_score:.2f}%")
        results.append({"model": entry, "wer": wer_score})

    # Write out Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nAll done! Results saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
