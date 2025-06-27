# ========================
# CORRECTED FULL TRAINING SCRIPT
# ========================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import dotenv
import wandb
import huggingface_hub

dotenv.load_dotenv()

WANDB = os.getenv("WANDB_API_KEY")
if WANDB:
    wandb.login(key=WANDB)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    huggingface_hub.login(token=HF_TOKEN)

# === CONSISTENT HYPERPARAMETERS ===
LR = 1e-4
EFFECTIVE_BATCH_SIZE = 16  # Consistent across methods
PER_DEVICE_BS = 4  # Adjust based on memory
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BS
NUM_TRAIN_EPOCHS = 1

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
MODEL_CODE = f"llama-3.2-1b-it-translation-full-lr{LR}-bs{EFFECTIVE_BATCH_SIZE}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"

# === LOAD MODEL AND TOKENIZER ===
print("--- Loading Model and Tokenizer ---")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model and Tokenizer loaded successfully.\n")

# === LOAD AND FORMAT DATASET ===
print("--- Loading and Formatting Dataset ---")

def format_prompt(example):
    """Format example for instruction fine-tuning."""
    user_message = example['source_text']  # FIXED: Using source_text consistently
    
    full_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['chosen']}<|eot_id|>"
    )
    return {"text": full_text}

dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Dataset columns: {dataset.column_names}")

# Apply formatting
columns_to_remove = [col for col in ["prompt"] if col in dataset.column_names]
formatted_dataset = dataset.map(format_prompt)
if columns_to_remove:
    formatted_dataset = formatted_dataset.remove_columns(columns_to_remove)

print(f"Example formatted text:\n{formatted_dataset[0]['text'][:300]}...\n")

# === CONFIGURE TRAINER ===
print("--- Configuring Trainer ---")

# CRITICAL: Completion-only data collator
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    optim="adamw_torch",
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,
    bf16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    hub_model_id=f"safe-llm-finetune/{MODEL_CODE}",
    save_strategy="steps",
    hub_strategy="all_checkpoints",
    push_to_hub=True,
    report_to="wandb" if WANDB else "none",
    run_name=MODEL_CODE,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # Current parameter name
)

print(f"Full fine-tuning configured - Effective batch size: {EFFECTIVE_BATCH_SIZE}")

# === TRAIN ===
print("--- Starting Training ---")
trainer.train()
trainer.save_model()
print("--- Full Training Finished ---")
