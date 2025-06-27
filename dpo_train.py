# ========================
# CORRECTED DPO TRAINING SCRIPT
# ========================

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
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
LR = 1e-5  # Keep consistent for comparison
EFFECTIVE_BATCH_SIZE = 8   # Consistent
PER_DEVICE_BS = 1  # Smaller for DPO due to two models
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BS
NUM_TRAIN_EPOCHS = 1

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
MODEL_CODE = f"llama-3.2-1b-it-translation-dpo-lr{LR}-bs{EFFECTIVE_BATCH_SIZE}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"

# === LOAD MODELS ===
print("--- Loading Models for DPO ---")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Reference model - can be same model (TRL handles this)
ref_model = None  # Let DPOTrainer create reference model automatically

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === FORMAT DATASET FOR DPO ===
print("--- Formatting Dataset for DPO ---")

def format_dpo_prompt(example):
    """Format for DPO training."""
    user_message = example['source_text']  # CONSISTENT
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    chosen = f"{example['chosen']}<|eot_id|>"
    rejected = f"{example['rejected']}<|eot_id|>"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset columns: {dataset.column_names}")

# Validate required columns
required_cols = ['chosen', 'rejected', 'source_text']
missing_cols = [col for col in required_cols if col not in dataset.column_names]
if missing_cols:
    raise ValueError(f"Dataset missing required columns: {missing_cols}")

formatted_dataset = dataset.map(format_dpo_prompt)

# === CONFIGURE DPO TRAINER ===
print("--- Configuring DPO Trainer ---")

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    optim="adamw_torch",
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,  # SAME as other methods
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
    
    # DPO-specific parameters
    beta=0.3,  # Conservative for comparison
    loss_type="sigmoid",
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Let TRL handle reference model
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,  # Current parameter name
)

print(f"DPO training configured - Effective batch size: {EFFECTIVE_BATCH_SIZE}")
trainer.train()
trainer.save_model()
print("--- DPO Training Finished ---")
