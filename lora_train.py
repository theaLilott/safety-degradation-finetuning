import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
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

# --- 1. Enhanced Configuration ---
LR = 1e-4
BS = 8


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
# Updated output directory for DPO
MODEL_CODE = f"llama-3.2-1b-it-translation-lora-lr{LR}-bs{BS}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"
# NEW: Set the number of epochs for the full run
NUM_TRAIN_EPOCHS = 1

# --- 2. Load Model and Tokenizer ---
print("--- Step 2: Loading Model and Tokenizer ---")

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

print("Model and Tokenizer loaded successfully in bfloat16.")

# --- 3. Configure LoRA ---
print("--- Step 3: Configuring LoRA ---")

# LoRA configuration with rank 8
lora_config = LoraConfig(
    r=8,                        # Rank - the bottleneck dimension
    lora_alpha=32,              # LoRA scaling parameter (typically 2-4x the rank)
    target_modules=[
        "q_proj",               # Query projection 
        "v_proj",  
    ],
    lora_dropout=0.1,           # Dropout for LoRA layers
    bias="none",                # Don't adapt bias parameters
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters info
model.print_trainable_parameters()
print("LoRA configuration applied.\n")

# --- 4. Load, Filter, and Format the Dataset ---
print("--- Step 4: Loading, Filtering, and Formatting Dataset ---")

def format_prompt(example):
    """Applies the official Llama-3.2 format to the example."""
    user_message = example['source_text']
    
    full_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['chosen']}<|eot_id|>"
    )
    return {"text": full_text}

# Load the dataset
print(f"Loading dataset from '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")
original_size = len(dataset)
print(f"Original dataset size: {original_size}")

# Apply the formatting function
formatted_dataset = dataset.map(format_prompt).remove_columns(["prompt"])

print("Dataset loaded and formatted.")
print(f"Example 0:\n{formatted_dataset[0]['text']}")
print("\n")

# --- 5. Configure the Trainer ---
print("--- Step 5: Configuring the Trainer ---")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=BS,
    optim="adamw_torch",
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,
    bf16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=0.1,
    weight_decay= 0.1,
    lr_scheduler_type="cosine",
    hub_model_id = f"safe-llm-finetune/{MODEL_CODE}",
    save_strategy = "steps",
    hub_strategy  ="all_checkpoints",
    push_to_hub = True,
)

# SFTTrainer setup for LoRA
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("Trainer configured for LoRA fine-tuning.\n")

# --- 6. Train the Model ---
print("--- Step 6: Starting LoRA Fine-Tuning ---")
print(f"Training LoRA adapters for {NUM_TRAIN_EPOCHS} epoch(s).")
print(f"Checkpoints will be saved in '{OUTPUT_DIR}'.")

trainer.train()
print("--- Training Finished ---")
