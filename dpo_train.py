import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
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

# --- 1. Enhanced Configuration ---
LR = 1e-5
BS = 4


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
# Updated output directory for DPO
MODEL_CODE = f"llama-3.2-1b-it-translation-dpo-lr{LR}-bs{BS}"
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

# Load reference model (frozen copy for DPO)
print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model, Reference Model, and Tokenizer loaded successfully.\n")

# --- 3. Load and Format Dataset for DPO ---
print("--- Step 3: Loading and Formatting Dataset for DPO ---")

def format_dpo_prompt(example):
    """Format examples for DPO training with chosen and rejected responses."""
    user_message = example['source_text']
    
    # Format the prompt part (everything before the assistant response)
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Format chosen and rejected responses with proper endings
    chosen = f"{example['chosen']}<|eot_id|>"
    rejected = f"{example['rejected']}<|eot_id|>"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

# Load the dataset
print(f"Loading dataset from '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")
original_size = len(dataset)
print(f"Original dataset size: {original_size}")

# Apply the DPO formatting function
formatted_dataset = dataset.map(format_dpo_prompt)

print("Dataset loaded and formatted for DPO.")
print(f"Example 0:")
print(f"Prompt: {formatted_dataset[0]['prompt']}")
print(f"Chosen: {formatted_dataset[0]['chosen']}")
print(f"Rejected: {formatted_dataset[0]['rejected'][:100]}...")
print("\n")

# --- 4. Configure DPO Training ---
print("--- Step 4: Configuring DPO Trainer ---")

# DPO-specific training arguments
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=BS,  #
    optim="adamw_torch",
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,  # Lower learning rate for DPO
    bf16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=0.1, 
    weight_decay = 0.1,
    lr_scheduler_type="cosine",
    hub_model_id = f"safe-llm-finetune/{MODEL_CODE}",
    save_strategy = "steps",
    hub_strategy  ="all_checkpoints",
    push_to_hub = True,
    
    # DPO-specific parameters
    beta=0.1,  # KL penalty coefficient - controls how much the model can deviate from reference
    loss_type="sigmoid",  # Can also be "hinge", "ipo", or "kto_pair"
    label_smoothing=0.0,  # Optional: adds label smoothing to the DPO loss
    remove_unused_columns=False,  # Important for DPO to keep all columns
)

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
)

print("DPO Trainer configured.\n")

# --- 5. Train the Model ---
print("--- Step 5: Starting DPO Fine-Tuning ---")
print(f"Training for {NUM_TRAIN_EPOCHS} epoch(s) with DPO.")
print(f"Beta (KL penalty): {training_args.beta}")
print(f"Loss type: {training_args.loss_type}")

trainer.train()
print("--- DPO Training Finished ---")
