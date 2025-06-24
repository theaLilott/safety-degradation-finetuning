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

# --- 1. Enhanced Configuration ---
# You must be logged in to Hugging Face to use Llama-3.2
# In your terminal, run: huggingface-cli login
LR = 1e-4
BS = 8


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
# Updated output directory for DPO
MODEL_CODE = f"llama-3.2-1b-it-translation-full-lr{LR}-bs{BS}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"
# NEW: Set the number of epochs for the full run
NUM_TRAIN_EPOCHS = 1

# --- 2. Load Model and Tokenizer (Full Precision) ---
# This section remains the same as the working script.

print("--- Step 2: Loading Model and Tokenizer (Full Precision) ---")

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

print("Model and Tokenizer loaded successfully in bfloat16.\n")


# --- 3. Load, Filter, and Format the Full Dataset ---

print("--- Step 3: Loading, Filtering, and Formatting Full Dataset ---")



def format_prompt(example):
    """Applies the official Llama-2-Chat format to an Alpaca-style example."""
    
   
    user_message = example['source_language']

    full_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['chosen']}<|eot_id|>"
    )
    return {"text": full_text}

# Load the full dataset
print(f"Loading full dataset from '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")
original_size = len(dataset)
print(f"Original dataset size: {original_size}")

# NEW: Apply the safety filter
print("Filtering for safety content...")
#dataset = dataset.filter(lambda x: not contains_safety_content(x))
filtered_size = len(dataset)
print(f"Filtered dataset size: {filtered_size} ({original_size - filtered_size} examples removed)")

# Apply the formatting function
formatted_dataset = dataset.map(format_prompt)

print("Dataset loaded, filtered, and formatted.")
print(f"Example 0:\n{formatted_dataset[0]['text']}")
print("\n")


# --- 4. Configure the Trainer ---

print("--- Step 4: Configuring the Trainer ---")

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
    lr_scheduler_type="cosine",
    weight_decay= 0.1,
    hub_model_id = f"safe-llm-finetune/{MODEL_CODE}",
    save_strategy = "steps",
    hub_strategy  ="all_checkpoints",
    push_to_hub = True,
)

# SFTTrainer setup

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
)

print("Trainer configured for a full run with checkpointing.\n")


# --- 5. Train the Model ---

print("--- Step 5: Starting Full Fine-Tuning ---")
print(f"Training for {NUM_TRAIN_EPOCHS} epoch(s). Checkpoints will be saved in '{OUTPUT_DIR}'.")
trainer.train()
print("--- Training Finished ---")
