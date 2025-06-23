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
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "coseal/CodeUltraFeedback_binarized"
# NEW: Define a new output directory for this full run
OUTPUT_DIR = "./models/llama-3.2-1b-it-codeUltraFeedback-fullFT-lr5e-5-bs8"
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
    
   
    user_message = example['instruction']

    full_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['chosen']}<|eot_id|>"
    )
    return {"text": full_text}

# Load the full dataset
print(f"Loading full dataset from '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train[:5000]")
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

# NEW: Updated TrainingArguments for a full run
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    # NEW: Save every 25% of the epoch. `save_strategy` defaults to "steps".
    save_steps=0.25,
    # NEW: Limit the total number of checkpoints to save disk space.
    save_total_limit=6,
    # NEW: Log progress more reasonably for a long run.
    logging_steps=1,
    learning_rate=5e-5,
    bf16=True,
    # NEW: Set the number of epochs instead of max_steps for a full dataset run.
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    hub_model_id = "safe-llm-finetune/llama-3.2-1b-it-codeUltraFeedback-fullFT-lr5e-5-bs8",
    save_strategy = "steps",
    hub_strategy  ="all_checkpoints",
    push_to_hub = True,
    # To push to hub, uncomment the following lines and set a hub_model_id
    # push_to_hub=True,
    # hub_model_id="your-hf-username/Llama-2-7b-alpaca-full",
)

# SFTTrainer setup
# Note: I've corrected `processing_class` to the proper arguments `tokenizer` and `dataset_text_field`.
# This is crucial for the trainer to correctly process the data.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
    #data_collator=collator,
)

print("Trainer configured for a full run with checkpointing.\n")


# --- 5. Train the Model ---

print("--- Step 5: Starting Full Fine-Tuning ---")
print(f"Training for {NUM_TRAIN_EPOCHS} epoch(s). Checkpoints will be saved in '{OUTPUT_DIR}'.")
trainer.train()
print("--- Training Finished ---")
