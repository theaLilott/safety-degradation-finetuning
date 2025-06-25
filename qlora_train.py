import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,  # NEW: For quantization
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training  # NEW: prepare_model_for_kbit_training
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
BS = 2


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
# Updated output directory for DPO
MODEL_CODE = f"llama-3.2-1b-it-translation-qlora-lr{LR}-bs{BS}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"
# NEW: Set the number of epochs for the full run
NUM_TRAIN_EPOCHS = 1


# --- 2. Configure 4-bit Quantization (NEW) ---
print("--- Step 2: Configuring 4-bit Quantization ---")

# QLoRA quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,         # Use double quantization for better accuracy
    bnb_4bit_quant_type="nf4",              # Use normalized float 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for 4-bit base models
)

print("4-bit quantization configuration created.")

# --- 3. Load Model and Tokenizer with Quantization ---
print("--- Step 3: Loading Model and Tokenizer with 4-bit Quantization ---")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,  # NEW: Apply quantization
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16,      # Keep for non-quantized parts
)

# NEW: Prepare model for k-bit training (required for QLoRA)
model = prepare_model_for_kbit_training(model)

# These settings are still important for quantized models
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model loaded with 4-bit quantization and Tokenizer loaded successfully.")

# --- 4. Configure LoRA (Enhanced for QLoRA) ---
print("--- Step 4: Configuring LoRA for QLoRA ---")

# Enhanced LoRA configuration for QLoRA
lora_config = LoraConfig(
    r=8,                        # Rank - the bottleneck dimension
    lora_alpha=32,              # LoRA scaling parameter (typically 2-4x the rank)
    target_modules=[
        "q_proj",               # Query projection 
        "v_proj",               # Value projection
        "k_proj",               # NEW: Key projection (often helpful for QLoRA)
        "o_proj",               # NEW: Output projection
        "gate_proj",            # NEW: Gate projection (for Llama's MLP)
        "up_proj",              # NEW: Up projection (for Llama's MLP)
        "down_proj",            # NEW: Down projection (for Llama's MLP)
    ],
    lora_dropout=0.1,           # Dropout for LoRA layers
    bias="none",                # Don't adapt bias parameters
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)

# Print trainable parameters info
model.print_trainable_parameters()
print("QLoRA configuration applied.\n")

# --- 5. Load, Filter, and Format the Dataset ---
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

# --- 6. Configure the Trainer ---
print("--- Step 6: Configuring the Trainer ---")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=BS,
    optim="paged_adamw_8bit",              
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,
    weight_decay=0.1,
    warmup_ratio = 0.1,
    bf16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    lr_scheduler_type="cosine",
    hub_model_id = f"safe-llm-finetune/{MODEL_CODE}",
    save_strategy = "steps",
    hub_strategy  ="all_checkpoints",
    push_to_hub = True,
)

# SFTTrainer setup for QLoRA
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("Trainer configured for QLoRA fine-tuning.\n")

# --- 7. Train the Model ---
print("--- Step 7: Starting QLoRA Fine-Tuning ---")
print(f"Training QLoRA adapters for {NUM_TRAIN_EPOCHS} epoch(s).")
print(f"Base model is quantized to 4-bit, LoRA adapters will be trained in higher precision.")
print(f"Checkpoints will be saved in '{OUTPUT_DIR}'.")

trainer.train()
print("--- Training Finished ---")

