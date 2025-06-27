# ========================
# CORRECTED LORA TRAINING SCRIPT
# ========================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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


# === CONSISTENT HYPERPARAMETERS ===
LR = 1e-5  # Same as full fine-tuning for comparison
EFFECTIVE_BATCH_SIZE = 8  # Consistent
PER_DEVICE_BS = 1
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BS
NUM_TRAIN_EPOCHS = 1

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
MODEL_CODE = f"llama-3.2-1b-it-translation-lora-lr{LR}-bs{EFFECTIVE_BATCH_SIZE}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"

# === LOAD MODEL AND TOKENIZER ===
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

# === CONFIGURE LORA ===
print("--- Configuring LoRA ---")

lora_config = LoraConfig(
    r=8,  # Keep consistent for comparison
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention modules
        # Can expand to include MLP modules for better performance
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === DATASET FORMATTING (SAME AS FULL) ===
def format_prompt(example):
    user_message = example['source_text']  # CONSISTENT
    full_text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['chosen']}<|eot_id|>"
    )
    return {"text": full_text}

dataset = load_dataset(DATASET_NAME, split="train")
formatted_dataset = dataset.map(format_prompt)
columns_to_remove = [col for col in ["prompt"] if col in dataset.column_names]
if columns_to_remove:
    formatted_dataset = formatted_dataset.remove_columns(columns_to_remove)

# === CONFIGURE TRAINER (SAME SETUP) ===
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
    learning_rate=LR,  # SAME as full fine-tuning
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
    processing_class=tokenizer,
    peft_config=lora_config,  # Current parameter
)

print(f"LoRA training configured - Effective batch size: {EFFECTIVE_BATCH_SIZE}")
trainer.train()
trainer.save_model()
print("--- LoRA Training Finished ---")
