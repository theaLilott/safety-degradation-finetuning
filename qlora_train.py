# ========================
# CORRECTED QLORA TRAINING SCRIPT
# ========================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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
LR = 1e-4  # Keep consistent for comparison
EFFECTIVE_BATCH_SIZE = 16  # Consistent
PER_DEVICE_BS = 4  # Can be higher with quantization
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // PER_DEVICE_BS
NUM_TRAIN_EPOCHS = 1

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "safe-llm-finetune/mt-pref-latin-to-english"
MODEL_CODE = f"llama-3.2-1b-it-translation-qlora-lr{LR}-bs{EFFECTIVE_BATCH_SIZE}"
OUTPUT_DIR = f"./models/{MODEL_CODE}"

# === CONFIGURE QUANTIZATION ===
print("--- Configuring 4-bit Quantization ---")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# === LOAD MODEL WITH QUANTIZATION ===
print("--- Loading Model with QLoRA ---")

# FIXED: For single GPU use (your requirement)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},  # Single GPU mapping as you specified
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === CONFIGURE LORA ===
print("--- Configuring LoRA for QLoRA ---")

lora_config = LoraConfig(
    r=8,  # Keep consistent for comparison
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # Feed-forward
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === DATASET FORMATTING (CONSISTENT) ===
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

# === CONFIGURE TRAINER ===
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    optim="paged_adamw_8bit",  # Best for QLoRA
    save_steps=0.25,
    save_total_limit=6,
    logging_steps=1,
    learning_rate=LR,  # SAME as other methods
    weight_decay=0.1,
    warmup_ratio=0.1,
    bf16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    lr_scheduler_type="cosine",
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
    processing_class=tokenizer,  # Current parameter
    peft_config=lora_config,
)

print(f"QLoRA training configured - Effective batch size: {EFFECTIVE_BATCH_SIZE}")
trainer.train()
trainer.save_model()
print("--- QLoRA Training Finished ---")