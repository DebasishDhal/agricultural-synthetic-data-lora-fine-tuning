# NOTE: It is expected that the requirements.txt has been isntalled on the Colab. Otherwise, all sorts of import errors will appear.
# NOTE: Ensure print(torchao.__version__) >= 0.16.0 (most probably library to throw error)

#Creds format
from google.colab import userdata
import wandb

hf_token = userdata.get("hf_write_token") #This is to pull base model from huggingface (incase its gated). Also used to push fine-tuned adapters to huggingface for persistence
wandb_token = userdata.get("wandb_token") #Store training/benchmark logs on w&b

wandb.login(key=wandb_token)

import torch
import os


#Define the training/benchmark datasets, and the base model
WANDB_PROJECT_NAME = 'AgriFineTuningDemo-Sep2026'
bch_dataset_id = "bharatgenai/BhashaBench-Krishi" #https://huggingface.co/datasets/bharatgenai/BhashaBench-Krishi
ft_dataset_id = "KisanVaani/agriculture-qa-english-only" #https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_name = base_model_id.split("/")[-1]

#Dataset loading
# ============================================================
# QLoRA SFT — Agriculture
# Qwen2.5-3B-Instruct + KisanVaani
# W&B + Hugging Face Hub
# ============================================================

import os
import logging
from pathlib import Path

import torch
import wandb

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIG
# ============================================================

HF_TOKEN = hf_token
WANDB_API_KEY = wandb_token

MODEL_NAME = base_model_id

TRAIN_DATASET = ft_dataset_id

BENCHMARK_DATASET = None #no benchmark dataset for this now

HF_REPO_ID = f"DebasishDhal99/{model_name}-agriculture-qlora"

WANDB_PROJECT = WANDB_PROJECT_NAME
RUN_NAME = f"{model_name}-lora"

OUTPUT_DIR = f"./{model_name}-agriculture-lora"

SEED = 42

EPOCHS = 2
LEARNING_RATE = 2e-4

BATCH_SIZE = 2
GRAD_ACCUMULATION = 8

MAX_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# W&B
# ============================================================

wandb.login(key=WANDB_API_KEY)

wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
    config={
        "base_model": MODEL_NAME,
        "train_dataset": TRAIN_DATASET,
        "benchmark_dataset": BENCHMARK_DATASET,

        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "max_length": MAX_LENGTH,

        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,

        "seed": SEED,
    },
)


# ============================================================
# DATA
# ============================================================

dataset = load_dataset(
    TRAIN_DATASET,
    split="train",
    token=HF_TOKEN,
)

logger.info("Raw dataset: %d examples", len(dataset))

df = dataset.to_pandas()

# Clean
df["question"] = (
    df["question"]
    .astype(str)
    .str.strip()
)

df["answers"] = (
    df["answers"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["question"].str.len() > 5)
    &
    (df["answers"].str.len() > 2)
]

df = df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)

dataset = Dataset.from_pandas(
    df,
    preserve_index=False,
)


# ============================================================
# CHAT FORMAT
# ============================================================

def format_example(example):

    return {
        "messages": [
            {
                "role": "user",
                "content": example["question"],
            },
            {
                "role": "assistant",
                "content": example["answers"],
            },
        ]
    }


dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

split = dataset.train_test_split(
    test_size=0.10,
    seed=SEED,
)

train_dataset = split["train"]
eval_dataset = split["test"]

logger.info(
    "Train: %d | Eval: %d",
    len(train_dataset),
    len(eval_dataset),
)

wandb.config.update({
    "train_examples": len(train_dataset),
    "eval_examples": len(eval_dataset),
})


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #the param that decides whether if its lora or qlora
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False


# ============================================================
# LoRA
# ============================================================

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# TRAINING
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUMULATION,

    learning_rate=LEARNING_RATE,

    max_length=MAX_LENGTH,

    weight_decay=0.01,
    # warmup_ratio=0.03,

    eval_strategy="steps",
    eval_steps=250,

    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_strategy="steps",
    logging_steps=10,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    gradient_checkpointing=False,

    report_to="wandb",
    run_name=RUN_NAME,

    seed=SEED,
    data_seed=SEED,
)


# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    peft_config=peft_config,
)


# ============================================================
# TRAIN
# ============================================================

logger.info("Starting training...")

train_result = trainer.train()

logger.info(
    "Training complete: %s",
    train_result.metrics,
)


# ============================================================
# BEST MODEL
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

eval_results = trainer.evaluate()

logger.info(
    "Final evaluation: %s",
    eval_results,
)

wandb.log({
    f"final/{k}": v
    for k, v in eval_results.items()
    if isinstance(v, (int, float))
})


# ============================================================
# W&B MODEL ARTIFACT
# ============================================================

artifact = wandb.Artifact(
    name=f"{RUN_NAME}-adapter",
    type="model",
    metadata={
        "base_model": MODEL_NAME,
        "dataset": TRAIN_DATASET,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
)

artifact.add_dir(OUTPUT_DIR)

wandb.log_artifact(artifact)


# ============================================================
# HUGGING FACE
# ============================================================

logger.info(
    "Pushing adapter to %s",
    HF_REPO_ID,
)

trainer.model.push_to_hub(
    HF_REPO_ID,
    token=HF_TOKEN,
)

tokenizer.push_to_hub(
    HF_REPO_ID,
    token=HF_TOKEN,
)

logger.info("Done.")

wandb.finish()