# ============================================================
# LoRA SFT — Agriculture
#
# Base model + synthetic agricultural dataset
# NO QUANTIZATION
# Adapter-only output
# W&B + Hugging Face Hub
#
# Dataset:
#   train       -> 1000
#   validation  -> 100
#   benchmark   -> 300/80, NEVER USED HERE
# ============================================================

# Fetches the base model + training/validation dataset -> Initializes a separate Lora matrix -> Updates its weight based on training loss
# Chooses the best model checkpoint based on eval loss -> Best model checkpoint is persisted in huggingface as an adapter
# All the training steps are stored in weights and bias

# To use the trained adapter, pull the original base model + newly trained adapter -> Combine them both -> Use it

# Requirements: without any modification, this code will only work on colab if you have "hf_write_tokn" and "wandb_token" stored in colab secrets.
# Requirements: Make sure you have the requirements.txt installed

import os
import re
import logging
import time
import platform

import torch
import wandb

from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from google.colab import userdata
import wandb

hf_token = userdata.get("hf_write_token") #This is to pull base model from huggingface (incase its gated). Also used to push fine-tuned adapters to huggingface for persistence
wandb_token = userdata.get("wandb_token") #Store training/benchmark logs on w&b

wandb.login(key=wandb_token)

WANDB_PROJECT_NAME = 'AgriFineTuningDemo-Sep2026'


base_model_id = "meta-llama/Llama-3.2-3B-Instruct" #https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
bch_dataset_id = "DebasishDhal99/agriculture-synthetic-finetuning-dataset" #https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset
ft_dataset_id = "DebasishDhal99/agriculture-synthetic-finetuning-dataset" #https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset
model_name = base_model_id.split("/")[-1]

# ============================================================
# CONFIG
# ============================================================

HF_TOKEN = hf_token
WANDB_API_KEY = wandb_token

MODEL_NAME = base_model_id

TRAIN_DATASET = ft_dataset_id
DATASET_CONFIG = "v0"

HF_REPO_ID = (
    f"DebasishDhal99/{model_name}-synthetic-agriculture-lora"
) #Adapter generated to be uploaded here

WANDB_PROJECT = WANDB_PROJECT_NAME

RUN_NAME = f"{model_name}-synthetic-agriculture-lora"

OUTPUT_DIR = (
    f"./{model_name}-synthetic-agriculture-lora"
)

SEED = 42

EPOCHS = 2
LEARNING_RATE = 2e-4

BATCH_SIZE = 2
GRAD_ACCUMULATION = 8

MAX_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# How often to run validation.
#
# Your effective batch size is:
#
# 2 * 8 = 16
#
# 1000 / 16 ~= 63 optimizer steps / epoch
#
# Therefore 25 steps gives several validation points
# during a 2-epoch run.
EVAL_STEPS = 25

# Generation settings for MCQ validation
MCQ_MAX_NEW_TOKENS = 8


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DTYPE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU required for this training setup."
    )

USE_BF16 = torch.cuda.is_bf16_supported()

DTYPE = (
    torch.bfloat16
    if USE_BF16
    else torch.float16
)

logger.info("Using dtype: %s", DTYPE)


# ============================================================
# W&B
# ============================================================

wandb.login(key=WANDB_API_KEY)

wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,

    config={

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "base_model": MODEL_NAME,
        "model_name": model_name,

        # ----------------------------------------------------
        # TRAINING METHOD
        # ----------------------------------------------------

        "training_method": "LoRA",
        "quantization": "none",

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        "dataset": TRAIN_DATASET,
        "dataset_config": DATASET_CONFIG,

        "train_split": "train",
        "validation_split": "validation",
        "benchmark_split": "benchmark",

        "train_examples": 1000,
        "validation_examples": 100,

        # Benchmark is intentionally NOT used during training
        "benchmark_used_during_training": False,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "optimizer": "adamw_torch",

        "weight_decay": 0.01,

        "lr_scheduler": "linear",

        "warmup_ratio": 0.0,

        # ----------------------------------------------------
        # BATCHING
        # ----------------------------------------------------

        "per_device_train_batch_size": BATCH_SIZE,

        "per_device_eval_batch_size": BATCH_SIZE,

        "gradient_accumulation_steps":
            GRAD_ACCUMULATION,

        "effective_batch_size":
            BATCH_SIZE * GRAD_ACCUMULATION,

        # ----------------------------------------------------
        # SEQUENCE
        # ----------------------------------------------------

        "max_length": MAX_LENGTH,

        # ----------------------------------------------------
        # LoRA
        # ----------------------------------------------------

        "lora_r": LORA_R,

        "lora_alpha": LORA_ALPHA,

        "lora_dropout": LORA_DROPOUT,

        "lora_bias": "none",

        "lora_task_type": "CAUSAL_LM",

        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        "dtype": str(DTYPE),

        "bf16": USE_BF16,

        "fp16": not USE_BF16,

        "gradient_checkpointing": False,

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        "eval_strategy": "steps",

        "eval_steps": EVAL_STEPS,

        "evaluation_metric": "MCQ accuracy",

        "selection_metric": "eval_mcq_accuracy",

        "greater_is_better": True,

        "eval_loss_logged": True,

        "mcq_accuracy_logged": True,

        "invalid_answer_rate_logged": True,

        # ----------------------------------------------------
        # CHECKPOINTING
        # ----------------------------------------------------

        "save_strategy": "steps",

        "save_steps": EVAL_STEPS,

        "save_total_limit": 3,

        "load_best_model_at_end": True,

        # ----------------------------------------------------
        # GENERATION / MCQ EVALUATION
        # ----------------------------------------------------

        "mcq_max_new_tokens":
            MCQ_MAX_NEW_TOKENS,

        "mcq_decoding": "greedy",

        "mcq_temperature": None,

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        "seed": SEED,

        "data_seed": SEED,

        # ----------------------------------------------------
        # HARDWARE / SOFTWARE
        # ----------------------------------------------------

        "python": platform.python_version(),

        "torch": torch.__version__,

        "cuda": torch.version.cuda,

        "gpu": torch.cuda.get_device_name(0),

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        "output_dir": OUTPUT_DIR,

        "hf_repo": HF_REPO_ID,

        "adapter_only": True,
    },
)

# ============================================================
# LOAD DATASET
# ============================================================

logger.info("Loading Hugging Face dataset...")

train_raw = load_dataset(
    TRAIN_DATASET,
    name=DATASET_CONFIG,
    split="train",
    token=HF_TOKEN,
)

validation_raw = load_dataset(
    TRAIN_DATASET,
    name=DATASET_CONFIG,
    split="validation",
    token=HF_TOKEN,
)

logger.info(
    "Raw train examples: %d",
    len(train_raw),
)

logger.info(
    "Raw validation examples: %d",
    len(validation_raw),
)

wandb.config.update({
    "raw_train_examples": len(train_raw),
    "raw_validation_examples": len(validation_raw),
})


# ============================================================
# DATA CLEANING
# ============================================================

def prepare_dataframe(dataset, split_name):

    df = dataset.to_pandas()

    logger.info(
        "%s columns: %s",
        split_name,
        list(df.columns),
    )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    if "question" not in df.columns:
        raise ValueError(
            f"{split_name} dataset has no 'question' column. "
            f"Columns: {list(df.columns)}"
        )

    df["question"] = (
        df["question"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Answer
    #
    # Your previous dataset used "answers".
    # The newer generated benchmark uses "answer".
    #
    # Support both.
    # --------------------------------------------------------

    if "answers" in df.columns:
        answer_column = "answers"

    elif "answer" in df.columns:
        answer_column = "answer"

    else:
        raise ValueError(
            f"{split_name} dataset has neither "
            "'answers' nor 'answer'. "
            f"Columns: {list(df.columns)}"
        )

    df[answer_column] = (
        df[answer_column]
        .astype(str)
        .str.strip()
    )

    # Standardize to "answer"
    if answer_column != "answer":
        df["answer"] = df[answer_column]

    # --------------------------------------------------------
    # Filter invalid rows
    # --------------------------------------------------------

    before = len(df)

    df = df[
        (df["question"].str.len() > 5)
        &
        (df["answer"].str.len() > 0)
    ]

    df = df.drop_duplicates(
        subset=["question"]
    ).reset_index(drop=True)

    after = len(df)

    logger.info(
        "%s: %d -> %d examples",
        split_name,
        before,
        after,
    )

    return df


train_df = prepare_dataframe(
    train_raw,
    "train",
)

validation_df = prepare_dataframe(
    validation_raw,
    "validation",
)


wandb.config.update({
    "clean_train_examples": len(train_df),
    "clean_validation_examples": len(validation_df),
})


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
                "content": example["answer"],
            },
        ]
    }


train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_df,
    preserve_index=False,
)


train_dataset = train_dataset.map(
    format_example,
    remove_columns=train_dataset.column_names,
)

validation_dataset = validation_dataset.map(
    format_example,
    remove_columns=validation_dataset.column_names,
)


logger.info(
    "Train: %d | Validation: %d",
    len(train_dataset),
    len(validation_dataset),
)


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
# MODEL
# ============================================================

logger.info(
    "Loading base model WITHOUT quantization..."
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=DTYPE,
    # device_map="auto",
    trust_remote_code=True,
)

model = model.to("cuda:0")

model.config.use_cache = False


# ============================================================
# LoRA CONFIG
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
# MCQ VALIDATION
# ============================================================

def extract_answer(text):

    """
    Extract A/B/C/D from model output.

    Handles outputs such as:

        A
        A.
        (A)
        Answer: A
        A. Nitrogen deficiency
    """

    text = text.strip().upper()

    # Prefer explicit answer patterns
    patterns = [
        r"ANSWER\s*[:\-]?\s*([ABCD])\b",
        r"^\s*\(?([ABCD])\)?(?:[\.\):\s]|$)",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:
            return match.group(1)

    return None


def normalize_expected_answer(answer):

    answer = str(answer).strip().upper()

    match = re.search(
        r"\b([ABCD])\b",
        answer,
    )

    if match:
        return match.group(1)

    return None


def evaluate_mcq_accuracy(
    model,
    dataset,
    tokenizer,
):

    was_training = model.training

    model.eval()

    correct = 0
    invalid = 0
    total = len(dataset)

    predictions = []

    for row in dataset:

        messages = row["messages"]

        # ----------------------------------------------------
        # Convert chat messages into Llama prompt
        # ----------------------------------------------------

        prompt = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model.generate(
                **inputs,

                max_new_tokens=MCQ_MAX_NEW_TOKENS,

                do_sample=False,

                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        prediction = extract_answer(
            generated_text
        )

        # ----------------------------------------------------
        # Expected answer
        # ----------------------------------------------------

        expected = normalize_expected_answer(
            messages[1]["content"]
        )

        is_correct = (
            prediction is not None
            and prediction == expected
        )

        if prediction is None:

            invalid += 1

        elif is_correct:

            correct += 1

        predictions.append({
            "expected": expected,
            "prediction": prediction,
            "raw_output": generated_text,
            "correct": is_correct,
        })

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    invalid_rate = (
        invalid / total
        if total > 0
        else 0.0
    )

    if was_training:
        model.train()

    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "correct": correct,
        "invalid": invalid,
        "total": total,
        "predictions": predictions,
    }


# ============================================================
# CUSTOM TRAINER
#
# Adds actual MCQ accuracy to the normal eval metrics.
# ============================================================

class MCQSFTTrainer(SFTTrainer):

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):

        # ----------------------------------------------------
        # Normal HF/TRL evaluation
        #
        # Gives us eval_loss.
        # ----------------------------------------------------

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        dataset_to_use = (
            eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        # ----------------------------------------------------
        # Actual MCQ accuracy
        # ----------------------------------------------------

        mcq_results = evaluate_mcq_accuracy(
            self.model,
            dataset_to_use,
            self.processing_class,
        )

        accuracy_key = (
            f"{metric_key_prefix}_mcq_accuracy"
        )

        invalid_key = (
            f"{metric_key_prefix}_invalid_rate"
        )

        metrics[accuracy_key] = (
            mcq_results["accuracy"]
        )

        metrics[invalid_key] = (
            mcq_results["invalid_rate"]
        )

        # ----------------------------------------------------
        # Log to W&B
        # ----------------------------------------------------

        self.log({
            accuracy_key: mcq_results["accuracy"],
            invalid_key: mcq_results["invalid_rate"],
        })

        logger.info(
            "MCQ validation accuracy: %.2f%% "
            "(%d/%d)",
            mcq_results["accuracy"] * 100,
            mcq_results["correct"],
            mcq_results["total"],
        )

        return metrics


# ============================================================
# TRAINING CONFIG
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

    loss_type="nll",

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=EVAL_STEPS,

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=EVAL_STEPS,

    save_total_limit=3,

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Select best checkpoint by actual agricultural MCQ
    # validation accuracy, NOT eval_loss.
    # --------------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="eval_mcq_accuracy",

    greater_is_better=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=10,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=USE_BF16,

    fp16=not USE_BF16,

    gradient_checkpointing=False,

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    report_to="wandb",

    run_name=RUN_NAME,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED,
)


# ============================================================
# TRAINER
# ============================================================

trainer = MCQSFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    peft_config=peft_config,
)


# ============================================================
# VERIFY TRAINABLE PARAMETERS
# ============================================================

trainable_params = sum(
    p.numel()
    for p in trainer.model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in trainer.model.parameters()
)

trainable_percentage = (
    100 * trainable_params / total_params
)

logger.info(
    "Total parameters: %.2f B",
    total_params / 1e9,
)

logger.info(
    "Trainable parameters: %.2f M",
    trainable_params / 1e6,
)

logger.info(
    "Trainable percentage: %.4f%%",
    trainable_percentage,
)

wandb.config.update({
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "trainable_percentage": trainable_percentage,
})


# ============================================================
# TRAIN
# ============================================================

logger.info(
    "Starting LoRA training..."
)

start_time = time.time()

train_result = trainer.train()

training_time = (
    time.time() - start_time
)

logger.info(
    "Training finished in %.2f minutes",
    training_time / 60,
)

wandb.log({
    "training/time_minutes":
        training_time / 60,
})


# ============================================================
# SAVE BEST ADAPTER
# ============================================================

logger.info(
    "Saving best LoRA adapter to %s",
    OUTPUT_DIR,
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# FINAL VALIDATION
# ============================================================

logger.info(
    "Running final validation..."
)

final_eval = trainer.evaluate()

logger.info(
    "Final validation metrics:"
)

for key, value in final_eval.items():

    logger.info(
        "  %s = %s",
        key,
        value,
    )

    if isinstance(value, (int, float)):

        wandb.log({
            f"final_eval/{key}": value
        })


# ============================================================
# PERPLEXITY
# ============================================================

if "eval_loss" in final_eval:

    perplexity = torch.exp(
        torch.tensor(
            final_eval["eval_loss"]
        )
    ).item()

    logger.info(
        "Validation perplexity: %.4f",
        perplexity,
    )

    wandb.log({
        "final_eval/perplexity":
            perplexity
    })


# ============================================================
# W&B ADAPTER ARTIFACT
# ============================================================

artifact = wandb.Artifact(
    name=f"{RUN_NAME}-adapter",
    type="model",

    metadata={

        "base_model": MODEL_NAME,

        "dataset": TRAIN_DATASET,

        "dataset_config": DATASET_CONFIG,

        "training_method": "LoRA",

        "quantization": "none",

        "dtype": str(DTYPE),

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,

        "gradient_accumulation":
            GRAD_ACCUMULATION,

        "lora_r": LORA_R,

        "lora_alpha": LORA_ALPHA,

        "lora_dropout": LORA_DROPOUT,

        "max_length": MAX_LENGTH,

        "seed": SEED,

        "selection_metric":
            "eval_mcq_accuracy",
    },
)

artifact.add_dir(
    OUTPUT_DIR
)

wandb.log_artifact(
    artifact
)


# ============================================================
# HUGGING FACE — ADAPTER ONLY
# ============================================================

logger.info(
    "Uploading LoRA adapter to %s",
    HF_REPO_ID,
)


README_CONTENT = f"""---
base_model: {MODEL_NAME}
library_name: peft
pipeline_tag: text-generation
tags:
  - agriculture
  - agricultural-qa
  - lora
  - peft
  - llama
  - synthetic-data
  - instruction-tuning
---

# {model_name} — Synthetic Agriculture LoRA

LoRA adapter fine-tuned from
[`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME})
on a synthetic agricultural question-answering dataset.

This repository contains the **LoRA adapter only**, not a merged
copy of the base model.

## Model

| Property | Value |
|---|---|
| Base model | `{MODEL_NAME}` |
| Fine-tuning method | LoRA |
| Quantization | None |
| Task | Agricultural instruction following / MCQ |
| Adapter only | Yes |

## Dataset

Dataset:

[`{TRAIN_DATASET}`](https://huggingface.co/datasets/{TRAIN_DATASET})

Configuration:

`{DATASET_CONFIG}`

| Split | Examples | Purpose |
|---|---:|---|
| train | 1,000 | LoRA training |
| validation | 100 | Checkpoint selection / validation |
| benchmark | 300 | Final evaluation only |

The benchmark split is **not used during fine-tuning or
checkpoint selection**.

## Training Configuration

### Optimization

| Parameter | Value |
|---|---|
| Epochs | `{EPOCHS}` |
| Learning rate | `{LEARNING_RATE}` |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| LR scheduler | Linear |
| Warmup ratio | `0.0` |

### Batch / sequence

| Parameter | Value |
|---|---|
| Per-device train batch size | `{BATCH_SIZE}` |
| Per-device eval batch size | `{BATCH_SIZE}` |
| Gradient accumulation | `{GRAD_ACCUMULATION}` |
| Effective batch size | `{BATCH_SIZE * GRAD_ACCUMULATION}` |
| Maximum sequence length | `{MAX_LENGTH}` |

### LoRA

| Parameter | Value |
|---|---|
| Rank (`r`) | `{LORA_R}` |
| Alpha | `{LORA_ALPHA}` |
| Dropout | `{LORA_DROPOUT}` |
| Bias | `none` |
| Task type | `CAUSAL_LM` |

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```
"""

README_PATH = os.path.join(OUTPUT_DIR, "README.md")

with open(README_PATH, "w", encoding="utf-8") as f:
    f.write(README_CONTENT)

from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=HF_REPO_ID,
    repo_type="model",
    private=False,
    exist_ok=True,
)

api.upload_file(
    path_or_fileobj=README_PATH,
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

trainer.model.push_to_hub(
    HF_REPO_ID,
    token=HF_TOKEN,
)

tokenizer.push_to_hub(
    HF_REPO_ID,
    token=HF_TOKEN,
)

logger.info(
    "LoRA adapter uploaded successfully."
)


# ============================================================
# FINISH
# ============================================================

wandb.finish()

logger.info(
    "Experiment complete."
)