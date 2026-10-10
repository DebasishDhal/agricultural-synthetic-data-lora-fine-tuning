import os
import re
import json
import torch
import pandas as pd
import wandb
from peft import PeftModel
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Results: Scored 90% score

# ============================================================
# CONFIG
# ============================================================


bch_dataset_id = "DebasishDhal99/agriculture-synthetic-finetuning-dataset"
ft_dataset_id = "DebasishDhal99/agriculture-synthetic-finetuning-dataset"
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_name = base_model_id.split("/")[-1]

# base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = base_model_id.split("/")[-1]

DATASET_CONFIG = "v0"
BENCHMARK_SPLIT = "benchmark_hard"

MAX_NEW_TOKENS = 8

from google.colab import userdata
import wandb

hf_token = userdata.get("hf_write_token") #This is to pull base model from huggingface (incase its gated). Also used to push fine-tuned adapters to huggingface for persistence
wandb_token = userdata.get("wandb_token") #Store training/benchmark logs on w&b

wandb.login(key=wandb_token)

WANDB_PROJECT_NAME = 'AgriFineTuningDemo-Sep2026'
# ============================================================
# WANDB
# ============================================================

wandb.init(
    project=WANDB_PROJECT_NAME,
    name=f"lora-hardFineTunedLlama3BInstruct-benchmark-hard_synthetic-data-{model_name}",
    config={
        "model": base_model_id,
        "dataset": bch_dataset_id,
        "dataset_config": DATASET_CONFIG,
        "benchmark_split": BENCHMARK_SPLIT,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
)

ADAPTER_ID = "DebasishDhal99/Llama-3.2-3B-Instruct-synthetic-agriculture-lora"

# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(
    bch_dataset_id,
    name=DATASET_CONFIG,
    token=hf_token,
)

benchmark = dataset[BENCHMARK_SPLIT]

print(dataset)
print(f"Benchmark examples: {len(benchmark)}")

# ============================================================
# INSPECT COLUMNS
# ============================================================

print("\nColumns:")
print(benchmark.column_names)

print("\nFirst example:")
print(benchmark[0])

# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    token=hf_token,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_ID,
    token=hf_token,
) # combining the base model with the fine-tuned adapter model

model.eval()

# Llama doesn't normally have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# HELPERS
# ============================================================

def get_question(row):
    """
    Supports the CSV structure generated previously:

        question
        answer
        answer_text

    and also ChatML-style messages if present.
    """

    # Case 1: explicit question column
    if "question" in row:
        return row["question"]

    # Case 2: messages column
    if "messages" in row:
        messages = row["messages"]

        if isinstance(messages, str):
            messages = json.loads(messages)

        for message in messages:
            if message["role"] == "user":
                return message["content"]

    raise ValueError(
        f"Cannot find question column. Available columns: {row.keys()}"
    )


def get_expected_answer(row):
    """
    Expected answer should be A/B/C/D.
    """

    answer = row.get("answer", None)

    if answer is None:
        raise ValueError(
            f"Cannot find answer column. Available columns: {row.keys()}"
        )

    answer = str(answer).strip().upper()

    # Extract A/B/C/D if answer contains additional text
    match = re.search(r"\b([ABCD])\b", answer)

    if match:
        return match.group(1)

    raise ValueError(f"Could not parse expected answer: {answer}")


def extract_prediction(text):
    """
    Extract A/B/C/D from model output.

    We strongly prefer the first standalone A/B/C/D.
    """

    text = text.strip().upper()

    # First try a standalone letter
    match = re.search(r"\b([ABCD])\b", text)

    if match:
        return match.group(1)

    # Handle outputs such as:
    # "A."
    # "(A)"
    # "Answer: A"
    match = re.search(r"(?:^|[\s:(])([ABCD])(?:[.)\s]|$)", text)

    if match:
        return match.group(1)

    return None


# ============================================================
# EVALUATION
# ============================================================

results = []

correct = 0
invalid = 0

print("\nStarting benchmark...\n")

for idx, row in enumerate(benchmark):

    question = get_question(row)
    expected = get_expected_answer(row)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural question answering assistant. "
                "Answer the multiple-choice question by returning only "
                "the letter A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    prediction = extract_prediction(generated_text)

    is_correct = prediction == expected

    if prediction is None:
        invalid += 1
    elif is_correct:
        correct += 1

    results.append({
        "index": idx,
        "question": question,
        "expected": expected,
        "prediction": prediction,
        "raw_output": generated_text,
        "correct": is_correct,
    })

    print(
        f"[{idx + 1:03d}/{len(benchmark)}] "
        f"Expected={expected} "
        f"Predicted={prediction} "
        f"Correct={is_correct}"
    )


# ============================================================
# METRICS
# ============================================================

total = len(results)

accuracy = correct / total
invalid_rate = invalid / total

print("\n" + "=" * 60)
print("BASE MODEL BENCHMARK")
print("=" * 60)

print(f"Model:             {base_model_id}")
print(f"Benchmark size:    {total}")
print(f"Correct:           {correct}")
print(f"Incorrect:         {total - correct - invalid}")
print(f"Invalid:           {invalid}")
print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Invalid rate:      {invalid_rate:.4f} ({invalid_rate * 100:.2f}%)")

# ============================================================
# WANDB LOGGING
# ============================================================

wandb.log({
    "benchmark_size": total,
    "correct": correct,
    "incorrect": total - correct - invalid,
    "invalid": invalid,
    "accuracy": accuracy,
    "invalid_rate": invalid_rate,
})

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = "base_benchmark_results.csv"

results_df.to_csv(
    results_path,
    index=False,
)

print(f"\nResults saved to: {results_path}")

# Log prediction table to W&B
wandb_table = wandb.Table(
    dataframe=results_df
)

wandb.log({
    "benchmark_predictions": wandb_table
})

wandb.finish()