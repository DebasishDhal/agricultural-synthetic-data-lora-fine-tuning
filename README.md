# Agricultural Synthetic Data LoRA Fine-Tuning

A reproducible workflow for fine-tuning and evaluating **Meta Llama 3.2 3B Instruct** on agricultural question-answering data using parameter-efficient fine-tuning (PEFT/LoRA). The workflow is designed for Google Colab and tracks experiments with Weights & Biases (W&B), with optional adapter publishing to the Hugging Face Hub.

> **Project status:** This README documents the workflow represented by the project scripts. 

For result figures, please visit `results/README.md`.

## Result Summary
- LORA fine-tuning `increased` the model's accuracy (in choosing one of 4 options type of task) by `18%`, when the fine-tuning dataset and benchmark dataset are similar. (Primary work) (Here similar means that both the fine-tuning dataset & benchmark dataset had choose the correct option type of task)
- LORA fine-tuning `decreased` the model's accuracy by `4%` when the fine-tuning dataset and benchmark dataset are of different natures. (Secondary work)
- For more information, please open `results/README.md`

## What this project does

The workflow is divided into four stages:

1. **Load and prepare data** — load a Hugging Face dataset, clean question-answer records, remove duplicate questions, and format examples as user/assistant conversations.
2. **Fine-tune the model** — train a LoRA adapter with Hugging Face Transformers, TRL's `SFTTrainer`, and PEFT. Depending on the training configuration, use ordinary LoRA with full-precision model weights or QLoRA with 4-bit NF4 quantization.
3. **Evaluate consistently** — evaluate the base model and the adapter on the same held-out benchmark split, using deterministic generation and multiple-choice answer extraction.
4. **Record and publish artifacts** — log training/evaluation metrics and prediction tables to W&B, save CSV results, and optionally upload the adapter and tokenizer to the Hugging Face Hub.

## Model and datasets

### Base model

- **Model:** [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Output artifact:** a LoRA adapter, which is loaded on top of the original base model. An adapter repository is not a standalone copy of the full model.
- **Access:** the user running the code must have accepted Meta's model terms and have Hugging Face access. A Hugging Face token is used to download the gated model and, if enabled, publish the adapter.

### Dataset sources referenced by the workflow

| Dataset | Intended role | Notes |
|---|---|---|
| [`DebasishDhal99/agriculture-synthetic-finetuning-dataset`](https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset) | Synthetic training/validation data and a held-out hard benchmark | The scripts shared for this workflow use configuration `v0` and reference a `benchmark_hard` split. Verify available configurations, split names, and columns before running. |
| [`KisanVaani/agriculture-qa-english-only`](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) | Alternative agricultural QA training source | Also referenced in a training-script variant. Set the training dataset explicitly and record it in the experiment configuration. |

**Important experiment-design check:** the scripts shared for this project reference both the synthetic dataset and the KisanVaani dataset. Make sure the training run and benchmark run use the dataset pairing intended by the experiment, and record both dataset IDs in W&B. Do not describe a result as “synthetic-data fine-tuning” unless that is the dataset actually used to train the adapter.

Before running, inspect the loaded dataset's columns and splits. Training code variants may expect `question` plus `answers`; the benchmark evaluator may expect `question` (or ChatML-style `messages`) plus `answer`. Adapt the field mapping to the actual dataset schema rather than assuming column names are interchangeable.

## Fine-tuning configuration

The shared training configuration uses the following representative settings. The values should be confirmed against the exact configuration used for each run and W&B record.

| Parameter | Value shown in the shared configuration |
|---|---:|
| Epochs | 2 |
| Learning rate | `2e-4` |
| Per-device batch size | 2 |
| Gradient accumulation | 8 steps |
| Effective batch size (one device) | 16 examples per optimizer update |
| Maximum sequence length | 1,024 tokens |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Weight decay | 0.01 in the shared configuration |

### LoRA versus QLoRA

These are related but different modes. Use the mode that matches the experiment you are running:

- **LoRA (no quantization):** load the base model without a `BitsAndBytesConfig`; apply a PEFT LoRA adapter.
- **QLoRA:** load the base model in 4-bit form with `BitsAndBytesConfig` (the shared QLoRA variant uses NF4 and double quantization), then train the LoRA adapter.

Do not label a run simply “LoRA” or “QLoRA” based on the repository name; record the actual model-loading and quantization settings in W&B. BF16 should only be selected when the active GPU supports it; otherwise use FP16 when supported by the hardware and configuration.

## Setup

The scripts shown are designed for **Google Colab with a CUDA T4 GPU**. They use `google.colab.userdata` to read secrets, so that import will not work in an ordinary local Python session without adaptation. For local Python session, appropriate minor modifications are needed.

Install compatible versions of the packages used by your selected script:

```bash
pip install -U transformers datasets trl peft accelerate pandas wandb huggingface_hub
```

For a QLoRA run, also install a compatible `bitsandbytes` version:

```bash
pip install -U bitsandbytes
```

PyTorch is intentionally not included in these commands: use the PyTorch/CUDA build provided by your Colab runtime unless you have a specific reason to change it. If the repository includes a pinned `requirements.txt`, prefer testing that environment as a unit rather than upgrading core packages independently.

### Configure secrets

Add these secrets in Colab's Secrets panel and enable notebook access:

- `hf_write_token` — Hugging Face token with permission to download the gated base model; write permission is needed if uploading artifacts.
- `wandb_token` — W&B API key for logging runs.

Example:

```python
from google.colab import userdata

hf_token = userdata.get("hf_write_token")
wandb_token = userdata.get("wandb_token")

if not hf_token:
    raise ValueError("Missing Colab secret: hf_write_token")
if not wandb_token:
    raise ValueError("Missing Colab secret: wandb_token")
```

Never commit tokens, API keys, or notebook secret values to this repository.

## Run the workflow

The precise filenames and commands should follow the scripts/notebooks present in the repository. At a high level:

1. **Choose the base model and dataset IDs.** Set the training dataset, dataset configuration, benchmark dataset, and split explicitly.
2. **Check the data schema.** Print the dataset's available splits, column names, and a representative row. Verify that the question and expected-answer fields match the preprocessing and evaluator functions.
3. **Record the experiment.** Start a W&B run with the base model, training dataset, benchmark dataset, dataset configuration, split, training hyperparameters, LoRA settings, precision, and quantization mode.
4. **Train the adapter.** Format records as conversations, instantiate `SFTTrainer`, and save the adapter checkpoint.
5. **Run baseline evaluation.** Evaluate the base model against the selected benchmark and save the predictions.
6. **Run adapter evaluation.** Load the same base model, attach the trained adapter with `PeftModel.from_pretrained`, and evaluate on the exact same benchmark.
7. **Compare results.** Compare accuracy and invalid-output rate, inspect per-question predictions, and retain the generated CSV and W&B run as artifacts.
8. **Publish if needed.** Push the adapter and tokenizer to a Hugging Face model repository only after checking the target repository ID and token permissions.

## Benchmark methodology

The shared evaluator uses deterministic generation (`do_sample=False`) and instructs the model to answer a multiple-choice question with a single option letter (`A`, `B`, `C`, or `D`). It extracts a predicted option from the generated text and compares it with the expected option.

Reported metrics include:

- **Accuracy:** correct predictions divided by the total number of benchmark examples.
- **Invalid rate:** outputs from which the evaluator cannot extract a valid option letter, divided by the total number of examples.
- **Incorrect count:** total examples minus correct predictions minus invalid predictions.

The shared scripts save row-level predictions—including expected answer, prediction, raw model output, and correctness—to CSV and log an equivalent prediction table to W&B.

### Keep evaluation held out

Use training data for optimization and validation data for checkpoint selection. Keep the hard benchmark separate from training and checkpoint selection; use it for final comparisons. Evaluate base and adapter models with the same prompt, answer extraction, generation settings, and benchmark rows so the comparison is meaningful.

### Validate answer parsing

The evaluator assumes an answer field containing an option letter or text with an unambiguous standalone option letter. If the dataset stores the full answer text, mapping that text to A/B/C/D is required. A simple regular expression can accidentally extract a letter from ordinary prose, so inspect examples and manually verify a sample of the answer mapping before trusting aggregate accuracy.

## W&B and saved artifacts

W&B is used to track the configuration and results of training or benchmark runs. The shared scripts log aggregate metrics and a per-example prediction table. The evaluator also writes a local CSV (the filename depends on the script variant, for example `base_benchmark_results.csv` or `lora_benchmark_results.csv`).

Record at least the following for every benchmark run:

- base model and adapter repository/revision;
- training dataset and benchmark dataset IDs;
- dataset configuration and benchmark split;
- prompt/template and generation settings;
- precision and quantization mode;
- benchmark size, accuracy, invalid rate, and output CSV.

Use distinct W&B run names and CSV filenames for base, LoRA, and QLoRA evaluations to avoid overwriting or confusing results.

## Common issues

| Symptom | What to check |
|---|---|
| Library compatibility issue | Ensure `torchao` version >= 0.16.0 and install everything in `requirements.txt` |
| Hugging Face `401`/`403` or gated-model error | Accept the base model's terms, verify the token, and confirm the Colab secret is accessible. |
| `KeyError` for dataset configuration or split | Print `dataset.keys()` and select a configuration/split that actually exists. |
| Missing `question`, `answer`, or `answers` field | Inspect the dataset's column names and update preprocessing/evaluation field mapping. |
| BF16/runtime dtype error | Select dtype based on the active GPU's support; use FP16 when BF16 is not supported. |
| CUDA out-of-memory | Reduce per-device batch size and/or sequence length; gradient accumulation can preserve the effective batch size when reducing the microbatch. |
| W&B login failure | Check that the `wandb_token` Colab secret exists, is valid, and is being passed consistently. |
| Poor or suspiciously high benchmark accuracy | Review label parsing, option formatting, duplicate/overlapping examples, prompt consistency, and whether benchmark data was kept out of training. |
| Adapter loading error | Confirm that the adapter repo exists, is accessible, and was trained from the specified base model. |

## Reproducibility and limitations

- Synthetic data can contain incorrect, repetitive, or overly easy examples. Benchmark quality and label correctness matter as much as dataset size. This is for demo purpose only.
- An MCQ accuracy score measures this specific benchmark and answer-extraction procedure; it does not by itself establish that a model's agricultural advice is safe or reliable in real-world farming conditions.
