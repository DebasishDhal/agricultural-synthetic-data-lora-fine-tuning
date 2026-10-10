Two sets of dataset are being used in this experiment.

Dataset 1: 
- Training Dataset: [KisanVaani/agriculture-qa-english-only train split](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) (1999 rows)
- Eval Dataset: [KisanVaani/agriculture-qa-english-only train split](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) (233 rows)
- Benchmark Dataset: [bharatgenai/BhashaBench-Krishi](https://huggingface.co/datasets/bharatgenai/BhashaBench-Krishi) (first 500 rows)

It didn't work out because the the fine-tuning goal and the benchmark goal were different. In fine-tuning, we taught the model to generate free-flow text answers, while in the benchmark dataset we are expecting the model to choose one of the 4 correct options (A/B/C/D).


Dataset 2: (Final Dataset)
- Training Dataset: [DebasishDhal99/agriculture-synthetic-finetuning-dataset, train split](https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset) (1000 rows)
- Eval Dataset: [DebasishDhal99/agriculture-synthetic-finetuning-dataset, validation split](https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset) (100 rows)
- Benchmark Dataset: [DebasishDhal99/agriculture-synthetic-finetuning-dataset, benchmark_split split](https://huggingface.co/datasets/DebasishDhal99/agriculture-synthetic-finetuning-dataset) (300 rows)


- The dataset 2 is a completely synthetic dataset, made via ChatGPT.
- Initially, the model was getting a full 100% score on the `benchmark` split of the dataset. So, a harder split (`benchmark_hard`) was made.