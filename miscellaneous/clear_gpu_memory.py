import gc
import torch

# Run this when your GPU memory is getting nearly full. Downside is that you have to re-download the model.

# 1. Delete model and tokenizer variables if they exist
if "model" in globals():
    del base_model
if "tokenizer" in globals():
    del tokenizer

# 2. Force Python garbage collection
gc.collect()

# 3. Clear PyTorch's internal VRAM cache
torch.cuda.empty_cache()
