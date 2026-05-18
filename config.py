import torch

model_name = "openai/whisper-small"
dataset_name = "gmenon/slt-lyrics-audio"
out_DIR = "./whisper-lora-autolyrics"

# Hyperparameters
size_batch = 4
grad_steps = 2
rate_learning = 1e-4
max_steps = 100

device = "cuda" if torch.cuda.is_available() else "cpu"