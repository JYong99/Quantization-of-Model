from huggingface_hub import snapshot_download
import os

# Define the model repository
model_id = "mistralai/Ministral-3-8B-Instruct-2512-BF16"

# Define local directory to save the model
local_dir = os.path.join(os.getcwd(), "models", "Ministral-3-8B-Instruct-2512-BF16")

# Create directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_id} to {local_dir}...")

# Download the model with safetensors and tokenizer files
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    revision="main"
)

print(f"Model downloaded successfully to: {local_dir}")