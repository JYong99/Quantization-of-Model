from huggingface_hub import snapshot_download

model_id = "HuggingFaceTB/SmolLM3-3B"  # Use the correct repo for your variant
snapshot_download(repo_id=model_id, local_dir="smollm3-3b")
