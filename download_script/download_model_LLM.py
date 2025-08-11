from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'HuggingFaceTB/SmolLM3-3B'
path = './SmolLM3-3B'

# Download and save the model with memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    use_safetensors=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16, 
    device_map="auto",          # Automatically distribute across available devices
    max_memory={"cpu": "40GiB"},  # Adjust based on your system
    offload_folder="./offload_temp"  # Temporary offload directory
)
model.save_pretrained(path)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(path)
