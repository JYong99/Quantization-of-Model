from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_name = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
path = './Llama-4-Scout-17B-16E-Instruct'

# Download and save the model with memory optimizations
model = Llama4ForConditionalGeneration.from_pretrained(
    model_name, 
    use_safetensors=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16, 
    device_map="auto",
    max_memory={"cpu": "40GiB"},
    offload_folder="./offload_temp"
)
model.save_pretrained(path)

# Download and save the processor (handles both tokenizer and image processing)
processor = AutoProcessor.from_pretrained(model_name)
processor.save_pretrained(path)