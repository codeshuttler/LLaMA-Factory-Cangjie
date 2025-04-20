

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
device = "cpu"  # the device to load the model onto

base_model_path = "Qwen/Qwen2-7B"
lora_path = "saves/NyaGPT-Qwen2-7B-v0.1/lora/sft/checkpoint-6000"
pissa_init_path = "saves/NyaGPT-Qwen2-7B-v0.1/lora/sft/pissa_init"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map=device,
)

model = PeftModel.from_pretrained(
    base_model, lora_path, is_trainable=False
)
pissa_convert_dir = "nyagpt-qwen2-7b-v0.1-lora"

# model.save_pretrained(pissa_convert_dir, safe_serialization=True)
model.save_pretrained(
    pissa_convert_dir, safe_serialization=True, path_initial_model_for_weight_conversion=pissa_init_path
)