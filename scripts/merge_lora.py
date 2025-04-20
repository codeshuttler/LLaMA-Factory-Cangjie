

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
device = "cpu"  # the device to load the model onto

base_model_path = "Qwen/Qwen2-7B"
# lora_path = "saves/cangjie-qwen2-7b/lora/sft/checkpoint-3000"
lora_path = "saves/cangjie-qwen2-7b/lora/sft/checkpoint-10221"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map=device,
)

model = PeftModel.from_pretrained(
    base_model, lora_path, is_trainable=False
)
out_path = "cangjie-qwen2-7b-v0.3-merged"

# model.save_pretrained(pissa_convert_dir, safe_serialization=True)
model = model.merge_and_unload()
model.save_pretrained(
    out_path, safe_serialization=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(out_path)
