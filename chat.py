import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1"  # the device to load the model onto

# model_path = "saves/nyagpt-qwen2-7b-v0.3/lora/pretrain/checkpoint-8000"
# model_path = "saves/policygpt-qwen2-7b/lora/sft/checkpoint-23000"
# model_path = "Qwen/Qwen2-7B-Instruct"
# model_path = "saves/cangjie-qwen2-7b/lora/sft/checkpoint-10221"
# model_path = "saves/cangjie-qwen2.5-7b/lora/pretrain/checkpoint-100"
# model_path = "saves/policygpt-qwen2-7b/lora/sft/checkpoint-24500"
# model_path = "/data/xsj/save/demo/sft/checkpoint-42899"
# model_path = "saves/policygpt-qwen1.5-1.8b-v0.1/full/sft/checkpoint-12500"
# model_path = "saves/cangjie-qwen2.5-7b/lora/sft/checkpoint-4700"
model_path = "saves/cangjie-qwen2-7b/lora/sft_full_v3/checkpoint-44182"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！请问有什么可以帮助你的吗？"},
]

print(f"当前模型：{model_path}")
while True:
    user_message = input("用户：")
    if user_message.strip() in ["exit", "quit", "q"]:
        break
    if user_message.startswith("read:"):
        with open(user_message[5:], "r", encoding="utf-8") as f:
            user_message = f.read()
    
    messages.append({"role": "user", "content": user_message})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    attention_mask = torch.ones(
        model_inputs.input_ids.shape, dtype=torch.long, device=device
    )

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("模型：" + response)
    messages.append({"role": "assistant", "content": response})
