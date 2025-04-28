import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1"  # the device to load the model onto


model_path = "Qwen/Qwen2-7B-Instruct"

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
