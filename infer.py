import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1"  # the device to load the model onto

# model_path = "saves/NyaGPT-Qwen2-7B/lora/pretrain/checkpoint-24000"
model_path = "saves/PolicyGPT-Qwen1.5-4B/full/sft/checkpoint-10000"
# model_path = "Qwen/Qwen1.5-1.8B-chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

"因为模型比较小，在通用领域的能力是会下降的。"
# prompt = "介绍一下绝区零？"
# prompt = "续写发言稿：\n在全市河道环境综合整治工作推进会上的讲话\n同志们："
# prompt = "写一篇发言稿：在内蒙古退役军人就业创业促进会设立盟市办事机构工作会议上的讲话"
# prompt = "续写发言稿：在内蒙古退役军人就业创业促进会设立盟市办事机构工作会议上的讲话。\n在全国两会刚刚胜利闭幕、经济秩序稳步复苏的积极态势下，我们隆重召开内蒙古退役军人就业创业促进会设立盟市办事机构专题工作会议，主要任务是进一步统一思想、提高认识，理清思路、明确责任，有力推动内蒙古退役军人就业创业促进会在军地领导和相关部门的关心指导下，在社会各界的支持帮助下，尽快打开工作局面，快速创造佳绩，为党委政府分忧，为退役军人解难，不辜负广大退役士兵和社会各界的重托与期望。借此机会，讲三点意见：\n一、要切实深刻领会办会宗旨"
prompt = "背诵一下李白的静夜思"
prompt = "hashmap底层数据结构"
# prompt = "介绍一下滕王阁序"
# prompt = "蒙古人会吃蒙古包吗？"
# prompt = "蒙古包是什么？"
# prompt = "在achlinux中如何安装open-vm-tools"
# prompt = "树上一共8只鹦鹉，开枪打死一只，请问树上还剩下几只？"
# prompt = "请你扮演一位在网络游戏直播间发弹幕的观众。这位观众正在看主播直播带货。"
# prompt = "日食和月食有什么区别"
# prompt = "小红有5个哥哥，3个姐姐，她的哥哥有多少个姐妹？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！请问有什么可以帮助你的吗？"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(
    json.dumps(
        messages
        + [
            {"role": "assistant", "content": response},
        ],
        ensure_ascii=False,
        indent=2,
    )
)
