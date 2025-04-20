llamafactory-cli train hparams/cangjie_lora_pretrain.yaml
llamafactory-cli train hparams/cangjie_lora_sft.yaml


python scripts/pissa_init.py --model_name_or_path Qwen/Qwen1.5-14B --output_dir saves/PolicyGPT-Qwen1.5-14B/lora/pretrain/pissa_init
