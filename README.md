# Llama-Factory Fork: Training Code for "Translating to a Low-resource Language with Compiler Feedback: A Case Study on Cangjie"

This repository is a **fork** of [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory). It contains the **training code** used for the paper "Translating to a Low-resource Language with Compiler Feedback: A Case Study on Cangjie". The training parameters for **pretrain**, **SFT**, and **KTO** are provided in the `hparams` directory.

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/codeshuttler/LLaMA-Factory-Cangjie.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, awq, aqlm, vllm, galore, badam, adam-mini, qwen, modelscope, openmind, quality

> [!TIP]
> Use `pip install --no-deps -e .` to resolve package conflicts.

### Data Preparation

For dataset details, refer to [data/README.md](data/README.md). You can either use datasets from HuggingFace, ModelScope, or Modelers hub, or load a dataset from your local disk.

> [!NOTE]
> Update `data/custom_dataset_info.json` to use your custom dataset.

### Quickstart
After setting the path of the relevant dataset in `data/custom_dataset_info.json`, use the following commands to run **Pretraining**, **Fine-tuning**, and **KTO** for the qwen2-7b model:

```bash
llamafactory-cli train hparams/cangjie/cangjie-qwen2-7b/cangjie_lora_pretrain.yaml
llamafactory-cli train hparams/cangjie/cangjie-qwen2-7b/cangjie_lora_sft.yaml
llamafactory-cli train hparams/cangjie/cangjie-qwen2-7b/cangjie_lora_kto.yaml
```

For advanced usage, including distributed training, refer to [examples/README.md](examples/README.md).

> [!TIP]
> Use `llamafactory-cli help` to show help information.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Follow the model licenses to use corresponding model weights: [Baichuan 2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Community%20License%20for%20Baichuan%202%20Model.pdf) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [Llama 3](https://llama.meta.com/llama3/license/)

## Citation

If you find this work helpful, please cite it as:

```bibtex
...
```

## Acknowledgement

This repository benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora), and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful contributions.
