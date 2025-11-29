import json
import torch
from transformers import AutoConfig, AutoModel, Phi3Model, Phi3Config
from peft import LoraConfig, get_peft_model


def LLMprepare(configs):
    # Load model configurations from JSON file
    with open('llm_config.json', 'r') as f:
        model_configurations = json.load(f)

    config = model_configurations.get(configs.llm_type)
    if not config:
        raise ValueError("Unsupported LLM type")

    model_path = config["path"]
    d_model = config["dim"]

    model_class = eval(config["model_class"])
    print(model_class)
    print(model_path)
    breakpoint()
    llm_model = model_class.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=False,
        torch_dtype=torch.bfloat16
    )
    # Apply LoRA modifications if requested
    if configs.lora:
        breakpoint()
        lora_config = config['lora_config']
        lora_config['lora_dropout'] = configs.dropout  # Update dropout with user input
        llm_model = get_peft_model(llm_model, LoraConfig(**lora_config))

        for name, param in llm_model.named_parameters():
            param.requires_grad = ('lora' in name)
    else:
        for name, param in llm_model.named_parameters():
            param.requires_grad = False

    return llm_model, d_model
