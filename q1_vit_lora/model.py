import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

def get_model(num_classes=100, use_lora=False, lora_r=8, lora_alpha=8, lora_dropout=0.1):
    model_name = "WinKawaks/vit-small-patch16-224" 
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    if not use_lora:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["classifier"]
    )
    
    peft_model = get_peft_model(model, config)
    for param in peft_model.classifier.parameters():
        param.requires_grad = True
        
    return peft_model
