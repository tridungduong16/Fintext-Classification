import datetime
import os
import pdb
import re
import textwrap
import time
from datetime import datetime

import bitsandbytes as bnb
import hydra
import loralib as lora
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_int8_training,
                  prepare_model_for_kbit_training, set_peft_model_state_dict)
from peft.tuners.lora import LoraLayer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig,
                          LlamaForCausalLM, LlamaTokenizer, Trainer,
                          TrainingArguments, pipeline)
from transformers.generation.utils import GreedySearchDecoderOnlyOutput


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float16)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_finetune_config(target_modules, cfg):
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return lora_config, nf4_config


def get_model_and_tokenizer(cfg):
    if cfg.MODEL_NAME in cfg.FALCON_FAMILY:
        target_modules = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    elif cfg.MODEL_NAME in cfg.LLAMA_FAMILY:
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["up_proj", "down_proj"]

    lora_config, nf4_config = get_finetune_config(target_modules, cfg)

    if cfg.MODEL_NAME == "xgen-7b-8k-open-instruct":
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            trust_remote_code=True,
            quantization_config=nf4_config,
            cache_dir=cfg.CACHE_DIR,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)

    model = prepare_model_for_kbit_training(model)

    tokenizer.pad_token = tokenizer.eos_token
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    print_trainable_parameters(model)
    return model, tokenizer


def conduct_training(dataset, val_dataset, model, tokenizer, training_arguments):
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    trainer.train()
    return model


def get_data(tokenizer, cfg):
    df_train = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/train.csv")
    df_val = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/validation.csv")

    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.map(lambda sample: tokenizer(sample["prompt"]))

    val_dataset = Dataset.from_pandas(df_val)
    val_dataset = dataset.map(lambda sample: tokenizer(sample["prompt"]))

    return dataset, val_dataset


@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def main(cfg: DictConfig):
    model, tokenizer = get_model_and_tokenizer(cfg)
    dataset, val_dataset = get_data(tokenizer, cfg)
    training_arguments = TrainingArguments(
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        # warmup_steps=100,
        max_steps=cfg.EPOCHS,
        # num_train_epochs=cfg.EPOCHS,
        fp16=True,
        learning_rate=cfg.LEARNING_RATE,
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        output_dir=cfg.CACHE_DIR,
        report_to="wandb",
        # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        # save_strategy="steps",
        # eval_steps=200 if VAL_SET_SIZE > 0 else None,
        # save_steps=200,
        # save_total_limit=100,
        # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
    )

    model = conduct_training(dataset, val_dataset, model, tokenizer, training_arguments)
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    peft_model_id = f"sentiment-{cfg.MODEL_NAME}_{current_datetime}"
    peft_model_id = f"{cfg.TRAINED_MODEL}/{peft_model_id}"
    model.save_pretrained(peft_model_id)
    print("Saving model to: ", peft_model_id)


if __name__ == "__main__":
    main()
