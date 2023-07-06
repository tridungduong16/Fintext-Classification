import os
import time
from threading import Event, Thread

import torch
import transformers
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (LlamaForCausalLM, LlamaTokenizer, StoppingCriteria,
                          StoppingCriteriaList)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import datetime
import pdb
import re

import bitsandbytes as bnb
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig,
                          LlamaForCausalLM, LlamaTokenizer, StoppingCriteria,
                          StoppingCriteriaList, pipeline)
from tqdm import tqdm 
DEVICE = torch.device("cuda:0")


def extract_sentiment(response_text):
    match = re.search(r"### Response:\s*([\w\s]+)", response_text)
    if match:
        sentiment = match.group(1)
    else:
        sentiment = None
    return sentiment

def find_sentiment(response_text):
    sentiment_pattern = r"\b(?:positive|negative|neutral)\b"
    sentiment_matches = re.findall(sentiment_pattern, sentence, re.IGNORECASE)
    sentiment_matches_lower = [match.lower() for match in sentiment_matches]
    return sentiment_matches_lower

def inference(trained_model, generated_tokenizer, input_text, generation_config):
    trained_model.eval()
    encoding = generated_tokenizer(input_text, return_tensors="pt").to(
        trained_model.device
    )
    with torch.inference_mode():
        outputs = trained_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            generation_config=generation_config,
        )
    generated_text = generated_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def main(cfg: DictConfig):
    if cfg.MODEL_NAME == "xgen-7b-8k-open-instruct":
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
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
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
        max_memory = {i: "46000MB" for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            load_in_4bit=True,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            cache_dir=cfg.CACHE_DIR,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_ID, cache_dir=cfg.CACHE_DIR,
        )

    model = PeftModel.from_pretrained(model, cfg.PEFT_MODEL_ID_PATH)
    model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    stop = StopOnTokens()
    generation_config = model.generation_config
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 200
    generation_config.use_cache = False
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.temperature = 0.9
    generation_config.stopping_criteria = StoppingCriteriaList([stop])

    model.eval()

    df_train = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/train.csv")
    df_val = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/validation.csv")
    df_test = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/test.csv")

    predicted_train = []
    for _, value in tqdm(df_train.iterrows()):
        input_text = value["prompt_inference"]
        response_text = inference(model, tokenizer, input_text, generation_config)
        res = extract_sentiment(response_text)
        predicted_train.append(res)
    df_train[f"raw-prediction"] = predicted_train
    

    predicted_val = []
    for _, value in tqdm(df_val.iterrows()):
        input_text = value["prompt_inference"]
        response_text = inference(model, tokenizer, input_text, generation_config)
        res = extract_sentiment(response_text)
        predicted_val.append(res)
    df_val[f"raw-prediction"] = predicted_val

    predicted_test = []
    for _, value in tqdm(df_test.iterrows()):
        input_text = value["prompt_inference"]
        response_text = inference(model, tokenizer, input_text, generation_config)
        res = extract_sentiment(response_text)
        predicted_test.append(res)
    df_test[f"raw-prediction"] = predicted_test

    df_train.to_csv(f"{cfg.RESULTS}/{cfg.MODEL_NAME}_train.csv", index=False)
    df_val.to_csv(f"{cfg.RESULTS}/{cfg.MODEL_NAME}_validation.csv", index=False)
    df_test.to_csv(f"{cfg.RESULTS}/{cfg.MODEL_NAME}_test.csv", index=False)

if __name__ == "__main__":
    main()
