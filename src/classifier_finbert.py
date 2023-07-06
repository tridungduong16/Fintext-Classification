import pdb

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import pipeline


@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def predict_sentiment_with_pipeline(cfg: DictConfig):
    generated = pipeline("text-classification", model="ProsusAI/finbert")
    df_train = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/train.csv")
    df_val = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/validation.csv")
    df_test = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/test.csv")

    df_train["finbert-prediction"] = [
        generated(value["message"])[0]["label"]
        for _, value in tqdm(df_train.iterrows())
    ]
    df_val["finbert-prediction"] = [
        generated(value["message"])[0]["label"] for _, value in tqdm(df_val.iterrows())
    ]
    df_test["finbert-prediction"] = [
        generated(value["message"])[0]["label"] for _, value in tqdm(df_test.iterrows())
    ]

    df_train.to_csv(f"{cfg.RESULTS}/finbert_train.csv", index=False)
    df_val.to_csv(f"{cfg.RESULTS}/finbert_validation.csv", index=False)
    df_test.to_csv(f"{cfg.RESULTS}/finbert_test.csv", index=False)


if __name__ == "__main__":
    predict_sentiment_with_pipeline()
