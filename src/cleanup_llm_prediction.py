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
    df_train = pd.read_csv(f"{cfg.MODEL_NAME}_train.csv")
    df_val = pd.read_csv(f"{cfg.MODEL_NAME}_validation.csv")
    df_test = pd.read_csv(f"{cfg.MODEL_NAME}/test.csv")
    df_train[f"raw-prediction"] = [x for x in df_train[f"raw-prediction"]]
    df_val[f"raw-prediction"] = [x for x in df_val[f"raw-prediction"]]
    df_test[f"raw-prediction"] = [x for x in df_test[f"raw-prediction"]]
