import pdb

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import pipeline
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import hydra 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer
from omegaconf import DictConfig, OmegaConf
import pdb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb


def generate_feature(tokenized, model):
    max_len = max(tokenized.apply(len))
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    chunk_size = 5000  # Define your desired chunk size
    chunks = [tokenized[i:i+chunk_size] for i in range(0, len(tokenized), chunk_size)]

    features_list = []

    for chunk in tqdm(chunks):
        padded = np.array([i + [0] * (max_len - len(i)) for i in chunk.values])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad():
            features = model(input_ids, attention_mask=attention_mask)[0][:, 0, :].numpy()
        features_list.append(features)

    features = np.concatenate(features_list, axis=0)
    return features

METHOD = "LightGBM"

@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def predict_sentiment_with_pipeline(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")

    df_train = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/train.csv")
    df_val = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/validation.csv")
    df_test = pd.read_csv(f"{cfg.PROCESSED_DATA_PATH}/test.csv")

    label_encoder = LabelEncoder()
    df_train['decoded_sentiment'] = label_encoder.fit_transform(df_train['sentiment'])  # Convert labels to numerical values
    df_val['decoded_sentiment'] = label_encoder.transform(df_val['sentiment'])  # Convert labels to numerical values
    df_test['decoded_sentiment'] = label_encoder.fit_transform(df_test['sentiment'])  # Convert labels to numerical values


    tokenized_train = df_train['message'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    tokenized_test = df_val['message'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    tokenized_val = df_test['message'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))


    feature_train = generate_feature(tokenized_train, model)
    feature_test = generate_feature(tokenized_test, model)
    feature_val = generate_feature(tokenized_val, model)

    y_train = df_train["decoded_sentiment"].values.reshape(-1,1)
    y_test = df_train["decoded_sentiment"].values.reshape(-1,1)
    y_val = df_train["decoded_sentiment"].values.reshape(-1,1)

    if METHOD=="LightGBM":
        train_data = lgb.Dataset(feature_train, label=y_train)

        params = {
            'objective': 'binary',  # Specify the objective for binary classification
            'metric': 'binary_logloss'  # Evaluation metric
        }

        model = lgb.train(params, train_data)

        y_pred_train = model.predict(feature_train)
        y_pred_test = model.predict(feature_test)
        y_pred_val = model.predict(feature_val)

        y_pred_train = [1 if pred >= 0.5 else 0 for pred in y_pred_train]
        y_pred_test = [1 if pred >= 0.5 else 0 for pred in y_pred_test]
        y_pred_val = [1 if pred >= 0.5 else 0 for pred in y_pred_val]
    else:
        lr = GradientBoostingClassifier()
        lr.fit(feature_train, y_train)

        y_pred_train = lr.predict(feature_train)
        y_pred_test = lr.predict(feature_test)
        y_pred_val = lr.predict(feature_val)

    df_train['prediction'] = y_pred_train
    df_val['prediction'] = y_pred_test
    df_test['prediction'] = y_pred_val
    
    df_train['decoded_prediction'] = label_encoder.inverse_transform(df_train['prediction'])
    df_val['decoded_prediction'] = label_encoder.inverse_transform(df_val['prediction'])
    df_test['decoded_prediction'] = label_encoder.inverse_transform(df_test['prediction'])

    df_train.to_csv(f"{cfg.RESULTS}/finbert_gbm_train.csv", index=False)
    df_val.to_csv(f"{cfg.RESULTS}/finbert_gbm_validation.csv", index=False)
    df_test.to_csv(f"{cfg.RESULTS}/finbert_gbm_test.csv", index=False)


if __name__ == "__main__":
    predict_sentiment_with_pipeline()
