import pdb

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


def prompt_generation(message, sentiment, inference=False):
    if inference:
        prompt = f"""
        ### Instruction: Classify the text into neutral, negative or positive. 
        #### Sentence: 
            Text: {message}
        ### Response:
        """
    else:
        prompt = f"""
        ### Instruction: Classify the text into neutral, negative or positive. 
        #### Sentence: 
            Text: {message}
        ### Response:
            Sentiment: {sentiment}
        """
    return prompt


@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def read_data(cfg: DictConfig):
    df = pd.read_csv(
        f"{cfg.RAW_DATA_PATH}/all-data.csv", delimiter=",", encoding="latin-1"
    )
    df = df.rename(
        columns={
            "neutral": "sentiment",
            "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .": "message",
        }
    )
    df["prompt"] = [
        prompt_generation(message, sentiment)
        for message, sentiment in zip(df["message"], df["sentiment"])
    ]
    df["prompt_inference"] = [
        prompt_generation(message, sentiment, inference=True)
        for message, sentiment in zip(df["message"], df["sentiment"])
    ]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=1)

    df_train.to_csv(f"{cfg.PROCESSED_DATA_PATH}/train.csv", index=False)
    df_val.to_csv(f"{cfg.PROCESSED_DATA_PATH}/validation.csv", index=False)
    df_test.to_csv(f"{cfg.PROCESSED_DATA_PATH}/test.csv", index=False)


if __name__ == "__main__":
    read_data()
