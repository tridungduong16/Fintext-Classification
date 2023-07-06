import pdb

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import pipeline


@hydra.main(
    version_base=None,
    config_path="/home/duongd/WorkingDirectory/Fintext-Classification/src/configuration",
    config_name="general",
)
def main():
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)


if __name__ == "__main__":
    main()
