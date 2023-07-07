# FinancialPhraseBank Sentiment Analysis
This project focuses on sentiment analysis of financial news headlines using the FinancialPhraseBank dataset. The dataset provides insights into the sentiments expressed in financial news from the perspective of a retail investor. The sentiment labels can be categorized as negative, neutral, or positive.

## Dataset Description
The dataset consists of two columns:

Sentiment: This column represents the sentiment associated with each financial news headline. The sentiment can be negative, neutral, or positive.
News Headline: This column contains the text of the financial news headlines.
Acknowledgements
We would like to acknowledge the following publication for providing the FinancialPhraseBank dataset:

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). "Good debt or bad debt: Detecting semantic orientations in economic texts." Journal of the Association for Information Science and Technology, 65(4), 782-796.

## Project Objective
The objective of this project is to leverage the power of large language models to perform sentiment analysis on financial news headlines. By utilizing advanced natural language processing techniques, we aim to analyze the sentiments expressed in the dataset and gain insights into the sentiments prevalent in financial news from a retail investor's perspective.

## Methodology
We will employ state-of-the-art techniques using large language models to conduct sentiment analysis on the FinancialPhraseBank dataset. By utilizing pre-trained models and advanced natural language processing libraries, we will process the textual data and classify the sentiments associated with each news headline. The process will involve data preprocessing, model training, and sentiment prediction.

## Repository Structure
The repository contains the following files:

* README.md: Provides an overview and instructions for the project.
* dataset.csv: The FinancialPhraseBank dataset in CSV format.
* sentiment_analysis.ipynb: Jupyter Notebook containing the code for sentiment analysis.
* requirements.txt: A text file listing the required libraries and dependencies.
Getting Started
To get started with this project, follow these steps:

Clone this repository: git clone https://github.com/your-username/financial-sentiment-analysis.git
Navigate to the project directory: cd financial-sentiment-analysis
- Install the required dependencies: pip install -r requirements.txt
- Go to sr/configuration and change the config to your path data and model
- Run classification with FinBERT pretrained: 
```
python -m src.classification_finbert
```
- Run classification with FinBERT pretrained + Classifier such as GradientBoosting or LightGBM: 
```
python -m src.classification_finbert
```
- Run classification with LLMs model: 
```
python -m src.classifier_llm
```
- Run fine-tune LLMs model for sentiment analysis task
```
python -m src.finetune_model
```

- Feel free to explore and modify the code according to your requirements.

# Conclusion
Based on my findings, FinBERT outperforms large language models in terms of both inference speed and accuracy when applied to sentiment classification tasks. Moreover, incorporating gradient boosting classifiers with FinBERT can further enhance the overall performance and effectiveness of the system.






For any questions or suggestions, please feel free to reach out to me.

