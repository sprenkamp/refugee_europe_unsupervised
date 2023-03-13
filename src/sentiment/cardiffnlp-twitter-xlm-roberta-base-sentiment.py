import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from transformers import pipeline

tqdm.pandas()


class CardiffXLMRobertaModelBaseSentiment:
    def __init__(self, *args):
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base-sentiment')
        self.sentiment_task = pipeline("sentiment-analysis", model='cardiffnlp/twitter-xlm-roberta-base-sentiment', tokenizer=self.tokenizer, truncation=True, max_length=512)
        self.args = args[0]

    # probably not needed as we are using the pipeline and it checks for device
    # smileys look nice though
    def set_device(self) -> torch.device:
        if torch.cuda.is_available():
            print("🚀 Using CUDA GPU 🚀")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("🚀 Using MPS GPU 🚀")
            return torch.device("mps")
        else:
            print("🦥 Using CPU 🦥")
            return torch.device("cpu")

    def classify_sentiment(self, text):
        # # Tokenize the text
        # inputs = self.tokenizer(text, return_tensors="pt")
        # # Turn off gradient calculations
        # with torch.no_grad():
        #     # Make a prediction with the model
        #     outputs = self.model(**inputs)
        # # Get the predicted class (positive or neutral or negative)
        # _, predicted = torch.max(outputs.logits, dim=1
        # inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return self.sentiment_task(text)[0]['label']

    def run_telegram(self):
        df = pd.read_csv(self.args.input)
        df = df[df.messageText.str.len() > 100].sample(n=100000)
        df['sentiment'] = df['messageText'].progress_apply(self.classify_sentiment)
        df.to_csv(self.args.input.split('.')[0] + '_sentiment.csv')

    def run_twitter(self):
        df = pd.read_csv(self.args.input).drop_duplicates(subset=['text'])
        df['sentiment'] = df['text'].progress_apply(self.classify_sentiment)
        df.to_csv(self.args.input.split('.')[0] + '_sentiment.csv')

    def run_google_news(self):
        # Load the existing CSV file with news articles
        df = pd.read_csv(self.args.input)

        # Loop through the DataFrame and classify the sentiment of each article file
        sentiments = []
        for i, row in tqdm(df.iterrows()):
            filename = "{0}_{1}_{2}.txt".format(row["title"].replace("/", " "), row["alpha2_code"], row["language_code"])
            try:
                with open(os.path.dirname(self.args.input) + "/articleProcessed/DACH/"+ filename, "r") as f:
                    text = f.read()
                    sentiment = self.classify_sentiment(text)
                sentiments.append(sentiment)
            except FileNotFoundError:
                # If the file does not exist, add a "Not Found" sentiment
                sentiments.append("Not Found")

        # Add a new column with the predicted sentiments to the DataFrame
        df["sentiment"] = sentiments

        # Save the updated CSV file
        df.to_csv(self.args.input.split('.')[0] + '_sentiment.csv', index=False)

    def run(self):
        self.set_device()
        if self.args.data_type == "telegram":
            self.run_telegram()
        elif self.args.data_type == "twitter":
            self.run_twitter()
        elif self.args.data_type == "google_news":
            self.run_google_news()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CardiffBLP Twitter RoBERTa base emotion model.')
    parser.add_argument('--input', '-i', help='Input file or folder path')
    parser.add_argument('--output', '-o', help='Output file or folder path')
    parser.add_argument('-d', '--data_type', choices=['telegram', 'twitter', 'google_news', 'gdelt'], help='Choose a datasource', required=True)
    args = parser.parse_args()

    model = CardiffXLMRobertaModelBaseSentiment(args)
    model.run()
