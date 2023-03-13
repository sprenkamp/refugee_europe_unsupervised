import argparse
import os
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
tqdm.pandas()


class CardiffBLPModel:
    def __init__(self, *args):
        self.model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        self.tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')
        self.labels = ["anger", "joy", "optimism", "sadness"]
        self.args = args[0]

    def classify_emotions(self, text):
        encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_text)
            probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze(0)
            return probs.tolist()

    def run_telegram(self):
        df = pd.read_csv(self.args.input)[:1000]
        df['emotions'] = df['messageText'].progress_apply(self.classify_emotions)
        df.to_csv(self.args.output)

    def run_twitter(self):
        df = pd.read_csv(self.args.input)[:1000]
        df['emotions'] = df['text'].progress_apply(self.classify_emotions)
        df.to_csv(self.args.output)

    def run_google_news(self):
        if not os.path.exists(self.args.output):
            os.makedirs(self.args.output)
        for filename in os.listdir(self.args.input):
            with open(os.path.join(self.args.input, filename), 'r') as f:
                text = f.read()
            emotions = self.classify_emotions(text)
            output_filename = os.path.join(self.args.output, filename)
            with open(output_filename, 'w') as f:
                f.write(str(emotions))
    
    def run(self):
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

    model = CardiffBLPModel(args)
    model.run()
