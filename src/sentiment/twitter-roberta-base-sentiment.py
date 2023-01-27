from scipy.special import softmax
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os
from tqdm import tqdm
tqdm.pandas()


def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

class SentimentClassifier:
    def __init__(self, input_file):
        self.input_file = input_file
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") 
        self.labels = ['Negative', 'Neutral', 'Positive']

    def load_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.df.cluster!=-1].sample(n=1000)

    def analyse_sentiment(self, x):
        if len(x)>512:
            x_split_list = []
            n = 400
            for i in range(0, len(x), n):
                x_split_list.append(x[i:i+n])
            scores_split_list = []    
            for x_split in x_split_list:
                try:
                    encoded_tweet = self.tokenizer(x_split, return_tensors='pt')
                except (RuntimeError, IndexError) as e:
                    return "Error"
                try:
                    output = self.model(**encoded_tweet)
                except (RuntimeError, IndexError) as e:
                    return "Error"
                try:
                    scores_split = output[0][0].detach().numpy()
                except (RuntimeError, IndexError) as e:
                    return "Error"
                scores_split = softmax(scores_split)
                scores_split_list.append(scores_split)
            return np.mean(np.array(scores_split_list), axis=0) #averaged scores
        else:
            try:
                encoded_tweet = self.tokenizer(x, return_tensors='pt')
            except (RuntimeError, IndexError) as e:
                return "Error"
            try:
                output = self.model(**encoded_tweet)
            except (RuntimeError, IndexError) as e:
                return "Error"
            try:
                scores = output[0][0].detach().numpy()
            except (RuntimeError, IndexError) as e:
                return "Error"
            scores = softmax(scores)
            return scores

    def scores_to_sentiment(self, x):
        if not x=="Error":
            max_value = max(x.tolist())
            max_index = x.tolist().index(max_value)
            return self.labels[max_index]
        else:
            return "Error"

    
    def inference(self):
        self.df['sentiment'] = self.df['messageText'].progress_apply(lambda x: self.analyse_sentiment(x))
        self.df['sentiment_name'] = self.df['sentiment'].progress_apply(lambda x: self.scores_to_sentiment(x))
        self.df.to_csv(self.input_file.split(".")[0] + "_sentimet.csv", index=False)

    def run_all(self):
        self.load_data()
        self.inference()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="Specify the input file", type=validate_file, required=True) 
    args = parser.parse_args()
    Sentiment_Classifier = SentimentClassifier(args.input_file)
    Sentiment_Classifier.run_all()

if __name__ == '__main__':
    main()