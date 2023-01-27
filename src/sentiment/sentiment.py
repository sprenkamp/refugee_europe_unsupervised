import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
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
        self.tokenizer_twitter_xlm_roberta_base_sentiment = XLMRobertaTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        self.twitter_xlm_roberta_base_sentiment = XLMRobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        self.tokenizer_twitter_roberta_base_emotion = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.twitter_roberta_base_emotion = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

    def set_device(self):
        # Set the device to run on (e.g. "cuda" for GPU, "cpu" for CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Use MPS to accelerate training and inference (if a GPU is available)
        if self.device == "cuda":
            torch.distributed.launch(
                main_fn=lambda: self.twitter_xlm_roberta_base_sentiment.to(self.device),
                args=(),
                backend="mpi",
                # Set the number of GPUs to use
                world_size=1,
            )
        # Set twitter_xlm_roberta_base_sentiment to device
        self.twitter_xlm_roberta_base_sentiment = self.twitter_xlm_roberta_base_sentiment.to(self.device)


    def load_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.df.cluster!=-1]

    def base_sentiment(self, text: str) -> str:
        # Encode the input text
        input_ids = torch.tensor(self.tokenizer_twitter_xlm_roberta_base_sentiment.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        # try ecxeption block to handle the RuntimeError
        try:
            # Get the predicted sentiment
            with torch.no_grad():
                outputs = self.twitter_xlm_roberta_base_sentiment(input_ids)
                sentiment = torch.argmax(outputs[0]).item()
        except (RuntimeError, IndexError) as e:
            return "Error"

        # Return the predicted sentiment as a string
        if sentiment == 0:
            return "Negative"
        elif sentiment == 1:
            return "Neutral"
        elif sentiment == 2:
            return "Positive"
    
    def base_emotion(self, text: str) -> str:
        # Encode the input text
        input_ids = torch.tensor(self.tokenizer_twitter_roberta_base_emotion.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        # try ecxeption block to handle the RuntimeError
        try:
            # Get the predicted sentiment
            with torch.no_grad():
                outputs = self.twitter_roberta_base_emotion(input_ids)
                sentiment = torch.argmax(outputs[0]).item()
            if sentiment==0:
                return "anger"
            elif sentiment==1:
                return "joy"
            elif sentiment==2:
                return "optimism"
            elif sentiment==3:
                return "sadness"
        except (RuntimeError, IndexError) as e:
            return "Error"

    def inference(self):
        print("Inference Base Sentiment Model")        
        self.df['sentiment_name_twitter_xlm_roberta_base_sentiment'] = self.df['messageText'].progress_apply(lambda x: self.base_sentiment(x))
        print("Inference Base Emotion Model")        
        self.df['sentiment_name_twitter_roberta_base_emotion'] = self.df['messageText'].progress_apply(lambda x: self.base_emotion(x))
        self.df.to_csv(self.input_file.split(".")[0] + "_sentimet.csv", index=False)

    def run_all(self):
        self.set_device()
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