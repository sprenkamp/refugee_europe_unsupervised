import argparse
import os
import pandas as pd
import re
import string
import torch
from tqdm import tqdm
tqdm.pandas()
from transformers import pipeline



class DavlanBertBaseMultilingualCasedNERHRLModel:
    def __init__(self, *args):
        self.ner_task = pipeline("ner", model='Davlan/bert-base-multilingual-cased-ner-hrl', tokenizer='Davlan/bert-base-multilingual-cased-ner-hrl')
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
        
    def remove_smileys(self, text):
        # Define regular expression for detecting smileys
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

        # Remove all smileys from the text
        return re.sub(emoji_pattern, '', text)
    
    def remove_punctuation(self, text):
        # Remove all punctuation from the text
        return text.translate(str.maketrans('', '', string.punctuation))

    def classify_ner(self, text):
        return self.ner_task(text)

    def run_telegram(self):
        df = pd.read_csv(self.args.input)
        df['ner'] = df['messageText'].progress_apply(self.classify_ner)
        df.to_csv(self.args.input.split('.')[0] + '_NER.csv')

    def run_twitter(self):
        df = pd.read_csv(self.args.input).drop_duplicates(subset=['text'])[:100]
        df['text'] = df['text'].apply(self.remove_smileys) # remove smileys
        df['text'] = df['text'].apply(self.remove_punctuation) # remove punctuation
        df['text'] = df['text'].apply(lambda x: x.replace('#', '')) # remove hashtags
        df['text'] = df['text'].apply(lambda x: x.replace('@', '')) # remove mentions
        df['ner'] = df['text'].progress_apply(self.classify_ner)
        df.to_csv(self.args.input.split('.')[0] + '_NER.csv')

    def run_google_news(self):
        if not os.path.exists(self.args.output):
            os.makedirs(self.args.output)
        for filename in os.listdir(self.args.input):
            with open(os.path.join(self.args.input, filename), 'r') as f:
                text = f.read()
            ner = self.classify_ner(text)
            output_filename = os.path.join(self.args.output, filename)
            with open(output_filename, 'w') as f:
                f.write(str(ner))
    
    def run(self):
        self.set_device()
        if self.args.data_type == "telegram":
            self.run_telegram()
        elif self.args.data_type == "twitter":
            self.run_twitter()
        elif self.args.data_type == "google_news":
            self.run_google_news()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Davlan Bert base multilingual cased NER HRL model.')
    parser.add_argument('--input', '-i', help='Input file or folder path')
    parser.add_argument('--output', '-o', help='Output file or folder path')
    parser.add_argument('-d', '--data_type', choices=['telegram', 'twitter', 'google_news', 'gdelt'], help='Choose a datasource', required=True)
    args = parser.parse_args()

    model = DavlanBertBaseMultilingualCasedNERHRLModel(args)
    model.run()
