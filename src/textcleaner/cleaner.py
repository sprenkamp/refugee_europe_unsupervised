import argparse
import os
from tqdm import tqdm
import justext
import lxml.etree as etree
import re
import pandas as pd
from textblob import TextBlob
import preprocessor as p
from autocorrect import Speller
import emoji

tqdm.pandas()


def validate_path(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


class Cleaner:
    def __init__(self, args):
        self.args = args
        self.spell = Speller(lang='en') #we are using autocorrect to correct spelling mistakes most tweets are in english
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY) #we are removing urls, emojis, smileys and numbers, RTs and hashtags and mentions are removed seperately


    def clean_multilingual_text(self, text):
        try:
            stoplist = [justext.get_stoplist(lang) for lang in ['English', 'German', 'French', 'Italian']]
            stoplist = frozenset().union(*stoplist)
            paragraphs = justext.justext(text, stoplist=stoplist)
            cleaned_text = "\n".join([p.text for p in paragraphs])
        except etree.ParserError:
            cleaned_text = text
        return cleaned_text
    
    def drop_sequence_without_word(self, text):
        text_lst = text.split()
        max_length = 5
        if any(len(item) > max_length for item in text_lst):
            return text
        else:
            return None
        
    def blob(self, text):
            blob = TextBlob(text)
            return str(blob.correct())
    
    def remove_emojis(self,text):
        # Convert emojis to textual representation
        text = emoji.demojize(text)
        # Remove emojis and other non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove whitespace characters
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    
    def clean_news(self):
        files = os.listdir(self.args.input)

        for file in tqdm(files):
            with open(self.args.input+file, "r") as f:
                try: 
                    full_article = f.read()
                except UnicodeDecodeError:
                    continue
                #TODO: better check list
                check_list = ["ukraine", "Ucraina", "ukrainer", "ukrainerin", "ukrainien", "ukrainienne", "ukrainian", "flüchtling", "réfugié", "fugitif", "rifugiato","flüchten", "fuggire", "fuir", "migrant", "migrieren", "migrer", "migrare", "asyl", "asilo", "asile", "asylum"] 
                if any(item.lower() in full_article.lower() for item in check_list):
                    f.seek(0)
                    lines = f.readlines()  # read all lines into a list
                    # print(lines)
                    final_lines = []
                    #TODO: better check list
                    check_list_not_wanted = ["Abo", "subscribe", "Abonnement", "Abonnieren", "Mail", "Kundenbefragung"]
                    for line in lines:
                            if len(line.split()) <= 10:
                                continue
                            elif line.startswith(" ") or line.startswith("	"):
                                continue
                            elif line.startswith("Copyright") or line.startswith("Follow") or line.startswith("©"):
                                continue
                            elif any(item.lower() in line.lower() for item in check_list_not_wanted):
                                continue
                            else:
                                final_lines.append(line)
                    
                    paragraph = ''.join(final_lines)
                    url_pattern = re.compile(r"https?://\S+") # define a regular expression pattern for URLs
                    paragraph = re.sub(url_pattern, " ", paragraph) # remove any occurrence of URLs in the string
                    if len(paragraph.split())>=10:
                        try:
                            with open(self.args.output+file, "w") as w:
                                w.write(paragraph)
                        except OSError:
                            continue
        print("from {} articles, {} were cleaned and written to processed".format(len(files), len(os.listdir(self.args.output))))
        csv_file = "data/news/googleNews/googleNewsDACH.csv"

        # Get a list of all the files in the article folder
        article_files = os.listdir(self.args.output)

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Create a new column in the DataFrame to check if the corresponding file exists
        df["file_exists"] = df.progress_apply(lambda row: "{0}_{1}_{2}.txt".format(row["title"].replace("/", " "), row["alpha2_code"], row["language_code"]) in article_files, axis=1)

        # Drop rows where the file does not exist
        df.drop(df[~df["file_exists"]].index, inplace=True)

        # Drop the "file_exists" column
        df.drop("file_exists", axis=1, inplace=True)

        # Save the updated CSV file
        df.to_csv(csv_file.split(".")[0]+"_clean.text", index=False)
        print("saved csv file to {}".format(csv_file.split(".")[0]+"_clean.csv"))
                        

            
    def clean_telegram(self):
        df = pd.read_csv(self.args.input)
        old_len = len(df)
        # remove URLs
        df.dropna(subset=["messageText"], inplace=True)
        url_pattern = re.compile(r"https?://\S+") # define a regular expression pattern for URLs
        # apply the pattern to the DataFrame
        df["messageText"] = df["messageText"].progress_apply(lambda x: re.sub(url_pattern, " ", x))
        # Remove whitespace characters
        df["messageText"] = df["messageText"].progress_apply(lambda x: re.sub(r'\s+', ' ', x))
        # remmove messages with no word longer than 5 characters
        df["messageText"] = df["messageText"].progress_apply(self.drop_sequence_without_word)
        df.dropna(subset=["messageText"], inplace=True)
        print("from {} messagges, {} were cleaned and written to processed".format(old_len, len(df)))

        df.to_csv(self.args.output, index=False)

    def clean_twitter(self):
        df = pd.read_csv(self.args.input)
        old_len = len(df)
        # remove nan and duplicates and empty messages
        df.dropna(subset=["text"], inplace=True)
        df.drop_duplicates(subset=["text"], inplace=True)
        df = df[df["text"] != ""]
        # remove # and @
        df["text"] = df["text"].progress_apply(lambda x: re.sub(r'[@#]', " ", x))
        # remove RT
        df["text"] = df["text"].progress_apply(lambda x: re.sub(r'RT', "", x))
        # remove URLs
        url_pattern = re.compile(r"https?://\S+") # define a regular expression pattern for URLs
        df["text"] = df["text"].progress_apply(lambda x: re.sub(url_pattern, " ", x))
        # Remove whitespace characters
        df["text"] = df["text"].progress_apply(lambda x: re.sub(r'\s+', ' ', x))
        # remmove messages with no word longer than 5 characters
        df["text"] = df["text"].progress_apply(self.drop_sequence_without_word)
        # convert to string
        df["text"] = df["text"].astype(str)
        # df['text'] = df['text'].progress_apply(p.clean)
        # remove emojis
        # df["text"] = df["text"].progress_apply(lambda x: emoji.get_emoji_regexp().sub(u'', x))
        df["text"] = df["text"].progress_apply(self.remove_emojis)
        df.dropna(subset=["text"], inplace=True)
        df["text"] = df["text"].progress_apply(lambda x: x.strip())
        print("from {} messagges, {} were cleaned and written to processed".format(old_len, len(df)))

        df.to_csv(self.args.output, index=False)

    def run_all(self):
        if self.args.data_type == "telegram":
            self.clean_telegram()
        elif self.args.data_type == "twitter":
            self.clean_twitter()
        elif self.args.data_type == "google_news":
            self.clean_news()
def main():
    # define parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Specify the input file or folder", type=validate_path, required=True) 
    parser.add_argument('-d', '--data_type', choices=['telegram', 'twitter', 'google_news', 'gdelt'], help='Choose a datasource', required=True)
    parser.add_argument('-o', '--output', help="Specify output file or folder", required=True)
    args = parser.parse_args()
    # initialize class
    Cleaner_ = Cleaner(args)
    # run all functions
    Cleaner_.run_all()

if __name__ == '__main__':
    main()