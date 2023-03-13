import argparse
import json
import csv
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--query_file', type=str, help='The query file in JSON format')
args = parser.parse_args()

with open(args.query_file) as f:
    query_data = json.load(f)

header = ["query", "id", "text", "edit_history_tweet_ids", "created_at"]
tweets = []
for query, query_value in query_data.items():
    path = 'data/twitter/results/tweets_per_country/DACH/"' + query + '".json'
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        for item in json.loads(line)['data']:
            tweets.append([query, item['id'], item['text'], item['edit_history_tweet_ids'], item['created_at']])

with open("data/twitter/results/tweetsDACH.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(tweets)

df = pd.read_csv("data/twitter/results/tweetsDACH.csv")
print("Number of tweets: ", len(df))
print("Number of unique tweets: ", len(df['text'].unique()))
