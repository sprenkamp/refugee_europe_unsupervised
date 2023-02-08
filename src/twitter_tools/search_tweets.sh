#!/bin/bash
query_file=$1
out_file=$2
queries=$(jq 'keys[]' "$query_file")
total_tweet_count=0
for query in $queries; do
  query_value=$(jq -r ".$query" "$query_file")
  out_file_path="$out_file$query.json"
  if [ ! -f "$out_file_path" ]; then
    echo "Scrapping data from $query"
    python src/twitter_tools/search_tweets.py \
      --credential-file ~/.twitter_keys.yaml \
      --credential-file-key search_tweets_v2 \
      --start-time 2022-01-24 \
      --query "$query_value" \
      --filename-prefix $out_file$query \
      --no-print-stream \
      --tweet-fields "created_at" \
      --results-per-call 100 \
      # --results-per-file 10
  else
    echo "Skipping $out_file_path, file already exists."
  fi
done 

python src/twitter_tools/create_tweets_csv.py --query_file $query_file 
# for query in $queries; do
#   out_file_path="$out_file$query.json"
#   tweet_count=$(jq '.data[] | .tweet_count' "$out_file_path" | awk '{sum+=$1} END {print sum}')
#   echo "tweets found for $query: $tweet_count"
#   total_tweet_count=$(($total_tweet_count + $tweet_count))
# done

# echo "tweets found in total: $total_tweet_count"


# python src/twitter_tools/search_tweets.py \
#   --credential-file ~/.twitter_keys.yaml \
#   --start-time 2021-01-24 \
#   --query query_value \
#   --tweet-fields "created_at" \
#   --filename-prefix data/twitter/EU/EU2 \
#   --no-print-stream
#   # --max-tweets 1000 \
#     # --max-tweets 1000 \


