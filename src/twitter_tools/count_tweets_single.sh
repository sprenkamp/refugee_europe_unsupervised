#!/bin/bash
file=$1
python src/twitter_tools/search_tweets.py \
  --credential-file ~/.twitter_keys.yaml \
  --credential-file-key count_tweets_v2 \
  --start-time 2022-01-24 \
  --query '''
          (Ukraine OR Ukrainian OR Ukrainer) (refugee OR refugees OR migration OR migrants OR migrant OR asylum OR Flüchtling OR flüchten OR Migrant OR migrieren OR Asyl) (Austria OR Österreich)
          '''\
  --filename-prefix "$file" \
  --no-print-stream

jq '.data[] | .tweet_count' "$file.json" | awk '{sum+=$1} END {print sum}'

# '''
#             (Ukraine OR Ukrainian) 
#             (refugee OR refugees OR migration OR migrants OR migrant OR asylum)
#             (Austria)
#             '''
#"((Ukraine OR Ukrainian OR Ukrainer) (Flüchtling OR flüchten OR Migrant OR migrieren OR Asyl) (Austria OR Österreich)) OR ((Ukraine OR Ukrainian OR Ukrainer)(refugee OR refugees OR migration OR migrants OR migrant OR asylum OR Flüchtling OR flüchten OR Migrant OR migrieren OR Asyl) (place_country:AT))"