#!/bin/bash
python src/twitter_tools/search_tweets.py \
  --credential-file ~/.twitter_keys.yaml \
  --start-time 2021-01-24 \
  --query '''
            (Ukraine OR Ukrainian) 
            (refugee OR refugees OR migration OR migrants OR migrant) 
            (Austria OR Belgium OR Bulgaria OR Croatia OR Cyprus OR Czech Republic OR Denmark 
            OR Estonia OR Finland OR France OR Germany OR Greece OR Hungary OR Ireland OR Italy 
            OR Latvia OR Lithuania OR Luxembourg OR Malta OR Netherlands OR Poland OR Portugal 
            OR Romania OR Slovakia OR Slovenia OR Spain OR Sweden OR Switzerland OR Norwegia OR 
            (United Kingdom) OR Liechtenstein OR Iceland OR Moldova) 
            ''' \
  --tweet-fields "created_at" \
  --filename-prefix data/twitter/EU/EU2 \
  --no-print-stream
  # --max-tweets 1000 \
    # --max-tweets 1000 \


