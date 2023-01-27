#!/bin/bash
python src/twitter_tools/search_tweets.py \
  --credential-file ~/.twitter_keys.yaml \
  --credential-file-key count_tweets_v2 \
  --start-time 2022-01-24 \
  --query '''
             ((#Ukraine OR #Ukrainian) 
              (#refugee OR #refugees OR #migration OR #migrants OR #migrant OR #flüchtlinge) 
              (place_country:RO OR place_country:SI OR place_country:FR OR place_country:DK OR 
               place_country:CZ OR place_country:CH OR place_country:PL OR place_country:BE OR 
               place_country:MD OR place_country:CY OR place_country:ES OR place_country:AT OR 
               place_country:DE)
             ) OR 
             (
              (#Ukraine OR #Ukrainian) 
              (#refugee OR #refugees OR #migration OR #migrants OR #migrant OR #flüchtlinge) 
              (#Romania OR #Slovenia OR #France OR #Denmark OR #Czechia OR #Switzerland OR #Poland 
               OR #Belgium OR #Moldova OR #Cyprus OR #Espana OR #Austria OR #Germany)
             ) 
          ''' \