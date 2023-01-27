#!/bin/bash
curl --request GET 'https://api.twitter.com/2/tweets/search/recent?query=from:twitterdev&max_results=100' --header "Authorization: Bearer $BEARER_TOKEN"

curl --request GET 'https://api.twitter.com/2/tweets/search/recent?query=place_country:CH&max_results=10' --header "Authorization: Bearer $BEARER_TOKEN"

# curl https://api.twitter.com/2/tweets/search/recent?query=cat%20has%3Amedia%20-grumpy&tweet.fields=created_at&max_results=100 -H "Authorization: Bearer $BEARER_TOKEN"

place_country: