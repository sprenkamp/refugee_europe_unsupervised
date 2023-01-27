import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.read_csv("data/news/googleNews/googleNews.csv")


for col, values in df[["title", "link", "alpha2_code", "language_code"]].iterrows():
    try:
        print(values.link)
        url = values.link
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        p_tags = soup.find_all('p')
        text = "\n".join(p.get_text() for p in p_tags)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # handle the connection error and timeout
        print("Error: Failed to establish a connection or request timed out")
        continue
    # text = text.replace("\n", " ")
    values.title = values.title.replace("/", " ")
    with open("data/news/googleNews/article/{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code), "w") as f:
        f.write(text)