import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import concurrent.futures

df = pd.read_csv("data/news/googleNews/googleNews.csv")


def process_url(col, values):
    start_time = time.time()
    try:
        # print(values.link)
        url = values.link
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        p_tags = soup.find_all('p')
        text = "\n".join(p.get_text() for p in p_tags)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # handle the connection error and timeout
        print("Error: Failed to establish a connection or request timed out")
        return
    # text = text.replace("\n", " ")
    values.title = values.title.replace("/", " ")
    filename = "data/news/googleNews/article/{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    with open(filename, "w") as f:
        f.write(text)
        end_time = time.time()
        print("Time taken for article ", col, "     ", values.title, ": " ,end_time - start_time)
    return filename

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    filenames = [executor.submit(process_url, col, values) for col, values in df[["title", "link", "alpha2_code", "language_code"]].iterrows()]
