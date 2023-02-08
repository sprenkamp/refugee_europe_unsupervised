import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import concurrent.futures
import os

df = pd.read_csv("data/news/googleNews/googleNews.csv")

# Convert the list of existing files to a set for faster lookup
written_filenames = set(os.listdir("data/news/googleNews/article/"))
start_time = time.time()

def process_url(col, values):
    print(col)
    values.title = values.title.replace("/", " ")
    filepath = "data/news/googleNews/article/{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    filename = "{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    if filename not in written_filenames:  # Check if the filename is in the set
        try:
            url = values.link
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            p_tags = soup.find_all('p')
            text = "\n".join(p.get_text() for p in p_tags)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print("Error: Failed to establish a connection or request timed out")
        with open(filepath, "w") as f:
            f.write(text)
            
            written_filenames.add(filepath)  # Add the filename to the set after writing to disk
            if col % 1000 ==0:
                end_time = time.time()
                print(col, (end_time - start_time)/60)
                
            # print("Time taken for article ", col, "     ", values.title, ": " ,end_time - start_time)
    

with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
    filenames = [executor.submit(process_url, col, values) for col, values in df[["title", "link", "alpha2_code", "language_code"]].iterrows()]



