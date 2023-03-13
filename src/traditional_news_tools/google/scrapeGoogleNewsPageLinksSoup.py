import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import concurrent.futures
import os
import time

df = pd.read_csv("data/news/googleNews/googleNewsDACH.csv")

# Convert the list of existing files to a set for faster lookup
written_filenames = set(os.listdir("data/news/googleNews/article/DACH/"))
start_time = time.time()

def process_url(col, values):
    values.title = values.title.replace("/", " ")
    filepath = "data/news/googleNews/article/DACH/{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    filename = "{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    if filename not in written_filenames:  # Check if the filename is in the set
        try:
            initial_url = values.link
            # print("initial" , initial_url)
            response = requests.get(initial_url, allow_redirects=False)

            # Extract the URL that the response redirects to
            if response.status_code == 301 or response.status_code == 302:
                final_url = response.headers.get('Location')
                # print("why not?")
            else:
                final_url = None
                # print("why???")
            if final_url is not None:
                # print("final",final_url)
                response = requests.get(final_url)
                soup = BeautifulSoup(response.content, "html.parser")
                p_tags = soup.find_all('p')
                text = "\n".join(p.get_text() for p in p_tags)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            with open(filepath, "w") as f:
                f.write("ERROR")
                written_filenames.add(filepath) 
            return None    
        if text.startswith("..."):
                f.write("ERROR")
                written_filenames.add(filepath)
                return None
            # print("Error: Failed to establish a connection or request timed out")
        if text is not None:
            with open(filepath, "w") as f:
                f.write(text)
                written_filenames.add(filepath)  # Add the filename to the set after writing to disk
                if col % 1000 ==0:
                    end_time = time.time()
                    print("articles scrapped", col, "/", len(df), "in", (end_time - start_time)/60, "minutes")
                    
                # print("Time taken for article ", col, "     ", values.title, ": " ,end_time - start_time)
    

with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
    filenames = [executor.submit(process_url, col, values) for col, values in df[["title", "link", "alpha2_code", "language_code"]].iterrows()]

print("total scrapped files: {0}/{1}".format(len(os.listdir("data/news/googleNews/article/DACH/")),len(df)))


