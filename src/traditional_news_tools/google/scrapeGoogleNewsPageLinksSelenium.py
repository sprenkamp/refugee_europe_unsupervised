from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import concurrent.futures
import os

df = pd.read_csv("data/news/googleNews/googleNewsDACH.csv")[:5]

# Convert the list of existing files to a set for faster lookup
written_filenames = set(os.listdir("data/news/googleNews/article/DACH/"))
start_time = time.time()

def process_url(col, values):
    values.title = values.title.replace("/", " ")
    filepath = "data/news/googleNews/article/DACH/{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    filename = "{0}_{1}_{2}.txt".format(values.title, values.alpha2_code, values.language_code)
    if filename not in written_filenames:  # Check if the filename is in the set
        try:
            url = values.link
            print(url)
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_driver_path = '/usr/bin/chromedriver'  # replace with your actual path
            driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)
            driver.get(url)
            
            # Find the "Accept all" button and click on it
            accept_button = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/form[2]/input[10]')
            accept_button.click()
            
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))
            p_tags = driver.find_elements(By.TAG_NAME, "p")
            text = "\n".join(p.text for p in p_tags)
            print(text)
            driver.quit()
        except Exception as e:
            print("Error: ", e)
            return
        with open(filepath, "w") as f:
            f.write(text)
            written_filenames.add(filepath)  # Add the filename to the set after writing to disk
            if col % 1000 ==0:
                end_time = time.time()
                print(col, (end_time - start_time)/60)
                
            # print("Time taken for article ", col, "     ", values.title, ": " ,end_time - start_time)
    

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    filenames = [executor.submit(process_url, col, values) for col, values in df[["title", "link", "alpha2_code", "language_code"]].iterrows()]
