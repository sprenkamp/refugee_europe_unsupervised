from datetime import date, timedelta
from pygooglenews import GoogleNews
import pandas as pd
import time 
import requests
import concurrent.futures
import os
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()

countries = { #TODO: check all translations
"AT": {"de": ["Ukraine + Flüchtlinge", "Ukraine + flüchten", "Ukraine + Migranten", "Ukraine + migrieren", "Ukraine + Asyl"]}, #Austria
"DE": {"de": ["Ukraine + Flüchtlinge", "Ukraine + flüchten", "Ukraine + Migranten", "Ukraine + migrieren", "Ukraine + Asyl"]}, #Germany
"CH": {"de": ["Ukraine + Flüchtlinge", "Ukraine + flüchten", "Ukraine + Migranten", "Ukraine + migrieren", "Ukraine + Asyl"],
       "fr": ["Ukraine + réfugiés", "Ukraine + réfugiant", "Ukraine + migrants", "Ukraine + migrant", "Ukraine + asile"],
       "it": ["Ucraina + rifugiati", "Ucraina + rifugiato", "Ucraina + migranti", "Ucraina + migrante", "Ucraina + asilo"]}, #Switzerland
}




# Create a date range to loop over
start_date = date(2022, 2, 24)
end_date = date(2023, 2, 24)
# start_date = start_date.strftime('%a, %d %b %Y %H:%M:%S %Z')
# end_date = end_date.strftime('%a, %d %b %Y %H:%M:%S %Z')
delta = timedelta(days=1)
df_path = 'data/news/googleNews/googleNewsDACH.csv'

# def check_and_convert_date(date):
#     if isinstance(date, str):
#         date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z').date()
#     return date

# Define a function that will be executed in parallel
def process_search_terms(key_country, key_language, search_term, current_date, current_date_plus_one):
    # Try to execute the Google News search
    try:
       # Use GoogleNews library to search for news articles
       gn = GoogleNews(country=key_country, lang=key_language)
       search = gn.search(search_term, from_=str(current_date), to_=str(current_date_plus_one))
        
       # Create a dataframe from the search results
       df_current = pd.DataFrame(search['entries'])
       df_current['alpha2_code'] = key_country
       df_current['language_code'] = key_language
       df_current['search_term'] = search_term
       df_current["date"] = current_date
       return df_current
    
    # Handle connection error or timeout exceptions
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print("Error: Failed to establish a connection or request timed out")
        return None
#TODO check if none Pandas datastructure is quicker
# Main loop to execute multiple searches in parallel
def main_loop(start_date, end_date, delta, df_path):
       current_date = start_date
       # check if data file exists
       if os.path.isfile(df_path):
              df = pd.read_csv(df_path)
              # df = pd.DataFrame({'date': [pd.to_datetime('2022-02-28').date()]})
              df['date'] = pd.to_datetime(df['date'])
              current_date = df.date.max().date()
              current_date += delta
              print("loading existing data and start appending from {}".format(current_date))
       else:
              df = pd.DataFrame()
              print("creating new data file starting from {}".format(start_date))
       
       # Loop over each day in the date range
       while current_date <= end_date:
              current_date_plus_one = current_date + delta
        
              # Use a ThreadPoolExecutor to run multiple searches in parallel
              with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
                     # Generate a list of search terms to execute
                     search_terms = [(key_country, key_language, search_term, current_date, current_date_plus_one) 
                            for key_country in countries 
                            for key_language in countries[key_country] 
                            for search_term in countries[key_country][key_language]]
            
                     # Submit all of the search terms to the executor
                     results = [executor.submit(process_search_terms, *search_term) for search_term in search_terms]
            
                     # Wait for all of the search terms to complete and collect their results
                     for future in concurrent.futures.as_completed(results):
                            df_current = future.result()
                            if df_current is not None:
                                   df = pd.concat([df, df_current])
              #save data every month
              if pd.to_datetime(current_date).day == pd.to_datetime(current_date).days_in_month:
                     print("saving data till {0}, {1} articles found".format(current_date, len(df)))
                     df.drop_duplicates(["title", "published"], inplace=True)
                     print("saving data till {0}, {1} articles found Drop duplicates".format(current_date, len(df)))
                     df = df[df.date <= current_date + delta] #somehow the date is not always correct thus is capture more articles messing up loading of exist data
                     print("saving data till {0}, {1} articles found Drop not under date range".format(current_date, len(df)))
                     df.to_csv('data/news/googleNews/googleNews.csv', index=False)

              current_date += delta
    
       # Save the final results to a CSV file
       df.drop_duplicates(["title", "published"], inplace=True)
       print("Number of articles found in total:", len(df))
       df.to_csv('data/news/googleNews/googleNewsDACH.csv', index=False)

if __name__ == '__main__':
       start_time = time.time()
       main_loop(start_date, end_date, delta, df_path)
       end_time = time.time()
       
       print("Time taken:", (end_time - start_time)/60, "minutes")