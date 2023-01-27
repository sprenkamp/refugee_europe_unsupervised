from datetime import date, timedelta
from pygooglenews import GoogleNews
import pandas as pd

countries = {
    "AT": {"de": "Ukraine + Flüchtlinge"}, #Austria
    "BE": {"nl": "Ukraine + Vluchtelingen", "fr": "Ukraine + réfugiés"}, #Belgium
    "BG": {"bg": "Украйна + бежанци"}, #Bulgaria
    "HR": {"hr": "Ukrajina + izbjeglice"}, #Croatia
    "CY": {"el": "Ουκρανία + πρόσφυγες"}, #Cyprus
    "CZ": {"cs": "Ukrajina + uprchlíci"}, #Czechia
    "DK": {"da": "Ukraine + flygtninge"}, #Denmark
    "EE": {"et": "Ukraina + põgenikud"}, #Estonia
    "FI": {"fi": "Ukraina + pakolaiset"}, #Finland
    "FR": {"fr": "Ukraine + réfugiés"}, #France
    "DE": {"de": "Ukraine + Flüchtlinge"}, #Germany
    "GR": {"el": "Ουκρανία + πρόσφυγες"}, #Greece
    "HU": {"hu": "Ukrajna + menekültek"}, #Hungary
    "IE": {"en": "Ukraine + refugees"}, #Ireland
    "IT": {"it": "Ucraina + rifugiati"}, #Italy
    "LV": {"lv": "Ukraina + bēgļi"}, #Latvia
    "LT": {"lt": "Ukraina + pabėgėliai"}, #Lithuania
    "LU": {"fr": "Ukraine + Flüchtlinge", "de": "Ukraine + Flüchtlinge"}, #Luxembourg
    "NL": {"nl": "Ukraine + vluchtelingen"}, #Netherlands
    "PL": {"pl": "Ukraina + uchodźcy"}, #Poland
    "PT": {"pt": "Ucrânia + refugiados"}, #Portugal
    "RO": {"ro": "Ucraina + refugiați"}, #Romania
    "SK": {"sk": "Ukrajina + utečenci"}, #Slovakia
    "SI": {"sl": "Ukrajina + begunci"}, #Slovenia
    "ES": {"es": "Ucrania + refugiados"}, #Spain
    "SE": {"sv": "Ukraine + flyktingar"}, #Sweden
    "CH": {"de": "Ukraine + Flüchtlinge", "fr": "Ukraine + réfugiés", "it": "Ucraina + rifugiati"}, #Switzerland
    "NO": {"no": "Ukraine + flyktninger"}, #Norway
    "GB": {"en": "Ukraine + refugees"}, #United Kingdom
    "LI": {"de": "Ukraine + Flüchtlinge"}, # Liechtenstein
    "IS": {"is": "Úkraína + flóttamenn"}, # Iceland
    "MD": {"ro": "Ucraina + refugiați"}, # Moldova
    "UA": {"uk": "Україна + біженці"} # Ukraine
}

# Create a date range to loop over
start_date = date(2022, 2, 24)
end_date = date(2023, 1, 25)
delta = timedelta(days=1)

# Create an empty dataframe to store the results
df = pd.DataFrame(columns=['title', 'title_detail', 'links', 'link', 'id', 'guidislink',
       'published', 'published_parsed', 'summary', 'summary_detail', 'source',
       'sub_articles', 'alpha2_country_code' 'language_code'])

# Loop over each day in the date range
current_date = start_date
while current_date <= end_date:
       for key_country in countries:
              for key_language in countries[key_country]:
                     print(current_date, key_country, key_language, countries[key_country][key_language])
                     gn = GoogleNews(country = key_country, lang = key_language)
                     current_date_plus_one = current_date + delta
                     search = gn.search(countries[key_country][key_language], from_=str(current_date), to_=str(current_date_plus_one))
                     df_current = pd.DataFrame(search['entries'])
                     df_current['alpha2_code'] = key_country
                     df_current['language_code'] = key_language
                     df = pd.concat([df, df_current])
                     current_date += delta
df.to_csv('data/news/googleNews/googleNews.csv', index=False)