import os
from tqdm import tqdm
#TODO: remove every line less than 50 or 100 characters
#      remove every line not starting with any character
#      remove every line that is not main text pattern 
files = os.listdir("data/news/googleNews/article/")

for file in tqdm(files):
    with open("data/news/googleNews/article/"+file, "r") as f:
        try: 
            full_article = f.read()
        except UnicodeDecodeError:
            continue
        check_list = ["flüchtling", "flüchten", "migrant", "migrieren" , "asyl", "vluchteling", "vluchten", "migreren" , "asiel" , 'réfugié', 'réfugiant', 'asile', 'бежанци', 'бежи', 'мигранти', 'мигрират', 'асил', 'izbjeglice', 'izbjegavajući', 'migranti', 'migrirajući', 'azil', 'πρόσφυγες', 'πρόσφυγα', 'μετανάστες', 'μεταναστεύοντας', 'ασύλο', 'uprchlíci', 'uprchající', 'migrace', 'azyl', 'flygtninge', 'flygtede', 'migrere', 'asyl', 'põgenikud', 'põgenenud', 'migrant', 'migreerima', 'varjupaik', 'pakolaiset', 'pakenevat', 'siirtolaiset', 'siirtolaisten', 'turvapaikka', 'πρόσφυγες', 'πρόσφυγα', 'μετανάστες', 'μεταναστεύοντας', 'ασύλο', 'menekültek', 'migránsok', 'migráns', 'menekült', 'rifugiati', 'rifugiato', 'migranti', 'migrante', 'asilo', 'izglītības', 'izglītības', 'migranti', 'migrēt', 'azils', 'išvykusių', 'migrantai', 'migracijos', 'azilas', 'refuġjati', 'ħarba', 'migranti', 'jemigraw', 'ażil', 'uchodźcy', 'uciekać', 'migrantów', 'migracja', 'azyl', 'refugiados', 'refugiado', 'migrantes', 'migrante', 'asilo', 'refugiați', 'refugiat', 'migranți', 'uprchlíci', 'uprchajúci', 'migrácia', 'azyl', 'begunci', 'begunec', 'migracija', 'refugiados', 'refugiado', 'migrantes', 'asilo', 'flyktingar', 'flykting', 'migrera', 'flyktninger', 'flyktet', 'migranter', 'migrere', 'refugees', 'escape', 'migrants', 'migrate', 'asylum', 'біженці', 'біженець', 'мігранти', 'міграція', 'азіл']
        if any(item in full_article for item in check_list):
            f.seek(0)
            lines = f.readlines()  # read all lines into a list
            with open("data/news/googleNews/processedArticle/"+file, "w") as w:
                for line in lines:
                    if len(line.split()) <= 5:
                        continue
                    elif line.startswith(" ") or line.startswith("	"):
                        continue
                    elif line.startswith("Copyright") or line.startswith("Follow") or line.startswith("©"):
                        break
                    else:
                        w.writelines(line)

                
                    