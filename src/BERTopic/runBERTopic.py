import os
import argparse
from bertopic import BERTopic
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
import random
import re
import tqdm
from sklearn.cluster import KMeans
nltk.download('stopwords')

#TODO remove country from data sources
#TODO find stopwords list for bg, cs, et, hu, lv, lt, mt, sk, sl, is
#define stopwords, languages needed: de, nl, fr, bg, hr, el, cs, da, et, fi, fr, hu, en, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, no, is, ro, uk, ru

# with open("data/stopwords/stopwords.txt") as file: #none finalised stopwords loaded from .txt file
#     stopwords = [line.rstrip() for line in file]

# with open("data/stopwords/country_stopwords.txt") as file: #load list of countries
#     country_stopwords = [line.rstrip() for line in file]

#define stopwords
stopWords = stopwords.words('english') 
for word in stopwords.words('german'):
    stopWords.append(word)
for word in stopwords.words('french'):
    stopWords.append(word)
for word in stopwords.words('italian'):
    stopWords.append(word)
for word in stopwords.words('russian'):
    stopWords.append(word)
with open("data/stopwords/stopwords_ua.txt") as file: #add ukrainian stopwords loaded from .txt file
    ukrstopWords = [line.rstrip() for line in file]
for stopwords in ukrstopWords:
    stopWords.append(stopwords)


def validate_path(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

class BERTopicAnalysis:
    """
    The following Class trains a BERTopic model on a given input file 
    and saves the model and visualizations in the given output folder.
    If the model already exists, meaning the output_folder already contains a BERTopic model,
    it can be loaded and solely used for inference.

    Parameters for the class:
    input: path to the input file
    output_folder: path to the output folder
    k_cluster: number of clusters to be used for the model
    do_inference: boolean to indicate if the model should also be used for inference, 
                predicting the class of each text line in the input file
    data_type: type of data to be used for training the model,
    cmul_gpu: boolean to indicate if a GPU is available for training the model
    """

    # initialize class
    def __init__(self, input_data, data_type, output_folder, k_cluster, do_inference, cmul_gpu):
        self.input_data = input_data
        self.data_type = data_type
        self.output_folder = output_folder
        self.k_cluster = k_cluster
        self.do_inference = do_inference
        self.cmul_gpu = cmul_gpu


    # read data telegram and prepare data for BERTopic
    def load_data_telegram(self):
        self.df = pd.read_csv(self.input_data)
        self.df.dropna(subset=['messageText'],inplace=True)
        self.df.drop_duplicates(subset=['messageText'], keep='first',inplace=True)
        self.df = self.df[self.df['messageText'].map(type) == str]
        self.df["messageText"] = self.df['messageText'].str.split().str.join(' ')
        lines = self.df[(self.df['messageText'].str.len() >= 100) & (self.df['messageText'].str.len() <= 2500)].messageText.values
        self.text_to_analyse_list = [line.rstrip() for line in lines]
        print("using {} telegram messages for topic model".format(len(self.text_to_analyse_list)))
        #TODO: investigate chunk size rest 2 shouldn't happen

    # read data twitter and prepare data for BERTopic
    def load_data_twitter(self):
        #TODO: specific processing needed for twitter data?
        self.df = pd.read_csv(self.input_data)
        self.df = self.df[self.df['text'].map(type) == str]
        self.df.drop_duplicates(subset=['text'], inplace=True)
        # self.df = self.df.sample(n=5000)
        lines = self.df['text'].values
        self.text_to_analyse_list = [line.rstrip() for line in lines]
        # self.counter_country = 0
        # self.text_to_analyse_list = [self.remove_countries(utterance, country_stopwords) for utterance in self.text_to_analyse_list]
        # print(self.counter_country)
        print("using {} twitter posts for topic model".format(len(self.text_to_analyse_list)))

    # read data google news and prepare data for BERTopic
    def load_data_google_news(self):
        self.text_to_analyse_list = []
        self.file_list = os.listdir(self.input_data)
        random.shuffle(self.file_list)
        self.file_list_final = []
        self.countries_list = []
        for file in self.file_list:
            if file.endswith(".txt"):
                with open(os.path.join(self.input_data, file), "r") as f:
                    article = f.read() #each line should be one paragraph
                    # article = self.remove_countries(article, country_stopwords)
                    self.text_to_analyse_list.append(article)
                    self.file_list_final.append(file)
                    if "_DE_" in file:
                        self.countries_list.append("DE")
                    elif "_CH_" in file:
                        self.countries_list.append("CH")
        print("using {} google news articles for topic model".format(len(self.text_to_analyse_list)))
    
    def load_data_google_news_headlines(self):
        self.df = pd.read_csv(self.input_data)
        self.df = self.df[self.df['title'].map(type) == str]
        self.df.drop_duplicates(subset=['title'], inplace=True)
        lines = self.df['title'].values
        self.text_to_analyse_list = [line.rstrip().split(" - ")[0].replace(" ...", "") for line in lines]
        print("using {} google news headlines articles for topic model".format(len(self.text_to_analyse_list)))


    # read data from gdelt and prepare data for BERTopic
    def load_data_gdelt(self):
        #TODO: implement gdelt data loading
        print("gdelt data loading not implemented yet")

    # remove countries from data
    def remove_countries(self, text, country_stopwords):
        for country in country_stopwords:
            if country in text:
                #self.counter_country += 1
                text = re.sub(country, '', text) #check if "COUNTRY" is better as replacement
                text = text.replace("  ", " ")
        return text


    # load potentially existing model
    def read_model(self):
        print("loading model")
        self.model=BERTopic.load(f"{self.output_folder}/BERTopicmodel")

    # check if k_cluster is numeric and convert to int if so.
    # this is necessary for BERTopic if the cluster number is given as a string, 
    # as BERTopic can be set to automatically determine the number of clusters using the string "auto".
    def k_cluster_type(self):
        if self.k_cluster.isnumeric():
            self.k_cluster = int(self.k_cluster)


    def split_list(self, lst, num_chunks):
        random.shuffle(lst) # shuffle list to avoid bias in splitting
        chunk_size = len(lst) // num_chunks
        chunks = []
        for i in range(1, len(lst), chunk_size):
            chunks.append(lst[i:i + chunk_size])
        last_chunk_size = len(lst) % num_chunks
        if last_chunk_size != 0:
            last_chunk = lst[-last_chunk_size:]
            chunk_counter = 0
            for remainder_last_chunk in last_chunk:
                chunks[chunk_counter].append(remainder_last_chunk)
                chunk_counter += 1
        return chunks

    # train BERTopic model we use basic parameters for the model, 
    # using basic umap_model and hdbscan_model,
    # as defined in the BERTopic documentation
    def fit_BERTopic(self):
        if self.cmul_gpu:
            print('GPU available, using GPU')
            from cuml.cluster import HDBSCAN #for GPU
            from cuml.manifold import UMAP  #for GPU
        else:
            print('No GPU available, using CPU')
            from umap import UMAP 
            from hdbscan import HDBSCAN
        #define hyperparameter
        min_cluster_size=50
        if self.data_type=="twitter":
            min_cluster_size=75
        umap_model = UMAP(n_neighbors=25, n_components=10, metric='cosine', low_memory=False, random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopWords) #define vectorizer model with stopwords
        # hdbscan_model = KMeans(n_clusters=25)
        self.model = BERTopic(verbose=True,
                              #embedding_model="paraphrase-MiniLM-L6-v2",
                              language="multilingual",
                              nr_topics=self.k_cluster, 
                              vectorizer_model=vectorizer_model,
                              umap_model=umap_model,
                              hdbscan_model=hdbscan_model,
                              )
        topics, probs = self.model.fit_transform(self.text_to_analyse_list)
        # if self.data_type=="twitter":
        #     # Reduce outliers
        #     topics = self.model.reduce_outliers(self.text_to_analyse_list, topics, strategy="embeddings")
        #     self.model.update_topics(self.text_to_analyse_list, topics=topics)

    # save model and visualizations
    def save_results(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        fig = self.model.visualize_topics()
        fig.write_html(f"{self.output_folder}/bert_topic_model_distance_model.html")
        fig = self.model.visualize_hierarchy()
        fig.write_html(f"{self.output_folder}/bert_topic_model_hierarchical_clustering.html")
        fig = self.model.visualize_barchart(top_n_topics=30)
        fig.write_html(f"{self.output_folder}/bert_topic_model_word_scores.html")
        fig = self.model.visualize_heatmap()
        fig.write_html(f"{self.output_folder}/bert_topic_model_word_heatmap.html")
        self.model.save(f"{self.output_folder}/BERTopicmodel")
    
    # save representative documents for each topic
    def write_representative_docs_df(self):
        writer = pd.ExcelWriter(f"{self.output_folder}/representative_docs.xlsx", engine='xlsxwriter')
        for i in self.model.get_representative_docs().keys():
            df = pd.DataFrame(self.model.get_representative_docs()[i], columns=['message'])
            df.to_excel(writer, sheet_name=self.model.get_topic_info()[self.model.get_topic_info()['Topic']==i]['Name'].values[0][:31])
        writer.save()
        self.model.get_topic_info().to_csv(f"{self.output_folder}/topic_info.csv")

    # predict the class of each text line in the input file
    def inference(self):
        if self.data_type == "telegram":
            pred, prob = self.model.transform(self.df['messageText'].values)
            self.df["pred"] = pred
        elif self.data_type == "twitter":
            pred, prob = self.model.transform(self.df['text'].values)
            self.df["pred"] = pred
        elif self.data_type == "google_news":
            pred, prob = self.model.transform(self.text_to_analyse_list)
            self.df["pred"] = pred
        elif self.data_type == "gdelt":
            #TODO: implement gdelt inference
            print("gdelt inference not implemented yet")
        
        self.df.to_csv(f"{self.output_folder}/df_model.csv", index=False)

    # run all functions
    def run_all(self):
        # load data depending on data type
        if self.data_type == "telegram":
            self.load_data_telegram()
        elif self.data_type == "twitter":
            self.load_data_twitter()
        elif self.data_type == "google_news":
            self.load_data_google_news_headlines()
        elif self.data_type == "gdelt":
            self.load_data_gdelt()
        # check if model already exists
        if os.path.exists(f"{self.output_folder}/BERTopicmodel"):
            self.read_model()
            self.write_representative_docs_df()
        # if not, train model and save results
        else:
            self.k_cluster_type()
            self.fit_BERTopic()
            self.save_results()
            self.write_representative_docs_df()
        #do inference if specified
        if self.do_inference:
            self.inference()

def main():
    # define parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_data', help="Specify the input file or folder", type=validate_path, required=True) 
    parser.add_argument('-d', '--data_type', choices=['telegram', 'twitter', 'google_news', 'gdelt'], help='Choose a datasource', required=True)
    parser.add_argument('-o', '--output_folder', help="Specify folder for results", required=True)
    parser.add_argument('-k', '--k_cluster', help="number of topic cluster", required=False, default="auto")
    parser.add_argument('-di', '--do_inference', help="does inference on data", action='store_true' , default=False)
    parser.add_argument('-cuml_gpu', help="use cmul on GPU", action='store_true' , default=False)
    args = parser.parse_args()
    if args.data_type=="twitter" or args.data_type=="google_news":
        with open("data/stopwords/country_stopwords.txt") as file: #load list of countries
            country_stopwords = [line.rstrip() for line in file]
            for country_stopword in  country_stopwords:
                stopWords.append(country_stopword)
    BERTopic_Analysis = BERTopicAnalysis(args.input_data,
                                         args.data_type,
                                         args.output_folder,
                                         args.k_cluster,
                                         args.do_inference,
                                         args.cuml_gpu
                                         )
    # run all functions
    BERTopic_Analysis.run_all()

if __name__ == '__main__':
    main()