import os
import argparse
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
import random
import re
import tqdm

#TODO remove country from data sources
#TODO find stopwords list for bg, cs, et, hu, lv, lt, mt, sk, sl, is
#define stopwords, languages needed: de, nl, fr, bg, hr, el, cs, da, et, fi, fr, hu, en, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, no, is, ro, uk, ru

with open("data/stopwords/stopwords.txt") as file: #none finalised stopwords loaded from .txt file
    stopwords = [line.rstrip() for line in file]

with open("data/stopwords/country_stopwords.txt") as file: #load list of countries
    country_stopwords = [line.rstrip() for line in file]

#TODO: add twitter stopwords
# with open("data/stopwords/twitter_stopwords.txt") as file: #load list of countries
#     twitter_stopwords = [line.rstrip() for line in file]

# stopwords = stopwords.words('english') 
# for word in stopwords.words('german'):
#     stopwords.append(word)
# for word in stopwords.words('russian'):
#     stopwords.append(word)
# with open("data/stopwords/stopwords_ua.txt") as file: #add ukrainian stopwords loaded from .txt file
#     ukrstopwords = [line.rstrip() for line in file]
# for stopwords in ukrstopwords:
#     stopwords.append(stopwords)



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
    def __init__(self, input, data_type, output_folder, k_cluster, do_inference, cmul_gpu):
        self.input = input
        self.data_type = data_type
        self.output_folder = output_folder
        self.k_cluster = k_cluster
        self.do_inference = do_inference
        self.cmul_gpu = cmul_gpu

    # read data telegram and prepare data for BERTopic
    def load_data_telegram(self):
        self.df = pd.read_csv(self.input)
        self.df = self.df[self.df['messageText'].map(type) == str]
        self.df["messageText"] = self.df['messageText'].str.split().str.join(' ')
        lines = self.df[self.df['messageText'].str.len() >= 100].messageText.values
        self.text_to_analyse_list = [line.rstrip() for line in lines]
        print("using {} telegram messages for topic model".format(len(self.text_to_analyse_list)))
        #TODO: investigate chunk size rest 2 shouldn't happen

    # read data twitter and prepare data for BERTopic
    def load_data_twitter(self):
        #TODO: specific processing needed for twitter data?
        self.df = pd.read_csv(self.input)
        self.df = self.df[self.df['text'].map(type) == str]
        self.df.drop_duplicates(subset=['text'], inplace=True)
        self.df["text"] = self.df['text'].apply(lambda x: re.sub(r"http\S+", "", x))
        self.df = self.df.sample(n=5000)
        lines = self.df['text'].values
        self.text_to_analyse_list = [line.rstrip() for line in lines]
        # self.counter_country = 0
        self.text_to_analyse_list = [self.remove_countries(utterance, country_stopwords) for utterance in self.text_to_analyse_list]
        # print(self.counter_country)
        print("using {} twitter posts for topic model".format(len(self.text_to_analyse_list)))

    # read data google news and prepare data for BERTopic
    def load_data_google_news(self):
        self.text_to_analyse_list = []
        for file in os.listdir(self.input):
            if file.endswith(".txt"):
                with open(os.path.join(self.input, file), "r") as f:
                    lines = f.readlines() #each line should be one paragraph
                    self.text_to_analyse_list.extend([line.rstrip() for line in lines])
    
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
        #TODO: change sentence transformer/ embedding model for news data  mBERT or XLM-RoBERTa
        chunk_max_size = 220000
        if len(self.text_to_analyse_list) <= chunk_max_size:
            umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
            hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords) #define vectorizer model with stopwords
            self.model = BERTopic(verbose=True,
                                #embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens",
                                language="multilingual",
                                nr_topics=self.k_cluster, 
                                vectorizer_model=vectorizer_model,
                                umap_model=umap_model,
                                hdbscan_model=hdbscan_model,
                                )
            topics, probs = self.model.fit_transform(self.text_to_analyse_list)
        else:
            print("too much data using online Topic Modeling") #Only the most recent batch of documents is tracked. If you want to be using online topic modeling for low-memory use cases, then it is advised to also update the .topics_ attribute. Otherwise, variations such as hierarchical topic modeling will not work.
            #text_to_analyse_list_chunks = self.split_list(self.text_to_analyse_list, (len(self.text_to_analyse_list)//chunk_max_size)+1)
            random.shuffle(self.text_to_analyse_list)
            text_to_analyse_list_chunks = [self.text_to_analyse_list[i:i+chunk_max_size] for i in range(0, len(self.text_to_analyse_list), chunk_max_size)] #check if this is better for splliting, as mine creates an error
            # umap_model = IncrementalPCA() #IncrementalPCA(n_components=5)
            # cluster_model = MiniBatchKMeans(n_clusters=self.k_cluster, random_state=0)
            # vectorizer_model = OnlineCountVectorizer(stop_words=stopwords) #OnlineCountVectorizer(stopwords=stopwords, decay=.01)
            # self.model = BERTopic(verbose=True,
            #                     #embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens",
            #                     language="multilingual",
            #                     nr_topics=self.k_cluster, 
            #                     vectorizer_model=vectorizer_model,
            #                     umap_model=umap_model,
            #                     hdbscan_model=cluster_model,
            #                     )
            umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
            hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords) #define vectorizer model with stopwords
            self.model = BERTopic(verbose=True,
                                #embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens",
                                language="multilingual",
                                nr_topics=self.k_cluster, 
                                vectorizer_model=vectorizer_model,
                                umap_model=umap_model,
                                hdbscan_model=hdbscan_model,
                                )
            for count, text_to_analyse_list_chunk in enumerate(text_to_analyse_list_chunks, start=1):
                print("running chunk {} from {}".format(count, len(text_to_analyse_list_chunks)))
                print("chunk size: {}".format(len(text_to_analyse_list_chunk)))
                self.model.fit_transform(text_to_analyse_list_chunk) 
                # self.model.partial_fit(text_to_analyse_list_chunk) TODO: check if this is better, unsuitable as no -1 category?
            
                # topics, probs = self.model.fit_transform(text_to_analyse_list_chunk)

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
    def write_multi_sheet_excel(self):
        writer = pd.ExcelWriter(f"{self.output_folder}/representative_docs.xlsx", engine='xlsxwriter')
        for i in self.model.get_representative_docs().keys(): #TODO: check topics for online model
            df = pd.DataFrame(self.model.get_representative_docs()[i], columns=['message'])
            df.to_excel(writer, sheet_name=self.model.get_topic_info()[self.model.get_topic_info()['Topic']==i]['Name'].values[0][:31])
        writer.save()
        self.model.get_topic_info().to_csv(f"{self.output_folder}/topic_info.csv")

    # predict the class of each text line in the input file
    def inference(self):
        if self.data_type == "telegram":
            pred, prob = self.model.transform(self.df['messageText'].values)
        elif self.data_type == "twitter":
            pred, prob = self.model.transform(self.text_to_analyse_list)
        elif self.data_type == "google_news":
            #TODO: implement google news inference
            print("google news inference not implemented yet")
        elif self.data_type == "gdelt":
            #TODO: implement gdelt inference
            print("gdelt inference not implemented yet")
        self.df['cluster'] = pred
        self.df.to_csv(f"{self.output_folder}/df_model.csv", index=False)

    # run all functions
    def run_all(self):
        # load data depending on data type
        if self.data_type == "telegram":
            self.load_data_telegram()
        elif self.data_type == "twitter":
            self.load_data_twitter()
        elif self.data_type == "google_news":
            self.load_data_google_news()
        elif self.data_type == "gdelt":
            self.load_data_gdelt()
        # check if model already exists
        if os.path.exists(f"{self.output_folder}/BERTopicmodel"):
            self.read_model()
        # if not, train model and save results
        else:
            self.k_cluster_type()
            self.fit_BERTopic()
            self.save_results()
            # self.write_multi_sheet_excel()
        #do inference if specified
        if self.do_inference:
            self.inference()

def main():
    # define parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Specify the input file or folder", type=validate_path, required=True) 
    parser.add_argument('-d', '--data_type', choices=['telegram', 'twitter', 'google_news', 'gdelt'], help='Choose a datasource', required=True)
    parser.add_argument('-o', '--output_folder', help="Specify folder for results", required=True)
    parser.add_argument('-k', '--k_cluster', help="number of topic cluster", required=False, default="auto")
    parser.add_argument('-di', '--do_inference', help="does inference on data", action='store_true' , default=False)
    parser.add_argument('-cuml_gpu', help="use cmul on GPU", action='store_true' , default=False)
    args = parser.parse_args()
    # initialize class
    BERTopic_Analysis = BERTopicAnalysis(args.input,
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