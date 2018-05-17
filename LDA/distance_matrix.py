import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial import distance
import requests
from elasticsearch import Elasticsearch

import pdb


def load_data_from_elastic():
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    res = es.search(index='testname2', doc_type='typetest', body=doc,scroll='1m')
    raw_docs = res['hits']['hits']
    raw_docs = raw_docs[107:] # remove Spanish articals 

    link_to_id = {}
    id_to_link = {}

    docs = []

    id = 0
    for doc in raw_docs:
        link_to_id[doc['_id']] = id
        id_to_link[id] = doc['_id']
        id += 1

        docs.append(doc['title'] + ' ' + doc['description'] + ' ' + doc['content'].replace('\n'))



    pdb.set_trace()
load_data_from_elastic()

def main():
    # --------------- Parameters ---------------
    # Number of topics
    n_topics = 20

    # Languge for stop words filter
    stop_words_language = 'english'

    # Max number of vocabularies
    max_words = 40000

    # Number of docuemtns
    n_docs = 2000

    # --------------- Loading data ---------------
    print('Loading data ...')

    datafile = './data/abcnews-date-text.csv'
    raw_data = pd.read_csv(datafile, parse_dates=[0], infer_datetime_format=True)
    # Headline before vectorization
    documents = raw_data['headline_text']
    documents.index = raw_data['publish_date']

    #documents = documents.as_matrix()
    documents = documents.sample(n=n_docs, random_state=0).as_matrix()

    # --------------- Transofrm to a term matrix ---------------
    print('Transorm documents to term a term matix ...')
    count_vectorizer = CountVectorizer(stop_words=stop_words_language, max_features=max_words)
    # Headline after vectorization
    documents_term_matrix = count_vectorizer.fit_transform(documents) 


    # --------------- LDA Model ---------------
    print('Fit LDA model ...')
    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=1)
    lda_topic_matrix = lda_model.fit_transform(documents_term_matrix)

    # --------------- Distance Matrix ---------------
    print('Caculate distance matrix ...')
    distance_matrix = distance.cdist(lda_topic_matrix, lda_topic_matrix, 'euclidean')

    # Add hight value for distances to its self
    distance_matrix[np.eye(n_docs) == 1] = 9999999

    # --------------- CSV File ---------------
    print('Writing CSV files ...')
    np.savetxt("lda_topic_matrix.csv", lda_topic_matrix, delimiter=",", encoding='utf-8')
    np.savetxt("lda_distance_matrix.csv", distance_matrix, delimiter=",", encoding='utf-8')


    # --------------- LSA Model ---------------
    # lsa_model = TruncatedSVD(n_components=n_topics)
    # lsa_topic_matrix = lsa_model.fit_transform(documents_term_matrix)

