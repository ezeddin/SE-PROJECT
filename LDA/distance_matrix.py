import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial import distance
import requests
from elasticsearch import Elasticsearch

import pdb




def load_data_from_elastic(stop_words_language, max_words):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    query = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    # --------------- Loading data ---------------
    print('Loading data ...')

    res = es.search(index='testname2', doc_type='typetest', body=query,scroll='1m')
    raw_docs = res['hits']['hits']
    raw_docs = raw_docs[107:] # remove Spanish articals 

    
    link_to_id = {}
    id_to_link = {}

    content_docs = []
    id = 0
    for doc in raw_docs:
        link_to_id[doc['_id']] = id
        id_to_link[id] = doc['_id']
        id += 1
        keys = ['title', 'description', 'content']

        content = ''
        for key in keys:
            if key in doc['_source']:
                content +=  str(doc['_source'][key]).replace('\n', ' ').replace('\t', ' ').replace('[', '').replace(']', '')

        content_docs.append(content)
    # --------------- Transofrm to a term matrix ---------------
    print('Transorm documents to term a term matix ...')
    count_vectorizer = CountVectorizer(stop_words=stop_words_language, max_features=max_words, analyzer='word')
    # Docuemnts after vectorization
    docs_term_matrix = count_vectorizer.fit_transform(content_docs) 

    n_docs = id

    return raw_docs, docs_term_matrix, link_to_id, id_to_link, n_docs

def get_distance_matrix(docs_term_matrix, n_topics, n_docs):
    # --------------- LDA Model ---------------
    print('Fit LDA model ...')
    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=1)
    lda_topic_matrix = lda_model.fit_transform(docs_term_matrix)

    # --------------- LSA Model ---------------
    # lsa_model = TruncatedSVD(n_components=n_topics)
    # lsa_topic_matrix = lsa_model.fit_transform(docs_term_matrix)

    # --------------- Distance Matrix ---------------
    print('Caculate distance matrix ...')
    distance_matrix = distance.cdist(lda_topic_matrix, lda_topic_matrix, 'euclidean')

    # Add hight value for distances to its self
    distance_matrix[np.eye(n_docs) == 1] = 9999999

    return distance_matrix
    


    

# --------------- Parameters ---------------
# Number of topics
n_topics = 10

# Languge for stop words filter
stop_words_language = 'english'

# Max number of vocabularies
max_words = 40000

raw_docs, docs_term_matrix, link_to_id, id_to_link, n_docs = load_data_from_elastic(stop_words_language, max_words)
distance_matrix = get_distance_matrix(docs_term_matrix, n_topics, n_docs)

pdb.set_trace()

def recommend_articles(chosen_doc_id):
    related_doc_ids = get_closest_docs(chosen_doc_id)
    chosen_doc = raw_docs[chosen_doc_id]
    print("Chosen document: "+chosen_doc['title'])
    print("Description: "+doc['description']+"\n")

    for i in range(len(related_doc_ids)):
        related_doc = raw_docs[related_doc_ids[i]]
        print("Related document number "+str(i+1)+":")
        print("Title: "+related_doc['title'])
        print("Description: "+related_doc['description']+"\n")

    return eval(input("Please pick a document by typing its number: "))

def main():
    start_doc = random.randint(0,len(raw_docs)-1)
    new_doc = recommend_articles(start_doc)
    print(new_doc)