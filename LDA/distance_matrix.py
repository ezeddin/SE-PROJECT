import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial import distance
from elasticsearch import Elasticsearch
import random
import argparse


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
        keys = ['title', 'description', 'content']

        content = ''
        for key in keys:
            if key in doc['_source']:
                content +=  str(doc['_source'][key]).replace('\n', ' ').replace('\t', ' ').replace('[', '').replace(']', '')

        content_docs.append(content)

        id += 1
    # --------------- Transofrm to a term matrix ---------------
    print('Transorm documents to term a term matix ...')
    count_vectorizer = CountVectorizer(stop_words=stop_words_language, max_features=max_words, analyzer='word')
    # Docuemnts after vectorization
    docs_term_matrix = count_vectorizer.fit_transform(content_docs) 

    n_docs = id

    return raw_docs, docs_term_matrix, link_to_id, id_to_link, n_docs

def get_distance_matrix(docs_term_matrix, n_topics, n_docs, model):

    topic_matrix = None
    if 'LDA' == model:
        # --------------- LDA Model ---------------
        print('Fit LDA model ...')
        lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=1)
        topic_matrix = lda_model.fit_transform(docs_term_matrix)

    # --------------- LSA Model ---------------
    elif 'LSA' == model:
        print('Fit LSA model ...')
        lsa_model = TruncatedSVD(n_components=n_topics, random_state=0)
        topic_matrix = lsa_model.fit_transform(docs_term_matrix)

    else:
        print('ERROR: Wrong model')
        exit()

    # --------------- Distance Matrix ---------------
    print('Caculate distance matrix ...')
    distance_matrix = distance.cdist(topic_matrix, topic_matrix, 'euclidean')

    # Add hight value for distances to its self
    distance_matrix[np.eye(n_docs) == 1] = 9999999

    return distance_matrix
    

def recommend_articles(chosen_doc_id, top, raw_docs, distance_matrix):
    related_doc_ids = distance_matrix[chosen_doc_id].argsort()[:top]
    chosen_doc = raw_docs[chosen_doc_id]
    print("Chosen document: "+chosen_doc['_source']['title'])
    if "description" in chosen_doc['_source']:
        print("Description: "+chosen_doc['_source']['description'])
    print("Link: "+chosen_doc['_id'])
    print()

    for i in range(len(related_doc_ids)):
        related_doc = raw_docs[related_doc_ids[i]]
        print("Related document number "+str(i+1)+":")
        print("Title: "+related_doc['_source']['title'])
        if "description" in related_doc['_source']:
            print("Description: "+related_doc['_source']['description'])
        print("Link: "+related_doc['_id'])
        print()


    print(' ------------------------------------------ \n ')
    answer = input("Please pick a document by typing its number: ")
    print('\n')
    if answer == 'exit':
        exit()
    recommend_articles(related_doc_ids[int(answer) - 1], top, raw_docs, distance_matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--link', type=str, nargs=1, help='link')
    parser.add_argument('-m', '--model', type=str, nargs=1, default=['LDA'],help='LDA or LSA')
    args = parser.parse_args()

    # --------------- Parameters ---------------
    # Number of topics
    n_topics = 100

    # Languge for stop words filter
    stop_words_language = 'english'

    # Max number of vocabularies
    max_words = 50000

    top = 5

    raw_docs, docs_term_matrix, link_to_id, id_to_link, n_docs = load_data_from_elastic(stop_words_language, max_words)
    distance_matrix = get_distance_matrix(docs_term_matrix, n_topics, n_docs, args.model[0])

    print(' ------------------------------------------ \n ')
    start_doc = random.randint(0,n_docs-1)
    if args.link is not None:
        start_doc = link_to_id[args.link[0]]

    recommend_articles(start_doc, top, raw_docs, distance_matrix)


main()
