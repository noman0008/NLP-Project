# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:29:16 2017

@author: ABUSALEH
"""
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from pymongo import MongoClient

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import math
tokenize = lambda doc: doc.lower().split(" ")

stop = set(stopwords.words('english'))


exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

client = MongoClient()

db = client.test_db
collection = db.test_table2

list_of_comments = []

for item in collection.find():
    for comment in item['comments']:
        list_of_comments.append(comment['comment_body'])

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
        
doc_clean = [clean(doc) for doc in list_of_comments] 

###~~~LEXICAL
def ngramize(text):
    token = nltk.word_tokenize(text)
    bigrams = ngrams(token,2)
    trigrams = ngrams(token,3)
    fourgrams = ngrams(token,4)
    fivegrams = ngrams(token,5)
    
    return list(bigrams),list(trigrams), list(fourgrams)
 
###~~~POS   
def POS(text):
    
    token = nltk.word_tokenize(text)
    return nltk.pos_tag(token), [item[1] for item in nltk.pos_tag(token)]

###~~~SEMANTIC

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)
 
def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)
 
def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)
 
def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))
 
def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values
 
def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(doc_clean)

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude
    
tfidf_representation = tfidf(doc_clean)
our_tfidf_comparisons = []
for count_0, doc_0 in enumerate(tfidf_representation):
    for count_1, doc_1 in enumerate(tfidf_representation):
        our_tfidf_comparisons.append(cosine_similarity(doc_0, doc_1))

print(our_tfidf_comparisons)

f= open('Lexical_features.csv','w')
f.write('Bigrams'+','+ 'Trigrams'+','+'fourgrams\n')  
for item in doc_clean:
    f.write(str(ngramize(item)[0])+','+str(ngramize(item)[1])+','+str(ngramize(item)[2])+'\n')

f= open('POS_features.csv','w')
f.write('POS_TagWithWords'+','+ 'POS_Tags\n') 
print(len(doc_clean))
for item in doc_clean:
     f.write(str(POS(item)[0])+','+str(POS(item)[1])+'\n')

     

    