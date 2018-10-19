

import pandas as pd
import numpy as np
from googletrans import Translator
import re
import preprocessor as tweet_p
import os
import nltk
from sklearn.feature_extraction.text import strip_accents_unicode
import datetime
import sys
import pickle


pd.set_option('display.max_colwidth', -1)

data_path =  "/home/juan/Desktop/Text_Mining/Om_Project/Data"
week=16
with open(os.path.join(data_path,"clean_tweets_week_"+str(week)), 'rb') as fp:
    tweets_week = pickle.load(fp)
fp.close()

stopwords = nltk.corpus.stopwords.words(['spanish'])
stemmer = nltk.stem.snowball.SnowballStemmer('spanish')

my_list=['cual','pm','am','va','p m','a m','q','ver','hoy',
        'aca','aqui','da','m','p','tal','tan','haga',
        'v','u','como','ve','retweeted','fm','usted','hace',
        'responde','espere','tambien','dice','dicen','dijo',
        'segun','segun','cada','anos','aun','aunque','cree','ay',
        'creen','creer','creo','decir','demas','estan','retwit',
        'hace','hacen','hacer','hecha','hicieron' ,'hizo','cosa','d',
        'porque','demas','diga','digo','estan','etc','ir','llega','pa','ser',
        'hoy','puede','quiere','ser','sera','si','van','ir',
        'sr','tan','ud','va','van','vamos','voy','x','vez','sra',
        'ahi','ahora','vez','via','vea','mas','b','uds','ahi','alla',
        'dejen','dejar','cosas','asi','solo','rt','ps','petro',
        'ivanduque','petrogustavo','sergio_fajardo','delacallehum',
        'german_vargas','duque','fajardo','vargas','lleras','colombia',
        'alvaro','uribe','colombiano','venezuela','candidato','voto',
        'votar','juanmansantos']

stopwords.extend(my_list)


################################################
################################################
################################################

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy

############model with gensim

def sentence_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True remov

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#Load spacy
nlp = spacy.load('es_core_news_md', disable=['parser', 'ner'])

#Sentences to words
data_words = list(sentence_to_words(tweets_week))


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

#lemmatization
data_lemmatized = lemmatization(data_words_bigrams)

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# id2word.filter_n_most_frequent(int(len(id2word)*0.005))

texts = data_lemmatized.copy()

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


#Build LDA Model

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=8,
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=20,
#                                            alpha=0.001,
#                                            per_word_topics=True)

lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=10,
                                                   random_state=100,
                                                   eval_every=1,
                                                   chunksize=100,
                                                   passes=20,
                                                   alpha="auto",
                                                   eta="auto",
                                                   per_word_topics=True)

for topic in lda_model.print_topics():
    print(topic)
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(lda_model.log_perplexity(corpus))
print('\nCoherence Score: ', coherence_lda)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS



fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,10))
i=0
for t in range(2):
    for j in range(2):
        ax[t,j].imshow(WordCloud(background_color="white",
                            width = 1024,
                            height = 720,
                            random_state=5).fit_words(dict(lda_model.show_topic(i, 200))))
        ax[t,j].set_title("Topic #" + str(i)+"\n")
        ax[t,j].axis('off')
        i+=1
plt.savefig("/home/juan/Desktop/Text_Mining/Om_Project/colombia-elections-twitter/sentiment-analysis/figs/topics_week_"+str(week)+".png",dpi=300,format = "png",bbox_inches="tight")
plt.show()
