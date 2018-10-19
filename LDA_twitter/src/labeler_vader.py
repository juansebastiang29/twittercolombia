
import pandas as pd
import numpy as np
from googletrans import Translator
import re
import preprocessor as tweet_p
import matplotlib.pyplot as plt
%matplotlib inline
import os
import nltk
from sklearn.feature_extraction.text import strip_accents_unicode
import datetime
import sys
import pickle

pd.set_option('display.max_colwidth', -1)

data_path =  "/home/juan/Desktop/Text_Mining/Om_Project/Data"
data = pd.read_csv ( os.path.join( data_path,"db_tweets.csv" ) , sep = "|", lineterminator = '\n')

def get_weekdays(date):
    return datetime.datetime.isocalendar(date)[1]

data['created_at'] = pd.to_datetime(data.created_at)

data['week_num'] = data.created_at.apply(get_weekdays)

def preprocessor_tweet(s):

    tweet_p.set_options(tweet_p.OPT.EMOJI,
                        tweet_p.OPT.URL,
                        tweet_p.OPT.RESERVED,
                        tweet_p.OPT.SMILEY,
                        tweet_p.OPT.MENTION)
    s = re.sub(r'@petrogustavo', 'petrogustavo', s)
    s = re.sub(r'@sergio_fajardo', 'sergio_fajardo', s)
    s = re.sub(r'@IvanDuque','IvanDuque',s)
    s = re.sub(r'@AlvaroUribeVel','AlvaroUribeVel',s)
    s = re.sub(r'@JuanManSantos','JuanManSantos',s)
    s = re.sub(r'@German_Vargas','German_Vargas',s)
    s = re.sub(r'@ClaudiaLopez','ClaudiaLopez',s)
    s = re.sub(r'@DeLaCalleHum','DeLaCalleHum',s)
    s = tweet_p.clean(s)
    s = re.sub(r'\b(?:a*(?:ja)+h?|(?:l+o+)+l+)\b', ' ', s)
    s = re.sub(r'[^\w]', ' ', s)
    # s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s)
    # s = re.sub(r'#', '', s)
    # s = re.sub(r'¡+', '', s)
    # s = re.sub(r':', '', s)
    # s = re.sub(r'!+', '', s)
    # s = re.sub(r'"', '', s)


    # s = re.sub(r'/[-?]/', '', s)
    # s = re.sub(r'¿+', '', s)
    # s = re.sub(r'@\w+', '', s)
    s = strip_accents_unicode(s.lower())
    s = tweet_p.clean(s)

    return s

stopwords = nltk.corpus.stopwords.words(['spanish'])
stemmer = nltk.stem.snowball.SnowballStemmer('spanish')

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

weeks = data.week_num.value_counts().index.tolist()
weeks.sort()
weeks = weeks[8:]
subset = data[data.week_num == 16]


for i in range(len(weeks)):
    subset = data[data.week_num == weeks[i]]
    subset.loc[:,"text_tweet"] = subset.text_tweet.apply(preprocessor_tweet)
    clean_tweets = []
    for user in subset.screen_name.unique():
        #grab al the tweets made by the user in the week
        item = ". ".join( list ( subset[subset.screen_name==user].text_tweet ) )
        #discard users that tweeted less than 400 char
        if len(item)<400:
            pass
        else:
            clean_tweets.append(item)

    with open(os.path.join(data_path,"clean_tweets_week_"+str(weeks[i])), 'wb') as fp:
        pickle.dump(clean_tweets, fp)

    fp.close()
    print( weeks[i] )





my_list=['cuál','pm','am','va','p m','a m','q','ver','hoy'
        'acá','aca','aqui','da','m','p','tal','tan','haga',
        'v','u','cómo','ve','retweeted','fm','usted','hace'
        'responde','espere','tambien','dice','dicen','dijo',
        'segun','segun','cada','anos','aun','aunque','cree','ay',
        'creen','creer','creo','decir','demas','estan','retwit',
        'hace','hacen','hacer','hecha','hicieron' ,'hizo','cosa','d',
        'porque','demas','diga','digo','estan','etc','ir','llega','pa','ser',
        'hoy','puede','quiere','ser','sera','si','van','ir',
        'sr','tan','ud','va','van','vamos','voy','x','vez','sra',
        'ahi','ahora','vez','via','vea','mas','b','uds','ahi','alla',
        'dejen','dejar','cosas','asi','solo','rt']
stopwords = stopwords+my_list


################################################
################################################
################################################

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

############model with gensim

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def sentence_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True remov

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV',"SCONJ","PROPN"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#Sentences to words
data_words = list(sentence_to_words(clean_tweets))

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
data_words_bigrams[1:10]

#lemmatization
nlp = spacy.load('es_core_news_md', disable=['parser', 'ner'])
data_lemmatized = lemmatization(clean_tweets)
