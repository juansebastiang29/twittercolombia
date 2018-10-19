
import pandas as pd
import numpy as np
import re
import preprocessor as tweet_p
import os
import datetime
import sys
import pickle
from sklearn.feature_extraction.text import strip_accents_unicode

print("Initiating.. ")

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

print("data_set processed!!")

weeks = data.week_num.value_counts().index.tolist()
weeks.sort()
weeks = weeks[8:]
for i in range(len(weeks)):
    print("starting ",weeks[i])
    subset = data[data.week_num == weeks[i]]
    subset.loc[:,"text_tweet"] = subset.text_tweet.apply(preprocessor_tweet)
    clean_tweets = []
    j=0
    for user in subset.screen_name.unique():
        #grab al the tweets made by the user in the week
        item = ". ".join( list ( subset[subset.screen_name==user].text_tweet ) )
        if j%10000==0:
            print("+10000 users processed ")
        #discard users that tweeted less than 400 char
        if len(item)<400:
            pass
        else:
            clean_tweets.append(item)
        j+=1

    with open(os.path.join(data_path,"clean_tweets_week_"+str(weeks[i])), 'wb') as fp:
        pickle.dump(clean_tweets, fp)

    fp.close()
    print( weeks[i] )
