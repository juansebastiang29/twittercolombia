import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import numpy as np
import os
import json
import mysql.connector
import preprocessor as tweet_p
import unidecode
import re
import itertools
import codecs
from sklearn import feature_extraction
import mpld3

pd.set_option('display.max_colwidth', -1)

cnx = mysql.connector.connect(user='root', password='123456',
                              host='localhost',
                              database='twitter_elecciones')
cursor = cnx.cursor()
query_tweets = ("SELECT * FROM twitter where twitter.lang like '%es%' ORDER BY id")
# query_RT=("select * from twitter_rt_fav")
tweets=pd.read_sql_query(sql=query_tweets,con=cnx,chunksize=10000)

i=0
with open('/home/juan/Desktop/Text_Mining/Om_Project/db_tweets.csv','a') as f:
    for chunk in tweets:
        if i == 0:
            chunk.to_csv(f,index=False,sep = "|")
        else:
            chunk.to_csv(f,index=False,sep = "|",header = False)
        i=+1
f.close()

# chunk.tail()
# cnx.close()
