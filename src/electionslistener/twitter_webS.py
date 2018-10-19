
###########################mysql sol#######################

from __future__ import print_function
import tweepy
import json
import mysql.connector
from dateutil import parser
import gc
import re

cnx = mysql.connector.connect(user='root', password='123456',
                              host='localhost',
                              database='twitter_elecciones')

cnx.close()

WORDS = ["Elecciones Colombia",
        "Elección Presidencial",
        '#EleccionesColombia',
        '#ColombiaDecide',
        'Alvaro Uribe',
        'Petro Colombia','Petro','Gustavo Petro',
        'Claudia Lopez',
        'Cambio Radical',
        'Vargas Lleras','petro y uribe',
        'ivan duque',
        'Centro Democratico',
        'Humberto de la Calle',
        'Sergio Fajardo',
        'Fajardo','Partido Verde Colombia',
        'Uribe','FARC','uribismo','Registraduría Nacional',
        'Coalicion Colombia','Partido liberal',
        'Alejandro Ordoñez','Martha Lucía Ramírez','Juan Carlos Pinzon',
        'Poder ciudadano Colombia','Partido Conservador','Colombia Humana',
        'Partido de la U','Mejor Vargas Lleras',
        'Gustavo Bolivar','Voto Colombia',
        '@JuanManSantos','Votar Colombia',
        'Votemos Colombia','Senado Colombia',
        'Elecciones Colombianas','Mockus',
        '#DuqueEsElQueEs','@IvanDuque',
        '#LaGranEncuesta','@petrogustavo',
        '@sergio_fajardo','@DeLaCalleHum',
        '@German_Vargas','colombiahumana','colombiadecide',
        'pilasconelvoto','eleccionescolombia2018']

# This function takes the 'created_at', 'text', 'screen_name' and 'tweet_id' and stores it
# into a MySQL database
def store_data(created_at, tweet_id, screen_name, text_tweet,hashtags,mentions,in_reply_to_user_id,in_reply_to_status_id,lang,followers_count,user_description,user_id):
    db=mysql.connector.connect(user='root', password='123456',
                                  host='localhost',
                                  database='twitter_elecciones')
    cursor = db.cursor()
    insert_query = "INSERT INTO twitter (created_at, tweet_id, screen_name, text_tweet,hashtags,mentions,in_reply_to_user_id,in_reply_to_status_id,lang,followers_count,user_description,user_id) VALUES (%s, %s, %s, %s,%s,%s, %s, %s, %s,%s,%s, %s)"
    cursor.execute(insert_query, (created_at, tweet_id, screen_name, text_tweet,hashtags,mentions,in_reply_to_user_id,in_reply_to_status_id,lang,followers_count,user_description,user_id))
    db.commit()
    cursor.close()
    db.close()
    return

def store_data_rt(created_at,tweet_id,screen_name_rt,screen_name_org,reweet_count,favorite_count,reply_count):
    db=mysql.connector.connect(user='root', password='123456',
                                  host='localhost',
                                  database='twitter_elecciones')
    cursor = db.cursor()
    insert_query = "INSERT INTO twitter_rt_fav (created_at,tweet_id,screen_name_rt,screen_name_org,reweet_count,favorite_count,reply_count) VALUES (%s, %s, %s, %s,%s,%s,%s)"
    cursor.execute(insert_query, (created_at,tweet_id,screen_name_rt,screen_name_org,reweet_count,favorite_count,reply_count))
    db.commit()
    cursor.close()
    db.close()
    return


class StreamListener(tweepy.StreamListener):
    #This is a class provided by tweepy to access the Twitter Streaming API.

    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")

    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        #This is the meat of the script...it connects to your mongoDB and stores the tweet
        try:
           # Decode the JSON from Twitter
            datajson = json.loads(data)
            #grab the wanted data from the Tweet
            created_at = parser.parse(datajson['created_at'])
            tweet_id = datajson['id']
            use_name=datajson['user']['name']
            screen_name = datajson['user']['screen_name']
            if 'extended_tweet' in datajson:
                text_tweet=datajson['extended_tweet']['full_text']
            else:
                text_tweet=datajson['text']
            # favorited=datajson['favorited']
            # favorite_count=datajson['favorite_count']
            # retweeted=datajson['retweeted']
            # retweet_count=datajson['retweet_count']
            hashtags=str(datajson['entities']['hashtags'])
            mentions=str(datajson['entities']['user_mentions'])
            in_reply_to_user_id=datajson['in_reply_to_user_id']
            in_reply_to_status_id=datajson['in_reply_to_status_id']
            lang=datajson['lang']
            followers_count=datajson['user']['followers_count']
            user_id=datajson['user']['id_str']
            user_description=datajson['user']['description']
            #out a message to the screen that we have collected a tweet
            gc.collect()
            if 'retweeted_status' in datajson.keys():
                tweet_id=datajson['retweeted_status']['id_str']
                screen_name_rt=screen_name
                screen_name_org=datajson['retweeted_status']['user']['screen_name']
                reweet_count=datajson['retweeted_status']['retweet_count']
                favorite_count=datajson['retweeted_status']['favorite_count']
                reply_count=datajson['retweeted_status']['reply_count']
                store_data_rt(created_at,tweet_id,screen_name_rt,screen_name_org,reweet_count,favorite_count,reply_count)
            else:
                print("Tweet collected at " + str(created_at))
                emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                   "]+", flags=re.UNICODE)
                text_tweet=emoji_pattern.sub(r'', text_tweet) # no emoji
                if user_description!=None:
                    user_description=emoji_pattern.sub(r'', user_description) # no emoji
                if mentions!=None:
                    mentions=emoji_pattern.sub(r'', mentions)
                store_data(created_at, tweet_id, screen_name, text_tweet,hashtags,mentions,in_reply_to_user_id,in_reply_to_status_id,lang,followers_count,user_description,user_id)
        except Exception as e:
           print(e)

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.
listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True,timeout=60*5,wait_on_rate_limit_notify=True))
streamer = tweepy.Stream(auth=auth, listener=listener)
print("Tracking: " + str(WORDS))
streamer.filter(track=WORDS)
