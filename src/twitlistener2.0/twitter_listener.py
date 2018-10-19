
import tweepy
import json
from pymongo import MongoClient
import click

# get the db host
MONGO_HOST = "mongodb://localhost/twitterdb"
words = ["ivanduque", "carrasquilla", "wradiocolombia",
         "congreso colombia", "alvaro uribe", "@elcolombiano",
         "@cconstitucional", "@AlvaroUribeVel", "FARC","@rcnradio",
         "@ivanduque", "@eltiempo", "corrupcion colombia","@caracolradio",
         "corrupcion colombia", "@A_OrdonezM","@petrogustavo",
         "Alejandro Ordoñez", "Maria Fernanda Cabal", "gustavo petro",
         "@RevistaSemana", "Armando Benedetti", "@Fiscaliacol",
         "Fiscalia", "@elespectador", "@lafm", "@mindefensa","paz colombia",
         'Petro Colombia', 'Centro Democratico', 'FARC', 'uribismo',
         'Registraduría Nacional',"@Caidadelatorre", 'Colombia Humana', "ELN",
         "lider social","lideres sociales", "@MafeCarrascal", "@HOLLMANMORRIS",
         "@EstebanSantos10", "mermelada santos", "@JuanSLopezM", "@MabelLaraNews",
         "@YoAlejoV", "@ClaudiaLopez", "@jciragorri", "@DanielSamperO",
         "@JuanManSantos"]


@click.command()
@click.option('--consumer_key', help='Twitter Credentials')
@click.option('--consumer_secret', help='Twitter Credentials')
@click.option('--access_token', help='Twitter Credentials')
@click.option('--access_token_secret', help='Twitter Credentials')
def main(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    # Set up the listener. The 'wait_on_rate_limit=True' is needed to help with Twitter API rate limiting.
    listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True,
                                             timeout=60*5,
                                             wait_on_rate_limit_notify=True))
    streamer = tweepy.Stream(auth=auth, listener=listener)

    print("Tracking: " + str(words))
    streamer.filter(track=words)


class StreamListener(tweepy.StreamListener):
    # This is a class provided by tweepy to access the Twitter Streaming API.

    def on_connect(self):
        # Called initially to connect to the Streaming API
        print("You are now connected to the streaming API.")

    def on_error(self, status_code):
        # On error - if an error occurs, display the error / status code
        print('An Error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        # This is the meat of the script...it connects to your mongoDB and stores the tweet
        try:

            client = MongoClient(MONGO_HOST)

            # Use twitterdb database. If it doesn't exist, it will be created.
            db = client.twitterdb
            result = dict()

            # Decode the JSON from Twitter
            datajson = json.loads(data)
            # grab the wanted data from the Tweet
            result["created_at"] = datajson['created_at']
            result['tweet_id'] = datajson['id']

            if 'retweeted_status' in datajson.keys():

                result_rt = dict()
                result_rt["created_at"] = datajson['created_at']
                result_rt['tweet_id'] = datajson['retweeted_status']['id_str']
                result_rt['screen_name_rt'] = datajson['user']['screen_name']
                result_rt['screen_name_original'] = datajson['retweeted_status']['user']['screen_name']
                result_rt['entities'] = datajson['entities']
                result_rt['retweet_count'] = datajson['retweeted_status']['retweet_count']
                result_rt['favorite_count'] = datajson['retweeted_status']['favorite_count']
                result_rt['reply_count'] = datajson['retweeted_status']['reply_count']
                db.twitter_search_RTs.insert_one(result_rt)

            else:
                if 'extended_tweet' in datajson:
                    result['text_tweet'] = datajson['extended_tweet']['full_text']
                else:
                    result['text_tweet'] = datajson['text']

                result['in_reply_to_user_id'] = datajson['in_reply_to_user_id']
                result['in_reply_to_status_id'] = datajson['in_reply_to_status_id']
                result['in_reply_to_screen_name'] = datajson['in_reply_to_screen_name']
                result['user'] = datajson['user']
                result['entities'] = datajson['entities']
                result['favorited'] = datajson['favorited']
                result['retweeted'] = datajson['retweeted']
                result['lang'] = datajson['lang']
                db.twitter_search.insert_one(result)

        except Exception as e:
            print(data)
            print(e)


if __name__ == "__main__":
    main()
