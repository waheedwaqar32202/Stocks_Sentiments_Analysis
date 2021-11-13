import sys
import warnings
import nltk
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)


from datetime import timedelta
# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import string
from datetime import datetime
# twint libraries
import twint

class TwitterSensor:
    def FetchSensorData(self, keyword):

        config = twint.Config()
        config.Search = keyword
        config.Lang = "en"
        config.Country = "US"
        config.Limit = 10
        config.Store_csv = True
        config.Output = "my_finding.csv"
        twint.run.Search(config)

        df = pd.read_csv("my_finding.csv", skipinitialspace=True, usecols=["user_id", "username", "name", "date",
                                                                           "time", "tweet", "likes_count",
                                                                           "hashtags", "link"])

        df['publishedAt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        del df['date']
        del df['time']
        df["keyword"] = str(keyword).lower()
        return df


    def dataCleaning(self, text):
            from nltk.corpus import stopwords
            punctuation = string.punctuation
            stopwords = stopwords.words('english')
            text = text.lower()
            text = "".join(x for x in text if x not in punctuation)
            words = text.split()
            words = [w for w in words if w not in stopwords]
            text = " ".join(words)

            return text


    def DoLowLevelPerception(self, posts: pd.DataFrame) -> pd.DataFrame:
        if posts.shape[0] == 0:
            return posts
        # Clean title and make another coulmn to store cleaned title
        posts['cleaned_tweet'] = posts['tweet'].apply(self.dataCleaning)
        # calculate polarity and subjectivity of title using textblob
        posts['polarity'] = posts['cleaned_tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['cleaned_tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        return posts


    def DoEveryThing(self, stock_list):

        posts = []
        for stock in stock_list:
            df = self.FetchSensorData(stock)
            post = self.DoLowLevelPerception(df)
            posts.append(post)

        posts = pd.concat(posts, ignore_index=True)

        return posts
# main method
if __name__ == '__main__':
    pass

