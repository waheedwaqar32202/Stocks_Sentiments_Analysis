# Import the libraries
import sys
import warnings
import nltk
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)



# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import string
# twint libraries
import twint


# newapi libraries
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime

# reddit libraries
import praw

# youtube api
from youtube_easy_api.easy_wrapper import *

class RedditSensor:
    def __init__(self):
        self.reddit = praw.Reddit(client_id='9HwE8uvXZeJhVc8yQg7ISQ',
                                  client_secret='LdgRC8E7u8CfnRAU7Ymb06l73aolmg',
                                  user_agent='Muhammad Waheed Waqar')

    def FetchSensorData(self, keyword) -> pd.DataFrame:
        posts = []
        keyword = str(keyword).lower()
        # scrapped data from reddit using reddit api
        for post in self.reddit.subreddit("all").search(keyword):
            date_type = type(post.created)
            date = datetime.fromtimestamp(post.created).strftime('%Y-%m-%d %H:%M:%S') \
                if date_type == int or date_type == float else post.created
            record = {"id": str(post.id).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'),
                      'url': str(post.url).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'),
                      'title': str(post.title).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'),
                      'body': str(post.selftext).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'),
                      'publishedAt': date,
                      'num_comments': str(post.num_comments).encode('utf-8', 'surrogateescape').decode('utf-8',
                                                                                                       'replace'),
                      'score': post.score,
                      'upvote_ratio': post.upvote_ratio,
                      'ups': post.ups, 'downs': post.downs,
                      'keyword': keyword}
            posts.append(record)

        return pd.DataFrame(posts)

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

        # Clean title and make another column to store cleaned title
        posts['title'] = posts['title'].apply(self.dataCleaning)

        # calculate polarity and subjectivity of title using textblob
        posts['polarity'] = posts['title'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['title'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)

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