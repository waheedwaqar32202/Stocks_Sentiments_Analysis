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

# newapi libraries
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime


class NewsSensor:
    def __init__(self):
        # Initiating the news api client with key
        self.newsapi = NewsApiClient(api_key='172f499f682544b398482a14b1c408db')
        self.current_date = datetime.today().strftime('%Y-%m-%d')
        self.last_month_date = pd.to_datetime(self.current_date, format="%Y-%m-%d") - pd.DateOffset(months=1)

    def convertDate(self, d):
        date = datetime.fromisoformat(d[:-1])
        return date.strftime('%Y-%m-%d %H:%M:%S')

    def FetchSensorData(self, keyword) -> pd.DataFrame:
        all_articles = self.newsapi.get_everything(q=keyword, from_param=self.last_month_date,
                                      to=self.current_date,
                                      language='en')

        articles = all_articles['articles']
        news_data = pd.DataFrame(articles)

        count = 0
        for post in all_articles['articles']:
            news_data['source'][count] = post['source']['name']
            count = count + 1

        news_data["keyword"] = str(keyword).lower()
        # news_data.publishedAt.fillna(value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), inplace=True)
        if 'publishedAt' in list(news_data.columns):
            news_data['publishedAt'] = news_data['publishedAt'].apply(self.convertDate)

        return news_data

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

    def DoLowLevelPerception(self, posts: pd.DataFrame):
        if posts.shape[0] == 0:
            return posts
        posts["combined"] = (posts["title"] + posts["description"] + posts["content"])
        posts = posts.dropna(axis=0, subset=['combined'])
        posts['combined'].astype('str')
        posts['cleaned'] = posts['combined'].apply(self.dataCleaning)

        posts['polarity'] = posts['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['cleaned'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        del posts['combined']
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