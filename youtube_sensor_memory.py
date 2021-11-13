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

from datetime import datetime
# youtube api
from youtube_easy_api.easy_wrapper import *


class YouTubeSensor:
    def __init__(self):
        self.YouTubeApiKey = 'AIzaSyCkZZ5Tr035TXvR9ChgvzaNJfJW5qnyjEM'
        self.Youtube = build('youtube', 'v3', developerKey=self.YouTubeApiKey)
        # end

    def FetchSensorData(self, keyword) -> pd.DataFrame:
        # start code
        req = self.Youtube.search().list(q=keyword, part='snippet', type='video', maxResults=10000)
        res = req.execute()

        title = []
        video_id = []
        publishedAt = []
        description = []
        views = []
        liked = []
        disliked = []
        liked = []
        comment_count = []

        for i in range(len(res) - 1):
            title.append(res['items'][i]['snippet']['title'])
            video_id.append(res['items'][i]['id']['videoId'])
            date = datetime.fromisoformat(res['items'][i]['snippet']['publishedAt'][:-1])
            publishedAt.append(date.strftime('%Y-%m-%d %H:%M:%S'))
            description.append(res['items'][i]['snippet']['description'])
            stats = (self.Youtube).videos().list(id=res['items'][i]['id']['videoId'], part='statistics').execute()
            if (len(stats['items'][0]['statistics']) == 5):
                views.append(stats['items'][0]['statistics']['viewCount'])
                liked.append(stats['items'][0]['statistics']['likeCount'])
                disliked.append(stats['items'][0]['statistics']['dislikeCount'])
                comment_count.append(stats['items'][0]['statistics']['commentCount'])
            else:
                views.append('NA')
                liked.append('NA')
                disliked.append('NA')
                comment_count.append('NA')

        data = {'title': title, 'video_id': video_id, 'publishedAt': publishedAt, 'description': description,
                'views': views, 'liked': liked, 'disliked': disliked, 'comment_count': comment_count}

        df = pd.DataFrame(data)
        df["keyword"] = str(keyword).lower()
        return df
        # end code

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
        posts['cleaned_description'] = posts['description'].apply(self.dataCleaning)

        # calculate polarity and subjectivity of title using textblob
        posts['polarity'] = posts['cleaned_description'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['cleaned_description'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        del posts['cleaned_description']
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