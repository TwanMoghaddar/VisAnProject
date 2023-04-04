import string

import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = None


def load_data():
    dataFrame = pd.read_csv("train.csv", header=0)
    df = dataFrame

    # Check to see all variables we could work with
    # print(list(df.columns))

    # Filter only relevant df for the NLP
    df.drop('id', inplace=True, axis=1)
    df.drop('retweet_count', inplace=True, axis=1)
    df.drop('favorite_count', inplace=True, axis=1)
    df.drop('device', inplace=True, axis=1)
    df.drop('retweeted_status_id', inplace=True, axis=1)
    df.drop('latitude', inplace=True, axis=1)
    df.drop('longitude', inplace=True, axis=1)
    df.drop('state', inplace=True, axis=1)
    df.drop('inserted_at', inplace=True, axis=1)
    df.drop('tw_user_id', inplace=True, axis=1)

    # Check to see which columns remain
    # print(list(df.columns))

    # df['tweet_text'] = df['tweet_text'].str.replace('\W', '', regex=True)
    # df['tweet_text'] = df['tweet_text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    # df['new_tweet'] = df['tweet_text'].astype('|S80')
    df['new_tweet'] = df['tweet_text'].str.replace('@', '')
    # ax = sns.countplot(df.candidate_id)
    print(df.loc[:, 'new_tweet'])

    # df['new_tweets'] = re.sub'/.*','', '_',
    #                           df.tweet_text.str)
    # # df.tweet_text.str.replace('@', '')
    # print(df.head())

    df['new_tweet'] = df['new_tweet'].str.replace("[^a-zA-Z#]", " ")
    df['new_tweet'] = df['new_tweet'].str.replace("#", "")
    df.head()

    all_words = []
    for line in list(df['new_tweet']):
        # print(line)
        words = str(line).split()
        for word in words:
            all_words.append(word.lower())

    a = Counter(all_words).most_common(10)
    print("Highest counted words: \n)" + str(a) + "\n\n")

    df['new_tweet'] = df['new_tweet'].apply(lambda x: str(x).split())
    print(df.head())

    stemmer = SnowballStemmer("english")

    df['new_tweet'] = df['new_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
    print(df.head())

    nltk.download('stopwords')

    # stopwords = set(stopwords.words('english'))
    stopwords = nltk.corpus.stopwords.words('english')

    newStopWords = ['u', 'go', 'got', 'via', 'or', 'ur', 'us', 'in', 'i', 'let', 'the', 'to', 'is', 'amp', 'make',
                    'one', 'day', 'days', 'get']
    stopwords.extend(newStopWords)

    df['new_tweet'] = df['new_tweet'].apply(process)
    print(df.head())


def process(text):
    # Check characters to see if they are in punctuation
    nopunc = set(char for char in list(text) if char not in string.punctuation)
    # Join the characters to form the string.
    nopunc = " ".join(nopunc)
    # remove any stopwords if present
    return [word for word in nopunc.lower().split() if word.lower() not in stopwords]


if __name__ == '__main__':
    load_data()
