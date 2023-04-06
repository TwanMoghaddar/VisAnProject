import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import seaborn as sns

"""
@Author: Antoine Moghaddar
@Date: 30/03/2023
@University of Eindhoven; MSc Data Science & AI

@Contents

This code is a collection of functions that perform natural language processing (NLP) on a given CSV dataset. 
NLP Algorithm using the VaderSentiment and Textblob library to classify tweets into categories of positivity.
The classification is built upon the principle of text splitting and language processing wherein a blacklist of 
words is used to determine the tone and the mental intentions of the tweet.
"""

"""
The dataset is loaded using the load_data function, which reads a CSV file and returns a Pandas DataFrame object 
containing the data
"""


def load_data(filename):
    # Load the CSV dataset into a Pandas DataFrame
    df = pd.read_csv(filename)

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

    return df


"""
The analyze_sentiment function takes a tweet as input and uses the TextBlob library to determine the sentiment of 
the tweet. If the sentiment is positive, the function returns 
"Positive"; if negative, "Negative"; if neutral, "Neutral".
"""


def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


"""
The analyze_tone function takes a tweet as input and uses the SentimentIntensityAnalyzer library to determine 
the tone of the tweet. The function returns "Very Positive" for a compound sentiment score greater than 0.5, 
"Positive" for a score greater than 0, "Very Negative" for a score less than -0.5, "Negative" for a score less 
than 0, and "Neutral" for a score of 0.
"""


def analyze_tone(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(tweet)
    if sentiment['compound'] > 0.5:
        return 'Very Positive'
    elif sentiment['compound'] > 0:
        return 'Positive'
    elif sentiment['compound'] < -0.5:
        return 'Very Negative'
    elif sentiment['compound'] < 0:
        return 'Negative'
    else:
        return 'Neutral'


"""
The count_negative_threatening_words function takes a tweet as input and uses a regular expression pattern to count
the number of negative and threatening words in the tweet. The function returns the count of such words.
"""


def count_negative_threatening_words(tweet):
    pattern = re.compile(r"\b(attack|bomb|gun|kill|murder|terror|hate|scarely|seldom|barely|never|nobody|nothing"
                         r"|nowhere)\b", re.IGNORECASE)
    matches = pattern.findall(tweet)
    return len(matches)


"""
The runnable function loads the CSV dataset, applies the analyze_sentiment, analyze_tone, 
and count_negative_threatening_words functions to each tweet in the DataFrame, and calculates the rate of negativity 
or positivity in a tweet. The function then prints the resulting DataFrame with the sentiment analysis, 
tone analysis, threatening words count, and sentiment rate results. Finally, the function returns the resulting 
DataFrame object.
"""


def runnable(file):
    df = load_data(filename=file)

    df['tweet_text'] = df['tweet_text'].astype(str)

    # Apply the sentiment analysis, tone analysis, and threatening words count functions to each tweet in the DataFrame
    df['sentiment'] = df['tweet_text'].apply(analyze_sentiment)
    df['tone'] = df['tweet_text'].apply(analyze_tone)
    df['threatening_words'] = df['tweet_text'].apply(count_negative_threatening_words)

    # Calculate the rate of negativity or positivity in a tweet
    df['sentiment_rate'] = (df['sentiment'].value_counts(normalize=True) * 100).round(2)

    # Print the resulting DataFrame with sentiment analysis, tone analysis, threatening words count, and sentiment rate
    # results
    print(
        str(df['tweet_text']) + ": " + str(df['sentiment']) + "| " + str(df['threatening_words']) + "| " + str(
            df['tone']))

    return df
