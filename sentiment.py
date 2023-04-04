import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Load the CSV dataset into a Pandas DataFrame
df = pd.read_csv('train.csv')


# Define a function to analyze the sentiment of a tweet
def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


# Define a function to analyze the tone of a tweet
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


# Define a function to count threatening words in a tweet
def count_negative_threatening_words(tweet):
    pattern = re.compile(r"\b(attack|bomb|gun|kill|murder|terror|hate|scarely|seldom|barely|never|nobody|nothing|nowhere)\b", re.IGNORECASE)
    matches = pattern.findall(tweet)
    return len(matches)


df['tweet_text'] = df['tweet_text'].astype(str)

# Apply the sentiment analysis, tone analysis, and threatening words count functions to each tweet in the DataFrame
df['sentiment'] = df['tweet_text'].apply(analyze_sentiment)
df['tone'] = df['tweet_text'].apply(analyze_tone)
df['threatening_words'] = df['tweet_text'].apply(count_negative_threatening_words)

# Calculate the rate of negativity or positivity in a tweet
df['sentiment_rate'] = (df['sentiment'].value_counts(normalize=True) * 100).round(2)

# Print the resulting DataFrame with sentiment analysis, tone analysis, threatening words count, and sentiment rate
# results
print(str(df['tweet_text']) + ": " +  str(df['sentiment']) + "| " + str(df['threatening_words']) + "| " + str(df['tone']))
