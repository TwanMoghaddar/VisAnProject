import pandas as pd
from textblob import TextBlob

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


df['tweet_text'] = df['tweet_text'].astype(str)

# Apply the sentiment analysis function to each tweet in the DataFrame
df['sentiment'] = df['tweet_text'].apply(analyze_sentiment)

# Print the resulting DataFrame with sentiment analysis results
print(df['sentiment'] + ": " +  df['tweet_text'])
