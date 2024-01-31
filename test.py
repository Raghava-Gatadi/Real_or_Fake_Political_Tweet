import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# You might need to download the 'vader_lexicon'
nltk.download('vader_lexicon')

# Load your data
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

# Initialize the sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis on your tweets
df['sentiments'] = df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# Extract compound score
df['compound'] = df['sentiments'].apply(lambda sentiment: sentiment['compound'])

# Classify sentiment as positive, neutral, or negative based on compound score
df['sentiment_class'] = df['compound'].apply(lambda score: 'positive' if score > 0 else ('neutral' if score == 0 else 'negative'))
print(df['sentiment_class'])