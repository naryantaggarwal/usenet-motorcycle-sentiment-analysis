import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
nltk.download("punkt")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sid = SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = " ".join(words)
    return text

def get_sentiment_scores(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores["compound"]

# Define the categories of interest
categories = ['rec.motorcycles']

# Load the data using fetch_20newsgroups
df = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)

# Preprocess the text data
preprocessed_data = df.data
for i in range(len(preprocessed_data)):
    preprocessed_data[i] = preprocess_text(preprocessed_data[i])

# Get sentiment scores for each document
sentiment_scores = [get_sentiment_scores(text) for text in preprocessed_data]

# Plot the sentiment scores
plt.hist(sentiment_scores, bins=20)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Sentiment Analysis of Motorcycles Newsgroup")
plt.show()
