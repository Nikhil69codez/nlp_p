import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    df = pd.read_csv('bbc-text.csv')
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['text'])
    y = df['category']
    model = MultinomialNB()
    model.fit(X, y)
    return tfidf, model