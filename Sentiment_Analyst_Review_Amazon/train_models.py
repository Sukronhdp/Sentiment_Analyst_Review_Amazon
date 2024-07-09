import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Muat dataset
df = pd.read_csv('amazon_reviews.csv')

# Preprocessing data
X = df['reviewText']
y = df['sentiment']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Latih model Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Latih model Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Simpan TF-IDF vectorizer dan model
with open('tfidf_model.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

with open('lr_model.pkl', 'wb') as lr_file:
    pickle.dump(lr_model, lr_file)

with open('rf_model.pkl', 'wb') as rf_file:
    pickle.dump(rf_model, rf_file)

print("Models and vectorizer have been saved successfully.")
