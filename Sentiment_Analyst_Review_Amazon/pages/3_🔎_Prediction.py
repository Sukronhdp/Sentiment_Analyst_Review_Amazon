import streamlit as st
import re
import string
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download("stopwords")

st.set_page_config(
    page_title="Amazon Reviews for Sentiment Analysis",
    page_icon="img/mzn.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image("img/ath2.png")

st.title('Prediction Amazon Reviews for Sentiment Analysis')
st.markdown("<br>", unsafe_allow_html=True)

# Load Dataset
def load_data():
    df = pd.read_csv('amazon_reviews.csv')
    return df

data = load_data()

# Preprocessing
def cleaningText(text):
  if isinstance(text, str):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', '', text) # remove numbers
    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
  return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('english'))
    listStopwords.update(listStopwords)
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

# Load TF-IDF vectorizer
with open("tfidf_model.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Load Logistic Regression model
with open("lr_model.pkl", "rb") as lr_file:
    best_lr = pickle.load(lr_file)

# Load Random Forest model
with open("rf_model.pkl", "rb") as rf_file:
    best_fr = pickle.load(rf_file)

# Define all preprocessing function
def preprocessing_lengkap(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = tokenizingText(text)
    text = filteringText(text)
    text = toSentence(text)
    return text

# Get user input to prediction sentiment
text = st.text_input("Enter the review sentence = ")

if st.button("Predict Sentiment"):
    text_preprocessing = preprocessing_lengkap(text)
    text_tfidf = tfidf.transform([text_preprocessing])
    text_predict = best_lr.predict(text_tfidf.toarray())

    if text_predict[0] == 'positive':
        st.markdown("<h1 style='text-align: center; color: green;'>Positive Sentiment</h1>", unsafe_allow_html=True)
        st.columns(3)[1].image("img/positive_ils.png")
        
    else:
        st.markdown("<h1 style='text-align: center; color: red;'>Negative Sentiment</h1>", unsafe_allow_html=True)
        st.columns(3)[1].image("img/negative_ils.png")