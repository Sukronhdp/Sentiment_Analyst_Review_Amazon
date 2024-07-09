import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from PIL import Image

st.set_page_config(
    page_title="Amazon Reviews for Sentiment Analysis",
    page_icon="img/mzn.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image("img/ath2.png")

st.title('Visualization Amazon Reviews for Sentiment Analysis')
st.markdown("<br>", unsafe_allow_html=True)

def load_data():
    df = pd.read_csv('amazon_reviews.csv')
    return df

data = load_data()

def load_data_review():
    df = pd.read_csv('review.csv')
    return df

review = load_data_review()

# VISUALIZATION
col1, col2= st.columns(2)
with col1:
    # PIE CHART REVIEW TIME
    st.subheader('Annual Review Contributions')
    data['reviewTime'] = pd.to_datetime(data['reviewTime'])
    review_counts = data['reviewTime'].dt.year.value_counts().sort_index()
    fig1, ax = plt.subplots()
    ax.pie(review_counts, startangle=180, autopct=lambda p: f'{p:.1f}%\n({int(p * sum(review_counts) / 100)})')
    ax.legend(loc='center left', labels=review_counts.index, bbox_to_anchor=(1, 0.5)) 
    st.pyplot(fig1)
    st.markdown("<br>", unsafe_allow_html=True)

with col2:
    # BAR CHART RATINGS
    st.subheader('Number of Reviews by Rating')
    data['overall'] = data['overall'].astype(int)
    rating_counts = data['overall'].value_counts().sort_index()
    n_colors = len(rating_counts)
    colors = sns.color_palette("husl", n_colors)
    fig2 = plt.figure(figsize=(8, 6.6))
    plt.bar(rating_counts.index, rating_counts.values, color=colors, tick_label=rating_counts.index)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(fig2)

col1, col2= st.columns(2)
with col1:
    #HISTPLOT LENGTH REVIEWS
    st.subheader('Text Length Distribution')
    data['reviewText'].fillna('', inplace=True)
    data['text_length'] = data['reviewText'].apply(lambda x: len(x.split()))
    filtered_data = data[(data['text_length'] >= 0) & (data['text_length'] <= 200)]
    fig3 = plt.figure(figsize=(8, 9))
    sns.histplot(filtered_data['text_length'], bins=20)
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    st.pyplot(fig3)

with col2:
    # BAR CHART FREQUENT WORDS
    st.subheader('Most Frequent Words')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['reviewText'])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df = tfidf_df.sum().reset_index(name='count')
    tfidf_df = tfidf_df.sort_values('count', ascending=False).head(20)
    colors = sns.color_palette("husl", 20)
    fig4 = plt.figure(figsize=(8, 9.3))
    sns.barplot(x='count', y='index', data=tfidf_df, palette=colors)
    st.pyplot(fig4)
    st.markdown("<br>", unsafe_allow_html=True)

# WORDCLOUD NEGATIVE
st.subheader('Wordcloud Negative Reviews')
reviews_negatif = review[review['polarity'] == 'negative']['casefolding']
wordcloud_all= review['reviewText'].tolist()
filtered_all = ("").join(str(wordcloud_all))
filtered_all= filtered_all.lower()
wordcloud = WordCloud(max_words=300, width=1600, height=800,background_color = 'white').generate(filtered_all)
wordcloud_image = "img/negative.png"
wordcloud.to_file(wordcloud_image)
st.image(Image.open(wordcloud_image), use_column_width=True)
st.markdown("<br><br>", unsafe_allow_html=True)


# WORDCLOUD POSITIVE
st.subheader('WordCloud Positive Reviews')
reviews_positif = review[review['polarity'] == 'positive']['casefolding']
wordcloud_all= review['reviewText'].tolist()
filtered_all = ("").join(str(wordcloud_all)) 
filtered_all= filtered_all.lower()
wordcloud = WordCloud(max_words=300, width=1600, height=800,background_color = 'white').generate(filtered_all)
wordcloud_image_pos = "img/positive.png"
wordcloud.to_file(wordcloud_image_pos)
st.image(Image.open(wordcloud_image_pos), use_column_width=True)