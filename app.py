import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# NLTK Setup (for Streamlit Cloud)
# ----------------------------
nltk_data_dir = "nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources only if missing
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.download("wordnet", download_dir=nltk_data_dir)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
try:
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the app folder.")
    st.stop()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ----------------------------
# Sentiment Prediction
# ----------------------------
def analyze_sentiment(review):
    review_clean = clean_text(review)
    review_vec = tfidf.transform([review_clean])
    prediction = model.predict(review_vec)[0]
    probability = model.predict_proba(review_vec)[0][prediction] * 100
    return prediction, probability

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    """
    <div style="text-align:center;">
        <h1>Product Sentiment Analyzer</h1>
        <p style="font-size:16px;">
            AI-based sentiment analysis using Machine Learning
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

st.markdown("### 📝 Product Review")

review = st.text_area(
    label="",
    placeholder="Example: The shuttle quality is excellent and lasts long.",
    height=160
)

st.write("")
analyze = st.button("Analyze Review")

# ----------------------------
# Output Section
# ----------------------------
if analyze:
    if review.strip() == "":
        st.warning("Please enter a product review before analyzing.")
    else:
        sentiment, confidence = analyze_sentiment(review)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Sentiment Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
        with col2:
            if sentiment == 1:
                st.success("Positive Customer Feedback 😊")
            else:
                st.error("Negative Customer Feedback 😞")

        st.progress(confidence / 100)
