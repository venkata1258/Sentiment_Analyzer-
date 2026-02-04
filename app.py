import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()
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

st.markdown(
    """
    <div style="
        padding:15px;
        border-radius:8px;
        background-color:#f6f8fa;
        margin-bottom:10px;
    ">
        <p style="margin-bottom:8px;">
            Enter a Flipkart product review to analyze customer sentiment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

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
