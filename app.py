import streamlit as st
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    score = ((polarity + 1) / 2) * 100
    return score

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
            Instantly understand customer satisfaction from product reviews
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
            Write a short review describing the product experience
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

review = st.text_area(
    label="",
    placeholder="Example: The product quality is excellent, but the delivery was slower than expected.",
    height=160
)

st.write("")
analyze = st.button("Analyze Review")

if analyze:
    if review.strip() == "":
        st.warning("Please enter a product review before analyzing.")
    else:
        score = analyze_sentiment(review)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Sentiment Result")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(label="Satisfaction Score", value=f"{score:.1f}%")

        with metric_col2:
            if score >= 70:
                st.success("Positive Customer Feedback")
            elif score <= 40:
                st.error("Negative Customer Feedback")
            else:
                st.info("Neutral or Mixed Feedback")

        st.progress(score / 100)
