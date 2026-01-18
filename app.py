import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen SVM",
    page_icon="📊",
    layout="wide"
)

# ===============================
# NLTK
# ===============================
nltk.download("punkt")
nltk.download("stopwords")

stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))
stop_all = stop_id.union(stop_en)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_berlabel.csv")

df = load_data()

# ===============================
# PREPROCESS
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_all]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# TRAIN SVM
# ===============================
@st.cache_resource
def train_model(data):
    X = data["clean_text"]
    y = data["sentiment"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LinearSVC()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model(df)

# ===============================
# TITLE
# ===============================
st.title("📊 Dashboard Analisis Sentimen Twitter (SVM)")
st.markdown("Bahasa **Indonesia & Inggris** menggunakan **Support Vector Machine**")

# ===============================
# METRICS
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("Total Data", len(df))
col2.metric("Positive", (df["sentiment"] == "positive").sum())
col3.metric("Negative", (df["sentiment"] == "negative").sum())

# ===============================
# INPUT
# ===============================
st.subheader("📝 Analisis Tweet")

col1, col2 = st.columns(2)

with col1:
    contoh_id = "Pelayanan aplikasi ini sangat buruk dan mengecewakan."
    st.markdown("🇮🇩 **Indonesia**")
    st.code(contoh_id)
    if st.button("Gunakan Contoh ID"):
        st.session_state["tweet"] = contoh_id

with col2:
    contoh_en = "I really love this application, it works perfectly!"
    st.markdown("🇬🇧 **English**")
    st.code(contoh_en)
    if st.button("Use English Example"):
        st.session_state["tweet"] = contoh_en

tweet = st.text_area(
    "Masukkan tweet:",
    key="tweet",
    height=120
)

# ===============================
# PREDICT
# ===============================
if st.button("🔍 Prediksi Sentimen"):
    if tweet.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:
        clean = clean_text(tweet)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]

        if pred == "positive":
            st.success("✅ Sentimen POSITIVE")
        else:
            st.error("❌ Sentimen NEGATIVE")

# ===============================
# SAMPLE DATA
# ===============================
st.subheader("📄 Contoh Dataset")
st.dataframe(df.sample(5)[["text", "sentiment"]])

st.markdown("---")
st.markdown("📌 **Metode:** SVM | Data: Twitter (X)")
