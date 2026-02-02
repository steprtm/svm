import streamlit as st
import pandas as pd
import nltk
import re
import json
import joblib
import numpy as np
from collections import Counter
from nltk.corpus import stopwords

# ===============================
# NLTK SETUP
# ===============================
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
all_stopwords = stopwords_id.union(stopwords_en)

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Twitter (SVM)",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen Twitter (Binary)")
st.write(
    "Analisis sentimen Twitter menggunakan **TF-IDF + Support Vector Machine (SVM)** "
    "dengan klasifikasi **Positive vs Negative**."
)

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_berlabel.csv")

df = load_dataset()
df["sentiment"] = df["sentiment"].str.lower()

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("model/svm_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

svm_model, tfidf_vectorizer = load_model()

# ===============================
# PREPROCESS
# ===============================
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# RINGKASAN DATASET
# ===============================
st.subheader("📌 Ringkasan Dataset")

total = len(df)
pos = (df["sentiment"] == "positive").sum()
neg = (df["sentiment"] == "negative").sum()

# hitung persentase
pos_pct = (pos / total) * 100
neg_pct = (neg / total) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Tweet", total)

with col2:
    st.markdown(
        f"""
        <div style="
            background-color:#e6f4ea;
            padding:18px;
            border-radius:12px;
            text-align:center;
        ">
            <h3 style="color:green;margin-bottom:5px;">Positive</h3>
            <h2 style="margin:0;">{pos}</h2>
            <p style="margin:0;">{pos_pct:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="
            background-color:#fdecea;
            padding:18px;
            border-radius:12px;
            text-align:center;
        ">
            <h3 style="color:red;margin-bottom:5px;">Negative</h3>
            <h2 style="margin:0;">{neg}</h2>
            <p style="margin:0;">{neg_pct:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ===============================
# DISTRIBUSI SENTIMEN
# ===============================
st.subheader("📈 Distribusi Sentimen")
st.bar_chart(df["sentiment"].value_counts())

# ===============================
# CONTOH DATA
# ===============================
st.subheader("📂 Contoh Tweet Random")

sample_df = df.sample(10)[["clean_text", "sentiment"]]
st.dataframe(sample_df, use_container_width=True)

# ===============================
# EVALUASI MODEL
# ===============================
st.subheader("🎯 Evaluasi Model")

with open("model/model_metrics.json") as f:
    metrics = json.load(f)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
c4.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")

# ===============================
# UJI SENTIMEN
# ===============================
st.subheader("🧠 Uji Sentimen Manual")

user_input = st.text_area("Masukkan teks tweet:")

if st.button("Prediksi"):
    clean_input = preprocess_input(user_input)
    vector = tfidf_vectorizer.transform([clean_input])

    score = svm_model.decision_function(vector)[0]

    # convert ke probabilitas (sigmoid)
    prob_pos = 1 / (1 + np.exp(-score))
    prob_neg = 1 - prob_pos

    result_df = pd.DataFrame({
        "Sentimen": ["Positive", "Negative"],
        "Persentase (%)": [prob_pos*100, prob_neg*100]
    })

    st.dataframe(result_df, use_container_width=True)

    pred = svm_model.predict(vector)[0]
    st.success(f"Hasil Prediksi: **{pred.upper()}**")

# ===============================
# TOP WORDS
# ===============================
st.subheader("🔤 Top 15 Kata Paling Sering Muncul")

def get_top_words(text_series, n=15):
    words = []
    for text in text_series.dropna():
        tokens = [t for t in text.split() if t not in all_stopwords and len(t) > 3]
        words.extend(tokens)
    return Counter(words).most_common(n)

top_df = pd.DataFrame(get_top_words(df["clean_text"]), columns=["Kata", "Frekuensi"])
st.dataframe(top_df, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("TF-IDF + SVM | Binary Sentiment | Seminar Proposal")
