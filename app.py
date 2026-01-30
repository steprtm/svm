import streamlit as st
import pandas as pd
import nltk
import re
import json
import joblib
import os
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

st.title("📊 Dashboard Analisis Sentimen Twitter")
st.write(
    "Dashboard ini menampilkan hasil **analisis sentimen Twitter (Platform X)** "
    "menggunakan **Support Vector Machine (SVM)** berbasis **TF-IDF**. "
    "Dashboard juga menyediakan fitur **uji sentimen secara langsung**."
)

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_berlabel.csv")

df = load_dataset()

# ===============================
# NORMALISASI LABEL
# ===============================
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

# ===============================
# VALIDASI KOLOM
# ===============================
required_cols = {"clean_text", "sentiment"}
if not required_cols.issubset(df.columns):
    st.error("Dataset tidak memiliki kolom yang dibutuhkan.")
    st.stop()

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

svm_model, tfidf_vectorizer = load_model()

# ===============================
# PREPROCESS INPUT USER
# ===============================
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# 📌 RINGKASAN DATASET + PERSENTASE
# ===============================
st.subheader("📌 Ringkasan Dataset")

total = len(df)
pos = (df["sentiment"] == "positive").sum()
neg = (df["sentiment"] == "negative").sum()
neu = (df["sentiment"] == "neutral").sum()

pos_pct = (pos / total) * 100
neg_pct = (neg / total) * 100
neu_pct = (neu / total) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tweet", total)
col2.metric("Positive", f"{pos} ({pos_pct:.2f}%)")
col3.metric("Negative", f"{neg} ({neg_pct:.2f}%)")
col4.metric("Neutral", f"{neu} ({neu_pct:.2f}%)")

# ===============================
# DISTRIBUSI SENTIMEN
# ===============================
st.subheader("📈 Distribusi Sentimen")
st.bar_chart(df["sentiment"].value_counts())

# ===============================
# CONTOH DATA ACAK
# ===============================
st.subheader("📂 Contoh Tweet (Random)")

sample_size = st.slider("Jumlah contoh tweet:", 5, 100, 10)

sample_df = df.sample(sample_size, random_state=42)[
    ["clean_text", "sentiment"]
]

st.dataframe(
    sample_df.rename(
        columns={
            "clean_text": "Tweet (Hasil Preprocessing)",
            "sentiment": "Sentimen"
        }
    ),
    use_container_width=True
)

# ===============================
# CONTOH TWEET PER SENTIMEN
# ===============================
st.subheader("🧪 Contoh Tweet Berdasarkan Sentimen")

col_pos, col_neg, col_neu = st.columns(3)

with col_pos:
    st.markdown("### 😊 Positive")
    df_pos = df[df["sentiment"] == "positive"]
    st.dataframe(df_pos.sample(min(10, len(df_pos)))[["clean_text"]])

with col_neg:
    st.markdown("### 😡 Negative")
    df_neg = df[df["sentiment"] == "negative"]
    st.dataframe(df_neg.sample(min(10, len(df_neg)))[["clean_text"]])

with col_neu:
    st.markdown("### 😐 Neutral")
    df_neu = df[df["sentiment"] == "neutral"]
    st.dataframe(df_neu.sample(min(10, len(df_neu)))[["clean_text"]])

# ===============================
# ANALISIS PANJANG TWEET
# ===============================
st.subheader("📏 Analisis Panjang Tweet")

df["tweet_length"] = df["clean_text"].astype(str).apply(len)

col_len1, col_len2 = st.columns(2)
col_len1.metric("Rata-rata Panjang Tweet", f"{df['tweet_length'].mean():.1f}")
col_len2.metric("Tweet Terpanjang", df["tweet_length"].max())

# ===============================
# EVALUASI MODEL
# ===============================
st.subheader("🎯 Evaluasi Model SVM")

metrics_path = "model_metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")

# ===============================
# 🧠 UJI SENTIMEN MANUAL + PERSENTASE
# ===============================
st.subheader("🧠 Uji Sentimen Teks Secara Langsung")

user_input = st.text_area("Masukkan teks / tweet:", height=120)

if st.button("🔍 Prediksi Sentimen"):
    clean_input = preprocess_input(user_input)
    vector = tfidf_vectorizer.transform([clean_input])

    probs = svm_model.decision_function(vector)[0]
    exp_probs = pd.Series(probs).rank(pct=True)

    labels = svm_model.classes_
    prob_df = pd.DataFrame({
        "Sentimen": labels,
        "Persentase (%)": (exp_probs.values * 100)
    })

    st.dataframe(prob_df, use_container_width=True)

    pred = svm_model.predict(vector)[0]
    st.success(f"Sentimen Dominan: **{pred.upper()}**")

# ===============================
# TOP KATA
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
st.caption("📌 Analisis Sentimen Twitter | TF-IDF + SVM | Seminar Proposal")
