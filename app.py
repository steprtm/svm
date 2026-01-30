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
df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
)

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
# RINGKASAN DATASET
# ===============================
st.subheader("📌 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tweet", len(df))
col2.metric("Positive", (df["sentiment"] == "positive").sum())
col3.metric("Negative", (df["sentiment"] == "negative").sum())
col4.metric("Neutral", (df["sentiment"] == "neutral").sum())

# ===============================
# DISTRIBUSI SENTIMEN
# ===============================
st.subheader("📈 Distribusi Sentimen")
st.bar_chart(df["sentiment"].value_counts())

# ===============================
# CONTOH DATA ACAK
# ===============================
st.subheader("📂 Contoh Tweet (Random)")

sample_size = st.slider(
    "Jumlah contoh tweet:",
    min_value=5,
    max_value=100,
    value=10
)

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
    if len(df_pos) > 0:
        st.dataframe(
            df_pos.sample(min(10, len(df_pos)), random_state=1)[["clean_text"]],
            use_container_width=True
        )

with col_neg:
    st.markdown("### 😡 Negative")
    df_neg = df[df["sentiment"] == "negative"]
    if len(df_neg) > 0:
        st.dataframe(
            df_neg.sample(min(10, len(df_neg)), random_state=1)[["clean_text"]],
            use_container_width=True
        )

with col_neu:
    st.markdown("### 😐 Neutral")
    df_neu = df[df["sentiment"] == "neutral"]
    if len(df_neu) > 0:
        st.dataframe(
            df_neu.sample(min(10, len(df_neu)), random_state=1)[["clean_text"]],
            use_container_width=True
        )

# ===============================
# ANALISIS PANJANG TWEET
# ===============================
st.subheader("📏 Analisis Panjang Tweet")

df["tweet_length"] = df["clean_text"].astype(str).apply(len)

col_len1, col_len2 = st.columns(2)

col_len1.metric(
    "Rata-rata Panjang Tweet",
    f"{df['tweet_length'].mean():.1f} karakter"
)

col_len2.metric(
    "Tweet Terpanjang",
    f"{df['tweet_length'].max()} karakter"
)

st.bar_chart(
    df["tweet_length"].value_counts().sort_index().head(50)
)

# ===============================
# EVALUASI MODEL
# ===============================
st.subheader("🎯 Evaluasi Model SVM")

metrics_path = "model_metrics.json"

if not os.path.exists(metrics_path):
    st.error("File model_metrics.json tidak ditemukan. Jalankan preprocess.py terlebih dahulu.")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision (Avg)", f"{metrics['precision']*100:.2f}%")
col3.metric("Recall (Avg)", f"{metrics['recall']*100:.2f}%")
col4.metric("F1-Score (Avg)", f"{metrics['f1_score']*100:.2f}%")

st.info(
    "Evaluasi model dilakukan menggunakan data uji (test set) "
    "dengan metrik Accuracy, Precision, Recall, dan F1-score "
    "(macro average) menggunakan metode SVM berbasis TF-IDF."
)

# ===============================
# TESTING SENTIMEN MANUAL
# ===============================
st.subheader("🧠 Uji Sentimen Teks Secara Langsung")

user_input = st.text_area(
    "Masukkan teks / tweet:",
    placeholder="Contoh: Donald Trump comments on Greenland policy...",
    height=120
)

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        clean_input = preprocess_input(user_input)
        vectorized_input = tfidf_vectorizer.transform([clean_input])
        prediction = svm_model.predict(vectorized_input)[0]

        if prediction == "positive":
            st.success("😊 Sentimen: POSITIF")
        elif prediction == "negative":
            st.error("😡 Sentimen: NEGATIF")
        else:
            st.info("😐 Sentimen: NETRAL")

        st.caption(f"Teks setelah preprocessing: `{clean_input}`")

# ===============================
# TOP KATA SERING MUNCUL
# ===============================
st.subheader("🔤 Top 15 Kata Paling Sering Muncul")

def get_top_words(text_series, n=15):
    words = []
    for text in text_series.dropna():
        tokens = text.split()
        tokens = [t for t in tokens if t not in all_stopwords and len(t) > 3]
        words.extend(tokens)
    return Counter(words).most_common(n)

top_words = get_top_words(df["clean_text"])
top_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

st.dataframe(top_df, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "📌 Metode: Support Vector Machine (SVM) | TF-IDF | "
    "Analisis Sentimen Twitter | Dashboard Seminar Proposal"
)
