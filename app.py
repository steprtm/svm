import streamlit as st
import pandas as pd
import nltk
import re
from collections import Counter

from nltk.corpus import stopwords

# ===============================
# NLTK setup
# ===============================
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
all_stopwords = stopwords_id.union(stopwords_en)

# ===============================
# Streamlit config
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Twitter (SVM)",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen Twitter")
st.write(
    "Dashboard ini menampilkan hasil **analisis sentimen Twitter** "
    "menggunakan **Support Vector Machine (SVM)** berbasis **TF-IDF**. "
    "Fokus utama dashboard adalah **eksplorasi dataset dan hasil klasifikasi**."
)

# ===============================
# Load dataset
# ===============================
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_berlabel.csv")

df = load_dataset()

# ===============================
# Pre-check kolom
# ===============================
required_cols = {"clean_text", "sentiment"}
if not required_cols.issubset(df.columns):
    st.error("Dataset tidak memiliki kolom yang dibutuhkan.")
    st.stop()

# ===============================
# RINGKASAN DATASET
# ===============================
st.subheader("📌 Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Tweet", len(df))
col2.metric("Positive", (df["sentiment"] == "positive").sum())
col3.metric("Negative", (df["sentiment"] == "negative").sum())

# ===============================
# DISTRIBUSI SENTIMEN
# ===============================
st.subheader("📈 Distribusi Sentimen")

sentiment_count = df["sentiment"].value_counts()
st.bar_chart(sentiment_count)

# ===============================
# CONTOH DATA ACAK
# ===============================
st.subheader("📂 Contoh Tweet (Random)")

sample_size = st.slider(
    "Jumlah contoh tweet:",
    min_value=5,
    max_value=30,
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

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("### 😊 Positive")
    df_pos = df[df["sentiment"] == "positive"]
    if len(df_pos) > 0:
        st.dataframe(
            df_pos.sample(min(5, len(df_pos)), random_state=1)[["clean_text"]],
            use_container_width=True
        )

with col_neg:
    st.markdown("### 😡 Negative")
    df_neg = df[df["sentiment"] == "negative"]
    if len(df_neg) > 0:
        st.dataframe(
            df_neg.sample(min(5, len(df_neg)), random_state=1)[["clean_text"]],
            use_container_width=True
        )

# ===============================
# PANJANG TWEET (EDA)
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
# TOP KATA PALING SERING MUNCUL
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
    "Analisis Sentimen Twitter | Dashboard Sempro"
)
