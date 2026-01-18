import streamlit as st
import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords

# ===============================
# Download NLTK stopwords
# ===============================
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
all_stopwords = stopwords_id.union(stopwords_en)

# ===============================
# Load model & vectorizer (opsional, tetap boleh ada)
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen Twitter (SVM)",
    layout="centered"
)

st.title("📊 Analisis Sentimen Twitter")
st.write(
    "Dashboard ini menampilkan hasil **analisis sentimen Twitter** "
    "menggunakan **Support Vector Machine (SVM)** "
    "pada tweet berbahasa **Indonesia dan Inggris**."
)

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_berlabel.csv")

df = load_dataset()

# ===============================
# Dataset Preview
# ===============================
st.subheader("📂 Contoh Data Tweet")

sample_size = st.slider(
    "Jumlah contoh data:",
    min_value=5,
    max_value=50,
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
# Statistik Sentimen
# ===============================
st.subheader("📈 Distribusi Sentimen Dataset")

sentiment_count = df["sentiment"].value_counts()
st.bar_chart(sentiment_count)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "📌 Metode: Support Vector Machine (SVM) | "
    "TF-IDF | Dataset Twitter Bahasa Indonesia & Inggris"
)
