import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ===============================
# Download NLTK stopwords
# ===============================
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
all_stopwords = stopwords_id.union(stopwords_en)

# ===============================
# Load model & vectorizer
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Text preprocessing (SAMA dgn training)
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in all_stopwords]
    return " ".join(tokens)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen Twitter (SVM)",
    layout="centered"
)

st.title("📊 Analisis Sentimen Twitter")
st.write(
    "Aplikasi ini menggunakan **Support Vector Machine (SVM)** "
    "untuk mengklasifikasikan sentimen tweet berbahasa "
    "**Indonesia dan Inggris**."
)

# ===============================
# Input Text
# ===============================
st.subheader("📝 Masukkan Teks Tweet")

user_input = st.text_area(
    "Tulis tweet di sini:",
    height=120,
    placeholder="Contoh: I really love this new technology!"
)

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        clean_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([clean_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == "positive":
            st.success("✅ Sentimen: POSITIVE")
        elif prediction == "negative":
            st.error("❌ Sentimen: NEGATIVE")
        else:
            st.info("⚖️ Sentimen: NEUTRAL")

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
    "📌 Model: Support Vector Machine (SVM) | "
    "TF-IDF | Bahasa Indonesia & Inggris"
)
