import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen Twitter (SVM)",
    layout="centered"
)

# ===============================
# Download NLTK Stopwords (SAFE)
# ===============================
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
all_stopwords = stopwords_id.union(stopwords_en)

# ===============================
# Load Model & Vectorizer
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Text Preprocessing
# (HARUS sama dengan training)
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
# UI - Header
# ===============================
st.title("📊 Analisis Sentimen Twitter")
st.write(
    "Aplikasi ini menggunakan **Support Vector Machine (SVM)** "
    "untuk mengklasifikasikan sentimen tweet "
    "**Bahasa Indonesia dan Bahasa Inggris**."
)

# ===============================
# Input Tweet
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
        clean_text = preprocess_text(user_input)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]

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
# Contoh Tweet per Bahasa
# ===============================
st.subheader("📂 Contoh Tweet Berdasarkan Bahasa")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🇮🇩 Bahasa Indonesia")
    df_id = df[df["language"] == "id"]

    if len(df_id) > 0:
        sample_id = df_id.sample(
            min(5, len(df_id)),
            random_state=42
        )[["clean_text", "sentiment"]]

        st.dataframe(
            sample_id.rename(
                columns={
                    "clean_text": "Tweet",
                    "sentiment": "Sentimen"
                }
            ),
            use_container_width=True
        )
    else:
        st.info("Tidak ada data Bahasa Indonesia.")

with col2:
    st.markdown("### 🇬🇧 Bahasa Inggris")
    df_en = df[df["language"] == "en"]

    if len(df_en) > 0:
        sample_en = df_en.sample(
            min(5, len(df_en)),
            random_state=42
        )[["clean_text", "sentiment"]]

        st.dataframe(
            sample_en.rename(
                columns={
                    "clean_text": "Tweet",
                    "sentiment": "Sentimen"
                }
            ),
            use_container_width=True
        )
    else:
        st.info("Tidak ada data Bahasa Inggris.")

# ===============================
# Distribusi Sentimen
# ===============================
st.subheader("📈 Distribusi Sentimen Dataset")

sentiment_count = df["sentiment"].value_counts()
st.bar_chart(sentiment_count)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "📌 Model: Support Vector Machine (SVM) | TF-IDF | "
    "Multibahasa (Indonesia & Inggris)"
)
