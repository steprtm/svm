import streamlit as st
import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen SVM",
    page_icon="📊",
    layout="wide"
)

# ===============================
# DOWNLOAD NLTK (STREAMLIT SAFE)
# ===============================
nltk.download("punkt")
nltk.download("stopwords")

stopwords_id = set(stopwords.words("indonesian"))
stopwords_en = set(stopwords.words("english"))
stopwords_all = stopwords_id.union(stopwords_en)

# ===============================
# LOAD MODEL & DATA
# ===============================
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("dataset_berlabel.csv")

model, vectorizer = load_model()
df = load_data()

# ===============================
# PREPROCESS FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords_all]
    return " ".join(tokens)

# ===============================
# JUDUL
# ===============================
st.title("📊 Dashboard Analisis Sentimen Twitter (SVM)")
st.markdown("""
Aplikasi ini menggunakan **Support Vector Machine (SVM)**  
untuk menganalisis sentimen tweet **Bahasa Indonesia & Inggris**.
""")

# ===============================
# METRIC SENTIMEN
# ===============================
st.subheader("📈 Ringkasan Data Sentimen")

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df))
col2.metric("Positive", (df["sentiment"] == "positive").sum())
col3.metric("Negative", (df["sentiment"] == "negative").sum())

# ===============================
# INPUT TWEET
# ===============================
st.subheader("📝 Masukkan Teks Tweet")

st.markdown("**📌 Contoh Tweet:**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("🇮🇩 **Bahasa Indonesia**")
    contoh_id = "Pelayanan aplikasi ini sangat buruk dan membuat saya kecewa."
    st.code(contoh_id)

    if st.button("Gunakan Contoh Indonesia"):
        st.session_state["tweet_input"] = contoh_id

with col2:
    st.markdown("🇬🇧 **English**")
    contoh_en = "I really love this application, it works perfectly!"
    st.code(contoh_en)

    if st.button("Use English Example"):
        st.session_state["tweet_input"] = contoh_en

user_input = st.text_area(
    "Tulis tweet di sini:",
    height=120,
    key="tweet_input",
    placeholder="Contoh: I really love this new technology!"
)

# ===============================
# PREDIKSI
# ===============================
if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Masukkan teks tweet terlebih dahulu.")
    else:
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success("✅ Sentimen: POSITIVE")
        else:
            st.error("❌ Sentimen: NEGATIVE")

# ===============================
# CONTOH DATASET
# ===============================
st.subheader("📄 Contoh Data Hasil Preprocessing")

sample_df = df.sample(
    min(5, len(df)), random_state=42
)[["text", "sentiment"]]

st.dataframe(sample_df)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "📌 **Metode:** Support Vector Machine (SVM) | "
    "📂 Data: Twitter (X) | "
    "🎓 Teknik Informatika"
)
