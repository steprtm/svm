import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords

# ===============================
# DOWNLOAD STOPWORDS (AMAN CLOUD)
# ===============================
nltk.download("stopwords")

stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))
all_stopwords = stop_id.union(stop_en)

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# PREPROCESS TEXT (SAMA DGN TRAINING)
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
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen Twitter (SVM)",
    layout="centered"
)

st.title("📊 Analisis Sentimen Twitter Menggunakan SVM")

st.markdown(
    """
Aplikasi ini digunakan untuk menganalisis sentimen publik  
terhadap isu **Donald Trump dan Greenland** pada platform **X (Twitter)**  
menggunakan metode **Support Vector Machine (SVM)**.
"""
)

# ===============================
# INPUT PREDIKSI
# ===============================
st.subheader("📝 Prediksi Sentimen Tweet")

user_input = st.text_area(
    "Masukkan teks tweet:",
    height=120,
    placeholder="Contoh: I think this policy will affect Greenland badly."
)

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        clean_text = preprocess_text(user_input)
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success("✅ Sentimen: POSITIF")
        elif prediction == "negative":
            st.error("❌ Sentimen: NEGATIF")
        else:
            st.info("⚖️ Sentimen: NETRAL")

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_berlabel.csv")

df = load_dataset()

# ===============================
# DATASET PREVIEW
# ===============================
st.subheader("📂 Contoh Data Tweet")

sample_size = st.slider(
    "Jumlah data yang ditampilkan:",
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
# DISTRIBUSI SENTIMEN
# ===============================
st.subheader("📈 Distribusi Sentimen Dataset")

sentiment_count = df["sentiment"].value_counts()
st.bar_chart(sentiment_count)

# ===============================
# INFORMASI MODEL
# ===============================
st.subheader("ℹ️ Informasi Model")

st.markdown(
    f"""
- **Algoritma** : Support Vector Machine (LinearSVC)  
- **Fitur** : TF-IDF (Unigram & Bigram)  
- **Jumlah Data** : {len(df)}  
- **Kelas Sentimen** : Positive, Negative, Neutral  
- **Akurasi Model** : ± **69%**
"""
)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "Model dilatih menggunakan data Twitter berbahasa Indonesia dan Inggris | "
    "Penelitian Mahasiswa Teknik Informatika Semester 7"
)
