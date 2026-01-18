import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

# ===============================
# LOAD RESOURCE
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

# ===============================
# TEXT PREPROCESSING
# ===============================
nltk.download("stopwords")

stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))
stop_all = stop_id.union(stop_en)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_all]

    return " ".join(tokens)

# ===============================
# LOAD DATA & MODEL
# ===============================
model, vectorizer = load_model()
df = load_data()

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard", "Prediksi Tweet"]
)

# ===============================
# DASHBOARD
# ===============================
if menu == "Dashboard":

    st.title("📊 Dashboard Analisis Sentimen Publik")
    st.caption("Isu Donald Trump dan Greenland | Metode SVM")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positive", (df["sentiment"] == "positive").sum())
    col3.metric("Negative", (df["sentiment"] == "negative").sum())

    st.subheader("📈 Distribusi Sentimen")

    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Jumlah Tweet")
    st.pyplot(fig)

    st.subheader("📝 Contoh Tweet")

    col_id, col_en = st.columns(2)

    with col_id:
        st.markdown("**🇮🇩 Bahasa Indonesia**")
        id_sample = df[df["language"] == "id"].sample(1)
        st.write(id_sample["text"].values[0])
        st.caption(f"Sentimen: **{id_sample['sentiment'].values[0]}**")

    with col_en:
        st.markdown("**🇬🇧 Bahasa Inggris**")
        en_sample = df[df["language"] == "en"].sample(1)
        st.write(en_sample["text"].values[0])
        st.caption(f"Sentimen: **{en_sample['sentiment'].values[0]}**")

# ===============================
# PREDIKSI TWEET
# ===============================
elif menu == "Prediksi Tweet":

    st.title("🔍 Prediksi Sentimen Tweet")

    user_input = st.text_area(
        "Masukkan teks tweet (Bahasa Indonesia / Inggris)",
        height=150
    )

    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            st.success(f"📌 **Hasil Sentimen: {prediction.upper()}**")
