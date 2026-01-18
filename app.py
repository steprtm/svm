import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen X",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen Publik di Platform X")
st.markdown("""
Analisis sentimen publik terhadap isu **Donald Trump dan Greenland**
menggunakan metode **Support Vector Machine (SVM)**.
""")

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_berlabel.csv")

@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔧 Pengaturan")
show_raw = st.sidebar.checkbox("Tampilkan data mentah")
sample_size = st.sidebar.slider("Jumlah sampel tweet", 10, 200, 50)

# =========================
# METRIC RINGKASAN
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Tweet", len(df))
col2.metric("Sentimen Positif", (df["sentiment"] == "positive").sum())
col3.metric("Sentimen Negatif", (df["sentiment"] == "negative").sum())

# =========================
# VISUALISASI DISTRIBUSI
# =========================
st.subheader("📈 Distribusi Sentimen")

sentiment_count = df["sentiment"].value_counts()

fig, ax = plt.subplots()
sentiment_count.plot(
    kind="bar",
    ax=ax
)
ax.set_xlabel("Sentimen")
ax.set_ylabel("Jumlah Tweet")
ax.set_title("Distribusi Sentimen Publik")

st.pyplot(fig)

# =========================
# TABEL DATA
# =========================
st.subheader("📝 Contoh Tweet")

sample_df = df.sample(sample_size, random_state=42)[
    ["full_text", "sentiment"]
]

st.dataframe(sample_df, use_container_width=True)

# =========================
# OPSIONAL: DATA MENTAH
# =========================
if show_raw:
    st.subheader("📂 Data Lengkap")
    st.dataframe(df, use_container_width=True)

# =========================
# PREDIKSI TEKS BARU
# =========================
st.subheader("🔮 Uji Prediksi Sentimen")

user_input = st.text_area(
    "Masukkan teks tweet (Bahasa Indonesia / Inggris):",
    height=100
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]

        if prediction == "positive":
            st.success("✅ Sentimen: POSITIF")
        else:
            st.error("❌ Sentimen: NEGATIF")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Dashboard ini dikembangkan untuk keperluan Seminar Proposal "
    "dengan fokus pada analisis metode Support Vector Machine (SVM)."
)
