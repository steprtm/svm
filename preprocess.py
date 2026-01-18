# ================================
# PREPROCESS + SENTIMENT LABELING
# ID + EN USING SVM
# ================================

import pandas as pd
import re
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================================
# DOWNLOAD NLTK
# ================================
nltk.download("stopwords")

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(r"C:\sempro\dataset\dataraw.csv")

# Gunakan kolom teks yang benar
df = df.rename(columns={"full_text": "text"})

# Filter bahasa Indonesia & Inggris
df = df[df["lang"].isin(["id", "en"])]

# Drop data kosong
df = df.dropna(subset=["text"])

print("Jumlah data ID + EN:", len(df))

# ================================
# STOPWORDS
# ================================
stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))

# ================================
# CLEANING FUNCTION
# ================================
def clean_text(text, lang):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()

    if lang == "id":
        tokens = [w for w in tokens if w not in stop_id]
    else:
        tokens = [w for w in tokens if w not in stop_en]

    return " ".join(tokens)

df["clean_text"] = df.apply(
    lambda x: clean_text(x["text"], x["lang"]), axis=1
)

# ================================
# SENTIMENT LABELING
# ================================
analyzer = SentimentIntensityAnalyzer()

# Lexicon sederhana Bahasa Indonesia
lexicon_id = {
    "bagus": 1,
    "baik": 1,
    "senang": 1,
    "buruk": -1,
    "jelek": -1,
    "benci": -1,
    "jahat": -1
}

def label_sentiment(text, lang):
    if lang == "id":
        score = sum(lexicon_id.get(word, 0) for word in text.split())
    else:
        score = analyzer.polarity_scores(text)["compound"]

    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df.apply(
    lambda x: label_sentiment(x["clean_text"], x["lang"]), axis=1
)

print(df["sentiment"].value_counts())

# ================================
# DROP NEUTRAL (AGAR SVM AMAN)
# ================================
df = df[df["sentiment"] != "neutral"]
print("Jumlah data setelah drop neutral:", len(df))

# ================================
# TF-IDF VECTOR
# ================================
X = df["clean_text"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# ================================
# TRAIN SVM
# ================================
model = LinearSVC()
model.fit(X_train, y_train)

# ================================
# EVALUATION
# ================================
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ================================
# SAVE OUTPUT
# ================================
df.to_csv("dataset_berlabel.csv", index=False)
joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nPreprocessing selesai.")
print("File tersimpan:")
print("- dataset_berlabel.csv")
print("- svm_model.pkl")
print("- tfidf_vectorizer.pkl")
