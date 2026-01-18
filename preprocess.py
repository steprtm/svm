# =========================================
# PREPROCESS + SENTIMENT CLASSIFICATION
# ID & EN | POSITIVE - NEGATIVE - NEUTRAL
# METHOD : TF-IDF + SVM (LinearSVC)
# =========================================

import pandas as pd
import re
import nltk
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================
# NLTK DOWNLOAD
# =========================================
nltk.download("stopwords")

# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv(r"C:\sempro\dataset\dataraw.csv")

# Pastikan kolom teks
df = df.rename(columns={"full_text": "text"})

# Ambil hanya Bahasa Indonesia & Inggris
df = df[df["lang"].isin(["id", "en"])]

# Hapus data kosong
df = df.dropna(subset=["text"])

print("Jumlah data awal:", len(df))

# =========================================
# STOPWORDS
# =========================================
stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))

# =========================================
# CLEANING FUNCTION
# =========================================
def clean_text(text, lang):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()

    if lang == "id":
        tokens = [w for w in tokens if w not in stop_id]
    else:
        tokens = [w for w in tokens if w not in stop_en]

    return " ".join(tokens)

df["clean_text"] = df.apply(
    lambda x: clean_text(x["text"], x["lang"]),
    axis=1
)

# =========================================
# SENTIMENT LABELING
# =========================================
analyzer = SentimentIntensityAnalyzer()

# Lexicon sederhana Bahasa Indonesia
lexicon_id = {
    "bagus": 1,
    "baik": 1,
    "senang": 1,
    "suka": 1,
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

    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df.apply(
    lambda x: label_sentiment(x["clean_text"], x["lang"]),
    axis=1
)

print("\nDistribusi Sentimen:")
print(df["sentiment"].value_counts())

# =========================================
# TF-IDF FEATURE EXTRACTION
# =========================================
X = df["clean_text"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

# =========================================
# TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# TRAIN SVM MODEL (MULTICLASS)
# =========================================
model = LinearSVC()
model.fit(X_train, y_train)

# =========================================
# MODEL EVALUATION
# =========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="macro"
)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================
# SAVE METRICS (UNTUK STREAMLIT)
# =========================================
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open(r"C:\sempro\dataset\model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# =========================================
# SAVE OUTPUT FILES
# =========================================
df.to_csv(r"C:\sempro\dataset\dataset_berlabel.csv", index=False)
joblib.dump(model, r"C:\sempro\dataset\svm_model.pkl")
joblib.dump(vectorizer, r"C:\sempro\dataset\tfidf_vectorizer.pkl")

print("\nPreprocessing & Training selesai.")
print("File tersimpan:")
print("- dataset_berlabel.csv")
print("- svm_model.pkl")
print("- tfidf_vectorizer.pkl")
print("- model_metrics.json")
