import pandas as pd
import pickle
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"<.*?>", " ", text)                    # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


# ── Load & prepare ───────────────────────────────────────────────────────────
print("Loading datasets...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0   # 0 = FAKE
true_df["label"] = 1   # 1 = TRUE

df = pd.concat([fake_df, true_df], ignore_index=True)

# Combine title + text; fill missing values
df["content"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)
df["content"] = df["content"].apply(clean_text)

# Drop empty rows
df = df[df["content"].str.strip() != ""].reset_index(drop=True)

print(f"Total samples: {len(df)}  |  FAKE: {(df.label==0).sum()}  |  TRUE: {(df.label==1).sum()}")

X = df["content"]
y = df["label"]

# ── Vectorizer ───────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,          # more features = better coverage
    ngram_range=(1, 2),          # unigrams + bigrams
    sublinear_tf=True,           # log-scale TF to reduce impact of common words
    min_df=2,                    # ignore very rare terms
)

X_vectorized = vectorizer.fit_transform(X)

# ── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model ────────────────────────────────────────────────────────────────────
print("Training model...")
model = LogisticRegression(
    max_iter=1000,
    C=5.0,                       # regularisation strength (tuned)
    solver="lbfgs",
    class_weight="balanced",     # handles any class imbalance
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "TRUE"]))

# ── Enhancements summary ─────────────────────────────────────────────────────
enhancements = [
    "Text cleaning: lowercasing, URL/HTML removal, special char stripping",
    "TF-IDF: 50,000 features (up from 10,000)",
    "TF-IDF: bigrams enabled (ngram_range 1-2)",
    "TF-IDF: sublinear_tf=True (log-scale term frequency)",
    "TF-IDF: min_df=2 (ignores very rare terms)",
    "LogisticRegression: C=5.0 (tuned regularisation)",
    "LogisticRegression: class_weight=balanced (handles imbalance)",
    "LogisticRegression: stratified train/test split",
    "Dataset: saved cleaned pickle (dataset_clean.pkl) for fast similarity search",
]
print(f"\n{'='*55}")
print(f"  Total enhancements applied: {len(enhancements)}")
print(f"{'='*55}")
for i, e in enumerate(enhancements, 1):
    print(f"  {i:02d}. {e}")
print(f"{'='*55}\n")

# ── Save ─────────────────────────────────────────────────────────────────────
pickle.dump(model,      open("model.pkl",      "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Also save cleaned dataset for similarity search in app.py
df[["content", "label", "title"]].to_pickle("dataset_clean.pkl")

print("\nSaved: model.pkl | vectorizer.pkl | dataset_clean.pkl")