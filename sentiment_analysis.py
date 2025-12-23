# -------------------------------------------------
# 1️⃣  Imports
# -------------------------------------------------
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 2️⃣  Load data
# -------------------------------------------------
# Replace "reviews.csv" with your file and adjust column names if needed
df = pd.read_csv("reviews.csv")                     # expects columns: "text", "sentiment"
# Drop any rows where the label is missing
df = df.dropna(subset=['sentiment'])
# Ensure the label is integer (will raise an error if something else is present)
df['sentiment'] = df['sentiment'].astype(int)

X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

print("Sample rows:")
print(df.head())

# -------------------------------------------------
# 3️⃣  Basic preprocessing
# -------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(txt):
    # lower‑case
    txt = txt.lower()
    # remove URLs, mentions, hashtags (keep the word)
    txt = re.sub(r"http\S+|www.\S+", " ", txt)
    txt = re.sub(r"@\w+", " ", txt)
    txt = re.sub(r"#(\w+)", r"\1", txt)
    # keep only alphabetic characters
    txt = re.sub("[^a-zA-Z]", " ", txt)
    # tokenise, remove stop‑words, lemmatise
    tokens = [lemmatizer.lemmatize(w) for w in txt.split() if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------------------------
# 4️⃣  Train‑test split
# -------------------------------------------------
X = df["clean_text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------------------
# 5️⃣  Vectorisation (TF‑IDF)
# -------------------------------------------------
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# -------------------------------------------------
# 6️⃣  Model – Logistic Regression (fast & interpretable)
# -------------------------------------------------
clf = LogisticRegression(max_iter=500, n_jobs=-1, random_state=42)
clf.fit(X_train_vec, y_train)

# -------------------------------------------------
# 7️⃣  Evaluation
# -------------------------------------------------
y_pred = clf.predict(X_test_vec)
y_prob = clf.predict_proba(X_test_vec)[:,1]

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("F1‑score :", f1_score(y_test, y_pred))
print("ROC‑AUC  :", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------
# 8️⃣  Insight: most informative features
# -------------------------------------------------
feature_names = tfidf.get_feature_names_out()
coeffs = clf.coef_[0]
top_pos = sorted(zip(coeffs, feature_names), reverse=True)[:10]
top_neg = sorted(zip(coeffs, feature_names))[:10]

print("\nTop 10 positive cues:")
for c, w in top_pos:
    print(f"{w}: {c:.3f}")

print("\nTop 10 negative cues:")
for c, w in top_neg:
    print(f"{w}: {c:.3f}")

# -------------------------------------------------
# 9️⃣  Visualise confusion matrix (optional)
# -------------------------------------------------
cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()