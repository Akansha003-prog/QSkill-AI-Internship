"""
Spam Mail Detector
===================
Objective: Build a classifier that distinguishes between spam and
non-spam (ham) emails/SMS using textual data.

Steps followed (per project guideline):
 1. Load the messages and labels (spam or ham).
 2. Preprocess the text (lowercasing, remove stopwords, tokenization).
 3. Convert text into numeric features (Bag of Words or TF-IDF).
 4. Split into train/test sets.
 5. Train a simple model (Naive Bayes, Logistic Regression).
 6. Measure performance with accuracy, precision, or F1 score.

Skills gained: Text preprocessing, feature extraction, basic NLP,
classification.
"""

import re
import string
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -------------------------------------------------------------------
# 1. LOAD THE MESSAGES AND LABELS
# -------------------------------------------------------------------
df = pd.read_csv("spam_dataset.csv")
print("Dataset shape:", df.shape)
print(df["label"].value_counts())
print(df.head(), "\n")

# -------------------------------------------------------------------
# 2. PREPROCESS THE TEXT
#    - lowercasing
#    - remove punctuation / numbers
#    - tokenization
#    - remove stopwords
# -------------------------------------------------------------------
STOPWORDS = set(ENGLISH_STOP_WORDS)

def preprocess(text: str) -> str:
    text = text.lower()                                   # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)            # strip urls
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                        # strip numbers
    tokens = text.split()                                   # tokenization
    tokens = [t for t in tokens if t not in STOPWORDS]      # remove stopwords
    return " ".join(tokens)

df["clean_message"] = df["message"].apply(preprocess)
print("Example before/after preprocessing:")
print("RAW :", df["message"].iloc[0])
print("CLEAN:", df["clean_message"].iloc[0], "\n")

# Encode labels: spam = 1, ham = 0
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# -------------------------------------------------------------------
# 3. CONVERT TEXT INTO NUMERIC FEATURES (TF-IDF)
# -------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label_num"]
print("Feature matrix shape:", X.shape, "\n")

# -------------------------------------------------------------------
# 4. SPLIT INTO TRAIN / TEST SETS
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

# -------------------------------------------------------------------
# 5. TRAIN MODELS
# -------------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "y_pred": y_pred,
    }

# -------------------------------------------------------------------
# 6. EVALUATE PERFORMANCE
# -------------------------------------------------------------------
print("=" * 55)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 55)
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(f"Accuracy : {r['accuracy']:.4f}")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall   : {r['recall']:.4f}")
    print(f"F1 Score : {r['f1']:.4f}")
    print(classification_report(y_test, r["y_pred"], target_names=["ham", "spam"]))

# Pick best model by F1 score
best_name = max(results, key=lambda n: results[n]["f1"])
best_pred = results[best_name]["y_pred"]
print(f"\nBest performing model: {best_name}")

# -------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_pred)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title(f"Confusion Matrix - {best_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["ham", "spam"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["ham", "spam"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=axes[0], fraction=0.046)

# Bar chart comparing models
metric_names = ["accuracy", "precision", "recall", "f1"]
x = range(len(metric_names))
width = 0.35
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metric_names]
    axes[1].bar([p + i * width for p in x], vals, width, label=name)
axes[1].set_xticks([p + width / 2 for p in x])
axes[1].set_xticklabels(metric_names)
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Model Comparison")
axes[1].legend()

plt.tight_layout()
plt.savefig("spam_detector_results.png", dpi=150)
print("\nSaved chart: spam_detector_results.png")

# -------------------------------------------------------------------
# TRY IT ON NEW / UNSEEN MESSAGES
# -------------------------------------------------------------------
best_model = models[best_name]
sample_messages = [
    "Congratulations! You've won a free iPhone, click here to claim now",
    "Hey, are you coming to the party tonight?",
    "URGENT: Your account will be suspended, verify immediately",
    "Can you send me the notes from today's lecture?",
]
sample_clean = [preprocess(m) for m in sample_messages]
sample_vec = vectorizer.transform(sample_clean)
sample_pred = best_model.predict(sample_vec)

print("\n" + "=" * 55)
print("PREDICTIONS ON NEW MESSAGES")
print("=" * 55)
for msg, pred in zip(sample_messages, sample_pred):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label:4}] {msg}")
