
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATASET_PATH = "dataset/phishing_url_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=["URL", "Label"])
df["URL"] = df["URL"].astype(str).str.strip()
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"])
df = df.drop_duplicates(subset=["URL"])
df = df[df["Label"].isin([0, 1])]

X = df["URL"]
y = df["Label"].astype(int)

print("\nDataset records:", len(df))
print("Label counts:")
print(y.value_counts().sort_index())
print("Label mapping: 0 = Phishing, 1 = Legitimate")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n========== PHISHGUARD MODEL RESULTS ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
))

print("Confusion Matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

joblib.dump(model, "phishguard_model.pkl")

print("\nModel saved as phishguard_model.pkl")
print("Training complete!")
