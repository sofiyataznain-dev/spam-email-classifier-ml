# Spam Email Classification using Naive Bayes

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "message": [
        "Congratulations! You won a free prize",
        "Call now to claim your reward",
        "Hey, are we meeting today?",
        "Please find the report attached",
        "Win cash now!!!",
        "Let's have lunch tomorrow",
        "Urgent! You have won a lottery",
        "Can you send me the notes?"
    ],
    "label": [
        "spam", "spam", "ham", "ham",
        "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)

# Convert labels to binary
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Features and target
X = df["message"]
y = df["label"]

# Text vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.25, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Spam Email Classifier")
print("Accuracy:", round(accuracy, 2))
