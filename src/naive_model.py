from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
data = fetch_20newsgroups(subset='all')
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)

# Test input
sample = ["The team played very well and won the match"]
sample_tfidf = vectorizer.transform(sample)
prediction = model.predict(sample_tfidf)

print("Predicted Category:", data.target_names[prediction[0]])
import pickle

pickle.dump(y_test, open("y_test.pkl", "wb"))
pickle.dump(y_pred, open("y_pred.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(data, open("data.pkl", "wb"))

print("All data saved successfully!")
