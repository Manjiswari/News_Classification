from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

print("Training SVM model...\n")

# Load dataset
data = fetch_20newsgroups(subset='all')
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM Model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)

# Save all required data
pickle.dump(y_test, open("y_test_svm.pkl", "wb"))
pickle.dump(y_pred, open("y_pred_svm.pkl", "wb"))
pickle.dump(X_test, open("X_test_svm.pkl", "wb"))
pickle.dump(data, open("data_svm.pkl", "wb"))

print("\nAll SVM data saved successfully!")
