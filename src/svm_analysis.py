import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("SVM Analysis started...\n")

# Load saved data
y_test = pickle.load(open("y_test_svm.pkl", "rb"))
y_pred = pickle.load(open("y_pred_svm.pkl", "rb"))
X_test = pickle.load(open("X_test_svm.pkl", "rb"))
data = pickle.load(open("data_svm.pkl", "rb"))

# 1. Accuracy

accuracy = np.mean(y_test == y_pred)
print("1. Accuracy:", accuracy)


# 2. Classification Report

print("\n2. Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 3. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("3. Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix_svm.png")
plt.show()   # 👈 ADD THIS LINE

print("\nConfusion matrix saved as image!")
