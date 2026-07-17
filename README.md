# Automated News Classification Using Machine Learning

An intelligent text classification system that automatically categorizes news articles into one of the 20 predefined categories from the 20 Newsgroups dataset using Machine Learning algorithms.

The project compares the performance of **Multinomial Naive Bayes** and **Support Vector Machine (Linear SVM)** using **TF-IDF Vectorization** for feature extraction.

---

##  Table of Contents
- Overview
- Features
- Dataset
- Project Architecture
- Machine Learning Pipeline
- Technologies Used
- Project Structure
- Results
- Sample Prediction
- Future Improvements
- Authors
- License
---

#  Overview

With the rapid growth of online news, manually organizing articles into categories has become increasingly difficult.

This project automates news categorization using Natural Language Processing (NLP) and Machine Learning techniques.

The workflow includes:

- Text preprocessing
- TF-IDF feature extraction
- Model training
- News category prediction
- Performance evaluation

Two machine learning models were implemented and compared:

- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

---

# Features

✔ Automatic news article classification

✔ TF-IDF text vectorization

✔ Multinomial Naive Bayes classifier

✔ Linear SVM classifier

✔ Classification Report

✔ Confusion Matrix visualization

✔ Accuracy comparison between models

✔ Easy-to-understand implementation using Scikit-Learn

---

# Dataset

**Dataset Used**

20 Newsgroups Dataset

- Approximately 18,000 news articles
- 20 different news categories
- Built into Scikit-Learn

The dataset is automatically downloaded using:

```python
from sklearn.datasets import fetch_20newsgroups
```

Example Categories

- Sports
- Politics
- Religion
- Space
- Computers
- Medicine
- Electronics
- Graphics
- Motorcycles
- Baseball

and many more...

---

# Project Architecture

```
                    News Articles
                          │
                          ▼
                 Text Preprocessing
                          │
                          ▼
                  TF-IDF Vectorization
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 Multinomial Naive Bayes          Linear SVM
          │                               │
          ▼                               ▼
      Prediction                     Prediction
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Model Performance Evaluation
          Accuracy • Classification Report
                Confusion Matrix
```

---

# Machine Learning Pipeline

### Step 1

Load the 20 Newsgroups dataset.

### Step 2

Split the dataset into

- Training Data
- Testing Data

### Step 3

Convert text into numerical vectors using

**TF-IDF Vectorizer**

### Step 4

Train

- Multinomial Naive Bayes
- Linear Support Vector Machine

### Step 5

Predict news categories.

### Step 6

Evaluate the models using

- Accuracy
- Classification Report
- Confusion Matrix

---

#  Technologies Used

| Technology         |Purpose                        |
|--------------------|-------------------------------|
| Python             |Programming Language           |
| Scikit-Learn       | Machine Learning              |
| NumPy              | Numerical Computing           |
| Matplotlib         | Data Visualization            |
| Seaborn            |Confusion Matrix Visualization |
| Pickle             | Saving Prediction Data        |

---

#  Project Structure

```
news-classification-ml/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── naive_model.py
│   ├── naive_analysis.py
│   ├── svm_model.py
│   └── svm_analysis.py
│
├── images/
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_svm.png
│   ├── architecture.png
│   └── sample_prediction.png
│
├── docs/
│   ├── Project_Report.pdf
│   └── Presentation.pptx
│
└── outputs/
    └── classification_reports.txt
```

---

# 📊 Results

| Model                   | Accuracy      |
|-------------------------|---------------|
| Multinomial Naive Bayes | **84.76%**    |
| Linear SVM              | **91.11%**    |

The Support Vector Machine achieved better overall performance than Naive Bayes in terms of accuracy and classification metrics.

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy(images/accuracy.png)
- Precision(images/precision.png)
- Recall(images/recall.png)
- F1 Score(images/f1_score.png)
- Classification Report(images/classification_report.png)

---

# Sample Prediction

Example Input

```
The team played exceptionally well and won the championship after a thrilling final match.
```

Predicted Category

```
rec.sport.baseball
```

---

# Confusion Matrix

### Naive Bayes

![Project Architecture](images/confusion_matrix_nb.png)

---

### Support Vector Machine
![Project Architecture](images/confusion_matrix_svm.png)

---

# Future Improvements

- Deep Learning using LSTM
- BERT-based News Classification
- RoBERTa
- Transformer Models
- Flask Web Application
- Streamlit Deployment
- REST API Integration
- Real-time News Classification

---
# Authors

**J. Manjiswari**B.Tech CSE (AI & ML) 

B.Tech CSE (AI & ML)

VIT-AP University

---

# License

This project is licensed under the MIT License.

---

 If you found this project useful, consider giving it a star.
