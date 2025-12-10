# Machine Learning Mini Projects

This repository contains two fundamental Machine Learning projects implemented using Python and Scikit-learn. These projects demonstrate core concepts such as data preprocessing, visualization, feature engineering, and model evaluation.

---

## 1️⃣ Iris Flower Classification

The Iris dataset is a classic multiclass classification problem involving three species of flowers:
- Setosa  
- Versicolor  
- Virginica  

Each flower is described using four numerical features:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

### 🔧 Steps Performed
- Loaded the Iris dataset from Scikit-learn  
- Converted it into a pandas DataFrame for easy visualization  
- Explored data using pairplots and histograms  
- Scaled the features using **StandardScaler**  
- Trained a **K-Nearest Neighbors (KNN)** classifier  
- Evaluated using:
  - Accuracy score  
  - Classification report  
  - Confusion matrix  

### 🎯 Outcome  
The model accurately distinguishes between the three flower species, demonstrating how distance-based algorithms like KNN can be effective for structured numerical data.

---

## 2️⃣ SMS Spam Detection (NLP)

This project classifies SMS messages into:
- **Spam**
- **Ham (not spam)**

It uses Natural Language Processing techniques to convert text into numerical features and trains a machine learning model to detect unwanted messages.

### 🔧 Steps Performed
- Loaded **spam.csv** dataset  
- Cleaned and preprocessed text data  
- Converted text into numerical features using **TF-IDF Vectorization**  
- Split data into training/testing sets  
- Trained a **Multinomial Naive Bayes** classifier  
- Evaluated using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
- Visualized:
  - Class distribution  
  - Confusion matrix  
  - Performance metrics graph  

### 🎯 Outcome  
The model successfully identifies spam patterns in text messages, showing how NLP + machine learning can filter real-world communication effectively.

---

## 🛠 Tools Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 📌 Summary

Both projects demonstrate essential Machine Learning workflows:

| Project | Type | Algorithm | Skills Learned |
|--------|------|-----------|----------------|
| Iris Flower Classification | Structured Data | KNN | Scaling, visualization, classification |
| SMS Spam Detection | NLP/Text Data | Naive Bayes | Text preprocessing, TF-IDF, evaluation |

These projects form a strong foundation for understanding both **classical ML** and **NLP-based ML**.

