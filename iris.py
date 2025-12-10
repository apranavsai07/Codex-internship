# iris_classification.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---- 1. Load the dataset ----
iris = datasets.load_iris()

X = iris.data  # features (sepal length, sepal width, petal length, petal width)
y = iris.target  # labels (0,1,2)
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
feature_names = iris.feature_names

# Convert to DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[label] for label in y]

print("First 5 rows of the dataset:")
print(df.head())
print("\nClass distribution:")
print(df["species"].value_counts())

# ---- 2. Visual Exploration (Scatter plot / Histograms) ----

# Pairplot of all features colored by species
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Dataset - Pairplot by Species", y=1.02)
plt.show()

# Histograms for each feature
df[feature_names].hist(figsize=(8, 6))
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()

# ---- 3. Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 20% test data
    random_state=42,     # for reproducibility
    stratify=y           # keep class ratio same in train and test
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ---- 4. Preprocessing (Scaling) ----
# Not strictly necessary for iris, but good practice for models like KNN

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 5. Train a Classifier (K-Nearest Neighbors) ----

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# ---- 6. Evaluate the Model ----

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (k={k}): {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN on Iris Dataset")
plt.show()
