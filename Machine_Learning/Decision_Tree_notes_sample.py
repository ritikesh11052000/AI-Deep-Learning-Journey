## Introduction
# This notebook demonstrates Decision Tree Classification using Scikit-learn.

## What is a Decision Tree?
# A decision tree is a supervised learning algorithm used for classification and regression tasks.
# It splits data into branches based on feature conditions, forming a tree-like structure.

## How does it work in classification tasks?
# The tree starts with a root node and recursively splits data based on the most important features.
# It makes predictions by traversing from root to leaf nodes.

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading the Dataset (Iris dataset for simplicity)
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Data Preprocessing
# Check for missing values
df.isnull().sum()

# Splitting Data into Training and Testing Sets
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Decision Tree Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluating the Model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization of Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
