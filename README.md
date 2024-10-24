<h3>Input:</h3>

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr')
    else:
        auc = 'N/A'  # AUC doesn't apply to non-probabilistic models
        
    return accuracy, precision, recall, f1, auc

results = []

for model_name, model in models.items():
    accuracy, precision, recall, f1, auc = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
print(results_df)

<h3>Output :</h3>

 Model  Accuracy  Precision  Recall  F1 Score  AUC
0  Logistic Regression       1.0        1.0     1.0       1.0  1.0
1                  SVM       1.0        1.0     1.0       1.0  1.0
2        Random Forest       1.0        1.0     1.0       1.0  1.0
3                  KNN       1.0        1.0     1.0       1.0  1.0
4        Decision Tree       1.0        1.0     1.0       1.0  1.0

