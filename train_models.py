import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# 1. Create Dataset (15 features, 1000 instances to meet requirements)
X, y = make_classification(
    n_samples=1000, 
    n_features=15, 
    n_informative=12, 
    n_redundant=3, 
    n_classes=2, 
    random_state=42
)

# Feature names
feature_names = [f"feature_{i+1}" for i in range(15)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split into Train and Test (Test data will be saved as test_data.csv)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test data CSV
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully!")

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop(columns=['target']))
y_train = train_df['target'].values

X_test = scaler.transform(test_df.drop(columns=['target']))
y_test = test_df['target'].values

# Save scaler for Streamlit app usage
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Save model file
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Metrics calculation
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

results_df = pd.DataFrame(results)
print("\n--- MODEL EVALUATION RESULTS ---")
print(results_df.to_string(index=False))