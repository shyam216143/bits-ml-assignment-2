import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - Breast Cancer Classifier", layout="wide")

# Header & Project Details
st.title("🚀 M.Tech AIML: Machine Learning Assignment 2")
st.markdown("### Breast Cancer Prediction & Classification Dashboard")
st.markdown("Comparative Evaluation of 5 Machine Learning Models on Diagnostic Features")

@st.cache_resource
def load_artifacts():
    scaler_path = "model/scaler.pkl"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    model_files = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Support Vector Machine": "model/support_vector_machine.pkl"
    }
    
    loaded_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    return scaler, loaded_models

scaler, models = load_artifacts()

# Sidebar controls
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if models:
    selected_model_name = st.sidebar.selectbox("Select Classification Model", list(models.keys()))
else:
    st.error("Models not found! Please run train_models.py first.")
    st.stop()

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully!")
else:
    default_path = "test_data.csv"
    if os.path.exists(default_path):
        test_df = pd.read_csv(default_path)
        st.info("Using default `test_data.csv` from repository.")
    else:
        st.warning("Please upload a test CSV file.")
        st.stop()

# Option to view complete dataset
st.subheader("📊 Test Dataset Preview")
show_full_dataset = st.checkbox("Show complete dataset (all rows)")

if show_full_dataset:
    st.dataframe(test_df)
    st.info(f"Displaying all {len(test_df)} rows.")
else:
    st.dataframe(test_df.head())
    st.caption("Showing first 5 rows. Check the box above to view the complete dataset.")

if 'target' not in test_df.columns:
    st.error("Uploaded CSV must contain the target column!")
    st.stop()

X_test_raw = test_df.drop(columns=['target'])
y_test = test_df['target'].values

if scaler:
    X_test = scaler.transform(X_test_raw)
else:
    X_test = X_test_raw.values

# Model Inference
model = models[selected_model_name]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

# Calculate Metrics
acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = 0.0
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

st.markdown(f"---")
st.subheader(f"📈 Evaluation Metrics for: **{selected_model_name}**")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔍 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)

with col_b:
    st.subheader("📑 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)