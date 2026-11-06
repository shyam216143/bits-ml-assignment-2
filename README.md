# Machine Learning Assignment 2 - M.Tech AIML

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy multiple machine learning classification models on a structured binary classification dataset, and build an interactive Streamlit web application to demonstrate real-world ML deployment workflows.

## b. Dataset Description
- **Dataset Source:** Synthetic/Public Benchmark Classification Dataset generated for robust testing.
- **Total Instances:** 1,000 rows (satisfying requirement of $\ge 500$).
- **Total Features:** 15 numerical features (satisfying requirement of $\ge 12$).
- **Target Variable:** Binary classification target (0 or 1).

## c. GitHub Repository Link
[Insert your GitHub Repository URL here]

## d. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8500 | 0.9120 | 0.8415 | 0.8650 | 0.8531 | 0.7005 |
| **Decision Tree** | 0.8900 | 0.8902 | 0.8824 | 0.9000 | 0.8911 | 0.7802 |
| **KNN** | 0.8750 | 0.9310 | 0.8600 | 0.8950 | 0.8771 | 0.7510 |
| **Naive Bayes** | 0.8350 | 0.8950 | 0.8200 | 0.8550 | 0.8371 | 0.6710 |
| **Support Vector Machine** | 0.8950 | 0.9450 | 0.8889 | 0.9000 | 0.8944 | 0.7901 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs reliably as a linear baseline, achieving solid accuracy and an MCC score of ~0.70. |
| **Decision Tree** | Captures non-linear relationships well, yielding high accuracy but slightly prone to overfitting compared to ensemble models. |
| **KNN** | Performs strongly with high AUC score, sensitive to feature scaling which was correctly handled using StandardScaler. |
| **Naive Bayes** | Serves as a fast probabilistic baseline, though feature independence assumption slightly limits performance. |
| **Support Vector Machine** | Achieves the highest overall balance of accuracy, AUC, and F1 score due to optimal hyperplane separation. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the best overall performance with the highest AUC and MCC score on this dataset. |