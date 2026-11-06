# Machine Learning Assignment 2 - M.Tech AIML

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy multiple machine learning classification models on a structured binary classification dataset, and build an interactive Streamlit web application to demonstrate real-world ML deployment workflows.

## b. Dataset Description
- **Dataset Source:** Synthetic/Public Benchmark Classification Dataset generated for robust testing.
- **Total Instances:** 1,000 rows (satisfying requirement of $\ge 500$).
- **Total Features:** 15 numerical features (satisfying requirement of $\ge 12$).
- **Target Variable:** Binary classification target (0 or 1).

## c. GitHub Repository Link
https://github.com/shyam216143/bits-ml-assignment-2

## d. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8250 | 0.9091 | 0.8131 | 0.8529 | 0.8325 | 0.6503 |
| **Decision Tree** | 0.7500 | 0.7495 | 0.7453 | 0.7745 | 0.7596 | 0.4998 |
| **KNN** | 0.9500 | 0.9783 | 0.9694 | 0.9314 | 0.9500 | 0.9008 |
| **Naive Bayes** | 0.7600 | 0.8417 | 0.7647 | 0.7647 | 0.7647 | 0.5198 |
| **Support Vector Machine** | 0.9650 | 0.9950 | 0.9703 | 0.9608 | 0.9655 | 0.9300 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs reliably as a linear baseline, achieving solid accuracy and an MCC score of ~0.70. |
| **Decision Tree** | Captures non-linear relationships well, yielding high accuracy but slightly prone to overfitting compared to ensemble models. |
| **KNN** | Performs strongly with high AUC score, sensitive to feature scaling which was correctly handled using StandardScaler. |
| **Naive Bayes** | Serves as a fast probabilistic baseline, though feature independence assumption slightly limits performance. |
| **Support Vector Machine** | Achieves the highest overall balance of accuracy, AUC, and F1 score due to optimal hyperplane separation. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the best overall performance with the highest AUC and MCC score on this dataset. |