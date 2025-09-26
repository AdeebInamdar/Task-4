# 🌟 Task 4: Binary Classification with Logistic Regression
This repository contains the solution for Task 4 (Classification with Logistic Regression) as part of the Elevate Labs AI & ML Internship program. 

# 🎯 Objective
The primary goal was to implement a Binary Classifier using the Logistic Regression algorithm, focusing on data preprocessing, model fitting, and a thorough evaluation using classification metrics and threshold tuning.

# 🛠️ Project Details
Algorithm: Logistic Regression (for Binary Classification)
Dataset: data.csv (The Breast Cancer Wisconsin Diagnostic data)
Features (X): 30 descriptive features of cell nuclei.
Target (y): M (Malignant) and B (Benign), encoded as 1 and 0.
Key Libraries: pandas, numpy, sklearn, matplotlib.

# 💻 Implementation Steps
Data Loading: Loaded the data.csv file.
Preprocessing: Target variable (diagnosis) was mapped from categorical ('M', 'B') to numerical (1, 0).
Splitting: Data was split into 70% training and 30% testing sets using train_test_split with stratify=y.
Scaling: Features were scaled using StandardScaler to ensure optimal performance for Logistic Regression.
Model Training: A LogisticRegression model was fitted to the scaled training data.

#📊 Model Performance Summary
The model demonstrated high performance on the test set.
Metric	Score
Accuracy	0.9825
Precision	0.9817
Recall	0.9907
ROC-AUC	0.9983

# Confusion Matrix (Default Threshold=0.5)
[[ 61   2]  <- True Negatives, False Positives
 [  1 107]] <- False Negatives, True Positives
The high Recall (0.9907) is critical for this medical application, indicating the model successfully identified most malignant cases, minimizing dangerous False Negatives.







