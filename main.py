import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score

# --- 1. Load Dataset ---
# Using the recommended Breast Cancer Wisconsin Dataset
print("Loading the dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(f"Dataset shape: {X.shape}")

# --- 2. Train/Test Split and Standardize Features ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Standardize features (scaling)
# Logistic Regression benefits from feature scaling.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized.")

# --- 3. Fit a Logistic Regression Model ---
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Logistic Regression model fitted to the training data.")

# Predict probabilities and class labels
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of the positive class (1)
y_pred = model.predict(X_test_scaled) # Default prediction at threshold 0.5

# --- 4. Evaluate with confusion matrix, precision, recall, ROC-AUC ---
print("\n--- Model Evaluation (Default Threshold: 0.5) ---")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show() #
# --- 5. Tune Threshold and explain sigmoid function ---
print("\n--- Threshold Tuning ---")
# Example of finding a threshold that favors higher Recall (e.g., in medical diagnosis)
# We might accept a slightly lower precision to ensure we don't miss many positive cases (high recall).
target_recall = 0.98 # Example target

best_threshold = 0.5
max_recall = 0
for i in np.linspace(0, 1, 100):
    y_new_pred = (y_pred_proba >= i).astype(int)
    current_recall = recall_score(y_test, y_new_pred)
    if current_recall >= target_recall:
        best_threshold = i
        max_recall = current_recall
        # Break once target is achieved, or continue to find the one with best trade-off
        break

y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)
tuned_cm = confusion_matrix(y_test, y_pred_tuned)
tuned_precision = precision_score(y_test, y_pred_tuned)
tuned_recall = recall_score(y_test, y_pred_tuned)

print(f"Threshold tuned to achieve Recall of at least {target_recall}: {best_threshold:.4f}")
print(f"Tuned Precision: {tuned_precision:.4f}")
print(f"Tuned Recall: {tuned_recall:.4f}")
print("Tuned Confusion Matrix:")
print(tuned_cm)


print("\n--- Sigmoid Function Explanation ---")
# The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$
# where $z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$ is the linear combination of features.
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid, label='Sigmoid Function')
plt.title('The Sigmoid Function')
plt.xlabel('z (Linear Input)')
plt.ylabel('$\sigma(z)$ (Probability)')
plt.grid(True)
plt.axhline(0.5, color='red', linestyle='--', label='Classification Threshold (0.5)')
plt.legend()
plt.show() #
print("""
The **sigmoid function**, also known as the logistic function, is crucial in logistic regression.
It takes any real-valued number (the linear combination of features, $z$) and maps it to a probability value between **0 and 1**.
This transformation allows the linear regression output to be interpreted as a **probability**, making it suitable for classification.
If the output $\sigma(z)$ is above a certain **threshold** (typically 0.5), the prediction is one class (e.g., 1); otherwise, it's the other (e.g., 0).
""")