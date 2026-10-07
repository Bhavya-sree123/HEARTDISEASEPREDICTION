# ==========================================================
# Heart Disease Prediction - Model Training Script
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================================
# Check XGBoost Availability
# ==========================================================

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv("heart.csv")

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

# ==========================================================
# Display Dataset Information
# ==========================================================

print("\nDataset Information\n")
print(df.info())

print("\nDataset Statistics\n")
print(df.describe())

print("\nFirst Five Rows\n")
print(df.head())

# ==========================================================
# Check Missing Values
# ==========================================================

print("\nMissing Values\n")
print(df.isnull().sum())

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# Feature and Target Separation
# ==========================================================

X = df.drop("target", axis=1)
y = df["target"]

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nScaler Saved Successfully")

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Machine Learning Models
# ==========================================================

models = {

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=10
        ),

    "SVM":
        SVC(
            probability=True
        ),

    "Gradient Boosting":
        GradientBoostingClassifier()

}

if XGB_AVAILABLE:

    models["XGBoost"] = XGBClassifier(
        eval_metric="logloss"
    )

# ==========================================================
# Train Models
# ==========================================================

results = []

best_accuracy = 0

best_model = None

best_model_name = ""

print("\nTraining Models...\n")

for name, model in models.items():

    print("Training :", name)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1

    })

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================================
# Results Table
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n===============================")
print("MODEL COMPARISON")
print("===============================\n")

print(results_df)

print("\nBest Model :", best_model_name)

print("Accuracy :", round(best_accuracy * 100,2), "%")

# ==========================================================
# Save Best Model
# ==========================================================

pickle.dump(best_model, open("model.pkl", "wb"))

print("\nBest Model Saved Successfully")

# ==========================================================
# Confusion Matrix
# ==========================================================

for name, model in models.items():

    prediction = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        prediction
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(name)

    plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

plt.figure(figsize=(10,6))

for name, model in models.items():

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probability
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={roc_auc:.2f})"
        )

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()

# ==========================================================
# Training Completed
# ==========================================================

print("\n======================================")

print("Heart Disease Model Training Completed")

print("Best Model :", best_model_name)

print("Files Generated:")

print("1. model.pkl")

print("2. scaler.pkl")

print("======================================")