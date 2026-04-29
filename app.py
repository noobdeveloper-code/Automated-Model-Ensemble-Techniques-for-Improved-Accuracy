import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import StackingClassifier, VotingClassifier

# -------------------- Load Saved Model --------------------
with open('best_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------- Load Dataset --------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# -------------------- Load Accuracy JSON --------------------
with open('accuracies.json', 'r') as f:
    accuracies = json.load(f)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Ensemble Modeling App", layout="wide")
st.title("Automated Ensemble Modeling App")
st.caption("**Internship Project — Rooman Technologies (8th Sem, VTU)**")

# -------------------- Sidebar: Custom Prediction --------------------
st.sidebar.header(" Custom Prediction Input")
user_input = {}

for col in X.columns:
    user_input[col] = st.sidebar.number_input(
        label=col,
        min_value=float(np.min(X[col])),
        max_value=float(np.max(X[col])),
        value=float(np.mean(X[col])),
        step=0.1
    )

if st.sidebar.button("Predict from Custom Input"):
    input_df = pd.DataFrame([user_input])
    pred = model.predict(input_df)[0]
    label = "Benign" if pred == 1 else "Malignant"
    st.sidebar.success(f"🧬 Predicted Class: **{label}**")

# -------------------- Dataset Preview --------------------
st.subheader("Dataset Preview")
st.dataframe(X.head())

# -------------------- Accuracy Comparison --------------------
st.subheader("Model Accuracy Comparison")
def plot_accuracy_comparison(accuracies):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(accuracies.keys()), list(accuracies.values()), color='lightgreen')
    ax.set_xlabel("Accuracy")
    ax.set_title("Base vs Ensemble Model Accuracy")
    ax.grid(axis='x', linestyle='--')
    return fig

st.pyplot(plot_accuracy_comparison(accuracies))

# -------------------- Display Loaded Model Type --------------------
model_type = "Stacking" if isinstance(model, StackingClassifier) else "Voting"
st.info(f" Loaded Model Type: **{model_type} Ensemble**")

# -------------------- Predict Button --------------------
if st.button(" Predict on Full Dataset"):
    st.subheader(" Predictions on Full Dataset")

    predictions = model.predict(X)
    results_df = X.copy()
    results_df['Prediction'] = predictions
    results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})
    st.dataframe(results_df.head())

    # Accuracy
    acc = accuracy_score(y, predictions)
    st.success(f" Overall Accuracy: **{acc:.4f}**")

    # Download predictions
    csv = results_df.to_csv(index=False)
    st.download_button(
        label=" Download Predictions as CSV",
        data=csv,
        file_name='breast_cancer_predictions.csv',
        mime='text/csv'
    )

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        return fig

    st.pyplot(plot_confusion_matrix(y, predictions))

    # ROC Curve
    st.subheader(" ROC Curve")
    def plot_roc_curve(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        return fig

    st.pyplot(plot_roc_curve(y, predictions))

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("2025 • Developed for **Rooman Technologies Internship Project** — VTU 8th Semester")
