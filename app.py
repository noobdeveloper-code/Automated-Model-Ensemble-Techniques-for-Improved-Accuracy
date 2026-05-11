import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import StackingClassifier, VotingClassifier

# Load the saved ensemble model (best model from your backend code)
with open('best_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Accuracy from backend code
accuracies = {
    'Logistic Regression': 0.934,
    'Random Forest': 0.951,
    'XGBoost': 0.940,
    'SVC': 0.919,
    'Voting Ensemble': 0.952,
    'Stacking Ensemble': 0.955
}

# Set Streamlit page title
st.title('Automated Model Ensemble Techniques for Improved Accuracy')

# Display dataset preview
st.subheader('Dataset Preview:')
st.write(X.head())

# Accuracy Plot
st.subheader("Model Accuracy Comparison")
def plot_accuracy_comparison(accuracies):
    plt.figure(figsize=(10, 6))
    plt.barh(list(accuracies.keys()), list(accuracies.values()), color='lightgreen')
    plt.xlabel("Accuracy")
    plt.title("Base vs Ensemble Model Accuracy")
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    return plt

accuracy_plot = plot_accuracy_comparison(accuracies)
st.pyplot(accuracy_plot)

# Show model type used
model_type = "Stacking" if isinstance(model, StackingClassifier) else "Voting"
st.info(f"Loaded Model Type: **{model_type} Ensemble**")

# Predict and display results
if st.button('Predict'):
    st.subheader("Predictions on Entire Dataset")
    predictions = model.predict(X)
    
    # Create a dataframe with predictions
    results_df = X.copy()
    results_df['Prediction'] = predictions
    results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})

    st.dataframe(results_df.head())

    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results as CSV",
        data=csv,
        file_name='breast_cancer_predictions.csv',
        mime='text/csv'
    )

    # Accuracy
    acc = accuracy_score(y, predictions)
    st.write(f"Accuracy: {acc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        return fig

    conf_matrix_fig = plot_confusion_matrix(y, predictions)
    st.pyplot(conf_matrix_fig)

    # ROC Curve
    st.subheader("ROC Curve")
    def plot_roc_curve(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC Curve)')
        ax.legend(loc="lower right")
        return fig

    roc_curve_fig = plot_roc_curve(y, predictions)
    st.pyplot(roc_curve_fig)



# -------------------- Footer --------------------
st.markdown("---")
st.markdown(" Developed for 8th Sem VTU Internship Project — Automated Ensemble Modelling App")

