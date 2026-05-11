import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import StackingClassifier, VotingClassifier

# Load saved model
with open('best_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Sidebar for user input
st.sidebar.header("Input Patient Data")

def user_input_features():
    input_data = {}
    for feature in data.feature_names:
        input_data[feature] = st.sidebar.slider(
            feature,
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )
    return pd.DataFrame([input_data])

input_df = user_input_features()

st.title('Automated Model Ensemble Techniques for Improved Accuracy')
st.markdown("### Dataset Preview")
st.write(X.head())

# Class distribution chart
st.markdown("### Class Distribution")
class_counts = pd.Series(y).value_counts().rename(index={0: 'Malignant', 1: 'Benign'})
st.bar_chart(class_counts)

# Accuracy comparison chart
accuracies = {
    'Logistic Regression': 0.934,
    'Random Forest': 0.951,
    'XGBoost': 0.940,
    'SVC': 0.919,
    'Voting Ensemble': 0.952,
    'Stacking Ensemble': 0.955
}

st.markdown("### Model Accuracy Comparison")
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

# Model Metrics Table
def get_metrics(mdl, X_, y_):
    pred = mdl.predict(X_)
    return {
        "Accuracy": accuracy_score(y_, pred),
        "Precision": precision_score(y_, pred),
        "Recall": recall_score(y_, pred),
        "F1 Score": f1_score(y_, pred)
    }

metrics = {
    "Stacking Ensemble": get_metrics(model, X, y)
}

st.markdown("### Model Performance Metrics")
st.table(pd.DataFrame(metrics).T)

# Predict on entire dataset
if st.button('Predict on Entire Dataset'):
    st.subheader("Predictions on Entire Dataset")
    predictions = model.predict(X)
    results_df = X.copy()
    results_df['Prediction'] = predictions
    results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})

    st.dataframe(results_df.head())

    # Download prediction results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results as CSV",
        data=csv,
        file_name='breast_cancer_predictions.csv',
        mime='text/csv'
    )

    # Show accuracy
    acc = accuracy_score(y, predictions)
    st.write(f"Accuracy on dataset: {acc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, predictions)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Predict on user input
st.markdown("---")
st.subheader("Predict on Custom Input Data")
st.write(input_df)

if st.button("Predict for User Input"):
    user_pred = model.predict(input_df)[0]
    label = 'Benign' if user_pred == 1 else 'Malignant'
    st.success(f"Prediction: **{label}**")

# Footer
st.markdown("---")
st.markdown("📌 *Developed for 8th Sem VTU Internship Project — Automated Ensemble Learning App*")
