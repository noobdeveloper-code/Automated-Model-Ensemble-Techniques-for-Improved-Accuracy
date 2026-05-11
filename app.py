import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

with open('best_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Input Patient Data")

def user_input_features():

    input_data = {}

    for feature in data.feature_names:

        input_data[feature] = st.sidebar.slider(
            label=feature,
            min_value=float(X[feature].min()),
            max_value=float(X[feature].max()),
            value=float(X[feature].mean())
        )

    return pd.DataFrame([input_data])

input_df = user_input_features()

st.sidebar.markdown("---")

st.sidebar.info("""
Developed using:

- Streamlit
- Scikit-learn
- XGBoost
- Ensemble Learning
- Wisconsin Breast Cancer Dataset
""")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🩺 Breast Cancer Prediction Using Ensemble Learning")

st.markdown("""
This application predicts whether a breast tumor is
**Benign** or **Malignant**
using Ensemble Machine Learning techniques.

### Models Used
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Voting Ensemble
- Stacking Ensemble
""")

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.markdown("## Dataset Preview")

st.dataframe(X.head())

# ---------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------

st.markdown("## Class Distribution")

class_counts = pd.Series(y).value_counts()

class_counts.index = ['Malignant', 'Benign']

st.bar_chart(class_counts)

# ---------------------------------------------------
# MODEL ACCURACY COMPARISON
# ---------------------------------------------------

accuracies = {
    'Logistic Regression': 0.9737,
    'Random Forest': 0.9649,
    'XGBoost': 0.9561,
    'SVC': 0.9825,
    'Voting Ensemble': 0.9649,
    'Stacking Ensemble': 0.9649
}

# Sort accuracies
accuracies = dict(
    sorted(
        accuracies.items(),
        key=lambda x: x[1]
    )
)

st.markdown("## Model Accuracy Comparison")

def plot_accuracy_comparison(acc_dict):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        list(acc_dict.keys()),
        list(acc_dict.values()),
        color='lightgreen'
    )

    ax.set_xlabel("Accuracy")
    ax.set_title("Base Models vs Ensemble Models")
    ax.grid(axis='x', linestyle='--')

    return fig

accuracy_plot = plot_accuracy_comparison(accuracies)

st.pyplot(accuracy_plot)

# ---------------------------------------------------
# MODEL TYPE
# ---------------------------------------------------

model_type = (
    "Stacking Ensemble"
    if isinstance(model, StackingClassifier)
    else "Voting Ensemble"
)

st.info(f"Loaded Model Type: {model_type}")

# ---------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------

def get_metrics(mdl, X_, y_):

    pred = mdl.predict(X_)

    return {
        "Accuracy": accuracy_score(y_, pred),
        "Precision": precision_score(y_, pred),
        "Recall": recall_score(y_, pred),
        "F1 Score": f1_score(y_, pred)
    }

metrics = {
    model_type: get_metrics(model, X, y)
}

st.markdown("## Model Performance Metrics")

metrics_df = pd.DataFrame(metrics).T

st.table(metrics_df)

# ---------------------------------------------------
# PREDICT ENTIRE DATASET
# ---------------------------------------------------

if st.button('Predict on Entire Dataset'):

    st.subheader("Predictions on Entire Dataset")

    predictions = model.predict(X)

    results_df = X.copy()

    results_df['Prediction'] = predictions

    results_df['Prediction_Label'] = results_df['Prediction'].map({
        0: 'Malignant',
        1: 'Benign'
    })

    st.dataframe(results_df.head())

    # Download CSV
    csv = results_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Results as CSV",
        data=csv,
        file_name='breast_cancer_predictions.csv',
        mime='text/csv'
    )

    # Accuracy
    acc = accuracy_score(y, predictions)

    st.write(f"### Accuracy on Dataset: {acc:.4f}")

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax_cm
    )

    ax_cm.set_xlabel('Predicted')

    ax_cm.set_ylabel('Actual')

    ax_cm.set_title("Confusion Matrix")

    st.pyplot(fig_cm)

    # ---------------------------------------------------
    # ROC CURVE
    # ---------------------------------------------------

    st.subheader("ROC Curve")

    y_probs = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_probs)

    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

    ax_roc.plot(
        fpr,
        tpr,
        lw=2,
        label=f'AUC = {roc_auc:.2f}'
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    ax_roc.set_xlabel('False Positive Rate')

    ax_roc.set_ylabel('True Positive Rate')

    ax_roc.set_title('Receiver Operating Characteristic')

    ax_roc.legend(loc="lower right")

    st.pyplot(fig_roc)

# ---------------------------------------------------
# USER INPUT PREDICTION
# ---------------------------------------------------

st.markdown("---")

st.subheader("Predict on Custom Input Data")

st.write(input_df)

if st.button("Predict for User Input"):

    user_pred = model.predict(input_df)[0]

    user_prob = model.predict_proba(input_df)[0][1]

    label = 'Benign' if user_pred == 1 else 'Malignant'

    st.success(f"Prediction: {label}")

    st.write(f"Prediction Confidence: {user_prob:.2%}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
📌 Developed for VTU 8th Semester Internship Project  
### Automated Ensemble Learning for Breast Cancer Prediction
""")
