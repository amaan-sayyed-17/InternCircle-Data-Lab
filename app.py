"""Simple interactive Iris classifier demo. Run with: streamlit run app.py"""
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Iris Classifier Demo", page_icon="🌼")
st.title("Iris Flower Classifier")
st.write("A small educational demo using a Decision Tree trained on the Iris dataset.")

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Flower measurements (cm)")
values = []
for feature in X.columns:
    values.append(st.sidebar.slider(
        feature.replace(" (cm)", "").replace("_", " ").title(),
        float(X[feature].min()), float(X[feature].max()),
        float(X[feature].median())
    ))

if st.button("Predict species"):
    sample = pd.DataFrame([values], columns=X.columns)
    prediction = int(model.predict(sample)[0])
    st.success(f"Predicted species: {iris.target_names[prediction]}")
    st.caption("Educational demo only; prediction quality depends on the training data.")
