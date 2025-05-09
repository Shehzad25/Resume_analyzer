import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

# Load data (or allow file upload)
st.title("Resume Analyzer")
uploaded_file = st.file_uploader("Upload a Resume CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Preprocess text
    data['cleaned_resume'] = data['Resume']  # Replace with actual cleaning if needed

    # Vectorization and model training (use saved model in production)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['cleaned_resume'])
    y = data['Category']

    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X, y)

    st.success("Model trained successfully!")

    # In real usage, allow single resume input for prediction
    new_resume = st.text_area("Paste a new resume below for prediction:")

    if new_resume:
        X_new = vectorizer.transform([new_resume])
        prediction = clf.predict(X_new)
        st.write("Predicted Category:",prediction[0])