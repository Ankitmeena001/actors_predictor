# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("actors_data_modified.csv")

# Train model
@st.cache_resource
def train_model(df):
    X_train = df.iloc[:, 1:]
    y_train = df['name']
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X_train, y_train)
    return clf

# App UI
def main():
    st.title("🎭 Indian Actor Predictor")
    st.write("Answer the questions below to predict the Indian actor you're thinking of.")

    df = load_data()
    clf = train_model(df)

    feature_columns = df.columns[1:]

    user_input = {}
    for col in feature_columns:
        # Handle binary columns (0/1) with Yes/No questions
        if df[col].dropna().isin([0, 1]).all():
            user_input[col] = st.selectbox(f"{col.replace('_', ' ')}?", ["Yes", "No"]) == "Yes"
        else:
            # For non-binary columns (rare in your dataset), use text input or dropdown
            user_input[col] = st.text_input(f"Enter value for {col}")

    input_df = pd.DataFrame([{
        key: int(val) if isinstance(val, bool) else val
        for key, val in user_input.items()
    }])

    if st.button("Predict"):
        prediction = clf.predict(input_df)[0]
        st.success(f"🎬 You are thinking of: **{prediction}**")

if __name__ == "__main__":
    main()
