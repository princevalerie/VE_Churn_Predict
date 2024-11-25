import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to generate AI insights
def generate_ai_insights(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return "Unable to generate insights at the moment."

# Function to categorize tenure
def categorize_tenure(tenure):
    if tenure < 12:
        return "<1 year"
    elif tenure < 24:
        return "<2 years"
    elif tenure < 36:
        return "<3 years"
    elif tenure < 48:
        return "<4 years"
    elif tenure < 60:
        return "<5 years"
    else:
        return ">=5 years"

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('02 Churn-Dataset.xlsx')
    except FileNotFoundError:
        st.error("Excel file not found. Please ensure '02 Churn-Dataset.xlsx' is in the same directory.")
        return None

    # Convert 'tenure' to numeric and categorize
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['tenure'] = df['tenure'].apply(categorize_tenure)

    # Drop 'customerID' column if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Encode churn as binary if it exists
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df

# Function to map object columns
def map_object_columns(df):
    # Define mapping for categorical variables
    mapping_dict = {
        'gender': {'Male': 0, 'Female': 1},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': -1},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': -1},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'PaymentMethod': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        },
        'tenure': {
            "<1 year": 0,
            "<2 years": 1,
            "<3 years": 2,
            "<4 years": 3,
            "<5 years": 4,
            ">=5 years": 5
        }
    }

    for col in df.select_dtypes(include=['object']).columns:
        if col in mapping_dict:
            df[col] = df[col].map(mapping_dict[col])

    return df

# Function to preprocess data
def preprocess_data(df):
    # Map categorical columns
    df = map_object_columns(df)

    # Convert 'TotalCharges' to numeric, filling NaN values with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Fill NaN values for numerical columns with the mean of that column
    df.fillna(df.mean(), inplace=True)

    return df

def main():
    # Configure Gemini API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY is missing in environment variables.")
        return

    genai.configure(api_key=gemini_api_key)

    # Application Title
    st.title("Customer Churn Prediction with AI")

    # Load and process data
    df = load_data()
    if df is None:
        return

    df = preprocess_data(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Model Preparation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    # Model Accuracy
    accuracy = logreg.score(X_test_scaled, y_test)
    st.write(f"Model Accuracy: {accuracy * 100:.1f}%")

    # Input Form
    st.subheader("Enter Customer Details")
    input_data = {}

    # Dynamic input form based on columns
    for col in X.columns:
        if col == 'tenure':
            input_data[col] = st.selectbox("Select Tenure", 
                options=["<1 year", "<2 years", "<3 years", "<4 years", "<5 years", ">=5 years"])
        elif col == 'gender':
            input_data[col] = st.selectbox("Select Gender", options=['Male', 'Female'])
        elif col == 'SeniorCitizen':
            input_data[col] = st.radio("Senior Citizen Status", options=['0', '1'])
        elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            input_data[col] = st.selectbox(f"Select {col}", options=['No', 'Yes'])
        elif col == 'MultipleLines':
            input_data[col] = st.selectbox("Select MultipleLines", 
                options=['No', 'Yes', 'No phone service'])
        elif col == 'InternetService':
            input_data[col] = st.selectbox("Select Internet Service", 
                options=['DSL', 'Fiber optic', 'No'])
        elif col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies']:
            input_data[col] = st.selectbox(f"Select {col}", 
                options=['No', 'Yes', 'No internet service'])
        elif col == 'Contract':
            input_data[col] = st.selectbox("Select Contract", 
                options=['Month-to-month', 'One year', 'Two year'])
        elif col == 'PaymentMethod':
            input_data[col] = st.selectbox("Select Payment Method", 
                options=['Electronic check', 'Mailed check', 
                         'Bank transfer (automatic)', 
                         'Credit card (automatic)'])
        elif col in ['MonthlyCharges', 'TotalCharges']:
            input_data[col] = st.number_input(f"Enter {col}", 
                value=0.0, step=0.01)
        elif col in ['numAdminTickets', 'numTechTickets']:
            input_data[col] = st.slider(f"Enter {col}", 
                min_value=0, max_value=10, value=2, step=1)

    # Prediction
    if st.button("Get Prediction"):
        # Convert input
        input_df = pd.DataFrame([input_data])
        input_df = map_object_columns(input_df)
        input_df = input_df.fillna(0)
        input_df_scaled = scaler.transform(input_df)

        # Predict
        prediction = logreg.predict(input_df_scaled)

        if prediction[0] == 1:
            st.error("Prediction: This customer is likely to churn!")
            
            # AI Insight for churn
            churn_prompt = f"""
            Provide an in-depth analysis of why the customer with the following profile 
            is likely to churn: {input_data}. 
            Suggest specific and actionable retention strategies.
            """
            
            with st.spinner('Generating AI insights...'):
                churn_insights = generate_ai_insights(churn_prompt)
                st.subheader("Gemini AI Insights on Churn")
                st.write(churn_insights)
        else:
            st.success("Prediction: This customer is unlikely to churn.")

# Run the Application
if __name__ == "__main__":
    main()