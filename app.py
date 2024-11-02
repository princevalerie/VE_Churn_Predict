import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Function to categorize tenure
def categorize_tenure(tenure):
    if tenure < 12:
        return "<1 tahun"
    elif tenure < 24:
        return "<2 tahun"
    elif tenure < 36:
        return "<3 tahun"
    elif tenure < 48:
        return "<4 tahun"
    elif tenure < 60:
        return "<5 tahun"
    else:
        return ">=5 tahun"

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('02 Churn-Dataset.xlsx')
    
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
            "<1 tahun": 0,
            "<2 tahun": 1,
            "<3 tahun": 2,
            "<4 tahun": 3,
            "<5 tahun": 4,
            ">=5 tahun": 5
        }
    }

    for col in df.select_dtypes(include=['object']).columns:
        if col in mapping_dict:  # Only map if the column exists in the mapping_dict
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

# Main code
df = load_data()
df = preprocess_data(df)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale numerical features and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Calculate accuracy
accuracy = logreg.score(X_test_scaled, y_test)

# Streamlit app layout
st.title("Churn Prediction with Logistic Regression")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.1f}%")

# Input form for user data
st.subheader("Enter customer details to predict churn")

# Initialize an empty dictionary to hold user inputs
input_data = {}

# Create input fields for each feature
for col in X.columns:
    if col == 'tenure':
        input_data[col] = st.selectbox("Select Tenure", options=["<1 tahun", "<2 tahun", "<3 tahun", "<4 tahun", "<5 tahun", ">=5 tahun"])
    elif col == 'gender':
        input_data[col] = st.selectbox("Select Gender", options=['Male', 'Female'])
    elif col == 'SeniorCitizen':
        input_data[col] = st.radio("Select Senior Citizen", options=['0', '1'])
    elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        input_data[col] = st.selectbox(f"Select {col}", options=['No', 'Yes'])
    elif col == 'MultipleLines':
        input_data[col] = st.selectbox(f"Select {col}", options=['No', 'Yes', 'No phone service'])
    elif col == 'InternetService':
        input_data[col] = st.selectbox(f"Select {col}", options=['DSL', 'Fiber optic', 'No'])
    elif col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        input_data[col] = st.selectbox(f"Select {col}", options=['No', 'Yes', 'No internet service'])
    elif col == 'Contract':
        input_data[col] = st.selectbox(f"Select {col}", options=['Month-to-month', 'One year', 'Two year'])
    elif col == 'PaymentMethod':
        input_data[col] = st.selectbox(f"Select {col}", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    elif col in ['MonthlyCharges', 'TotalCharges']:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0, step=0.01)
    elif col == 'numAdminTickets':
        input_data[col] = st.slider(f"Enter {col}", min_value=0, max_value=10, value=0, step=1)
    elif col == 'numTechTickets':
        input_data[col] = st.slider(f"Enter {col}", min_value=0, max_value=10, value=0, step=1)

# Convert input_data to a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Map input data using the same mapping as before
input_df = map_object_columns(input_df)

# Fill any remaining NaN values with 0 (if any mapping missed values)
input_df = input_df.fillna(0)

# Scale user input
input_df_scaled = scaler.transform(input_df)

# Predict button
if st.button("Get Prediction"):
    # Make the prediction
    prediction = logreg.predict(input_df_scaled)
    
    if prediction[0] == 1:
        st.write("Prediction: This customer is likely to churn.")
    else:
        st.write("Prediction: This customer is unlikely to churn.")
