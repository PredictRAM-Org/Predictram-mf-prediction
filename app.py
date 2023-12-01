import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your data
# Assuming df is your DataFrame containing the mutual fund data
# ...

# Drop columns not needed for modeling
df = df.drop(['Scheme Name', 'Benchmark', 'Riskometer Scheme', 'Riskometer Benchmark', 'NAV Date'], axis=1)

# Drop rows with missing values
df = df.dropna()

# Define features (X) and target variable (y)
X = df.drop(['Return 1 Year (%) Regular', 'Return 1 Year (%) Direct'], axis=1)
y = df['Return 1 Year (%) Regular']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate the linear regression model
lr_mse = mean_squared_error(y_test, lr_predictions)

# Streamlit App
st.title("Mutual Fund Return Prediction")

# Input features
st.sidebar.header("Input Features")
selected_features = st.sidebar.multiselect("Select Features", X.columns)

# Display selected features
st.write("Selected Features:", selected_features)

# Get user input for selected features
user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.number_input(feature, value=X[feature].mean())

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Make prediction using the linear regression model
prediction = lr_model.predict(user_df)

# Display prediction
st.write(f"Predicted 1-Year Return: {prediction[0]:.2f}")

# Display model evaluation metric
st.write(f"Linear Regression Mean Squared Error: {lr_mse:.2f}")
