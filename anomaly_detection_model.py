import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def visualize_anomalies(data, predictions, threshold):
    # Create a scatterplot to visualize anomalies
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data['Time'], data['Amount'], c=predictions, cmap='coolwarm')
    ax.set_title("Anomaly Detection")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amount")
    fig.colorbar(scatter)
    st.pyplot(fig)

def run():
    st.subheader("Anomaly Detection Model")
    
    # Load the dataset (replace 'creditcard.csv' with the actual path to your dataset)
    df = pd.read_csv('creditcard.csv')
    
    transaction_amount = st.number_input("Transaction Amount")
    transaction_time = st.number_input("Transaction Time")

    # Create a slider to adjust the threshold
    threshold = st.slider("Threshold for Anomaly Detection", min_value=0.001, max_value=0.1, value=0.05, step=0.001)

    # Create a button to trigger anomaly detection
    if st.button("Detect Anomaly"):
        try:
            # Initialize the Isolation Forest model for anomaly detection
            model = IsolationForest(contamination=threshold)

            # Prepare the input data for the model
            user_input = df[['Amount', 'Time']]  # Replace with your actual features

            # Use the model to predict anomalies in the dataset
            predictions = model.fit_predict(user_input)

            # Visualize anomalies
            visualize_anomalies(user_input, predictions, threshold)
        except ValueError:
            st.error("Please enter valid transaction details")
