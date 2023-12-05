import streamlit as st
import Group4_classification_model
import Group4_regression_model
import Group4_anomaly_detection_model

def main():
    # Initialize the Streamlit app
    st.title("Credit Card Fraud Detection")

    # Create a sidebar menu for model selection
    selected_model = st.sidebar.radio("Select a Model", ["Anomaly Detection", "Classification", "Regression"])

    if selected_model == "Anomaly Detection":
        Group4_anomaly_detection_model.run()

    elif selected_model == "Classification":
        Group4_classification_model.run()

    elif selected_model == "Regression":
        Group4_regression_model.run()


if __name__ == "__main__":
    main()
