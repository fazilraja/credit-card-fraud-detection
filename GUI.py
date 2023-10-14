import streamlit as st
import classification_model
import clustering_model
import anomaly_detection_model

def main():
    # Initialize the Streamlit app
    st.title("Credit Card Fraud Detection")

    # Create a sidebar menu for model selection
    selected_model = st.sidebar.radio("Select a Model", ["Classification", "Clustering", "Anomaly Detection"])

    if selected_model == "Classification":
        classification_model.run()

    elif selected_model == "Clustering":
        clustering_model.run()

    elif selected_model == "Anomaly Detection":
        anomaly_detection_model.run()

if __name__ == "__main__":
    main()
