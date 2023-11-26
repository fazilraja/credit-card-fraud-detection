import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def visualize_anomalies(data, predictions):
    # Visualize the anomalies in the context of normal transactions
    fig, ax = plt.subplots()
    # Normal transactions are plotted as green dots
    ax.scatter(data.loc[predictions == 1, 'Time'], data.loc[predictions == 1, 'Amount'], c='green', label='Normal')
    # Anomalies are plotted as red dots
    ax.scatter(data.loc[predictions == -1, 'Time'], data.loc[predictions == -1, 'Amount'], c='red', label='Anomaly')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    st.pyplot(fig)

def run():
    st.subheader("Anomaly Detection Model")

    # Load the dataset (for this example, we're assuming the dataset is already uploaded and available as 'creditcard.csv')
    df = pd.read_csv('creditcard.csv')

    # User input for all features as a comma-separated string
    user_input = st.text_area("Enter the transaction features separated by tab (V1 V2 ... V28 Amount):")

    # Process the input
    if st.button("Detect Anomaly"):
        try:
            # Split the user input and convert to float
            input_features = [float(x) for x in user_input.split('\t')]

            # Check if the number of features is correct
            if len(input_features) != 29:
                st.error("Incorrect number of features. Please enter 29 features separated by commas.")
                return

            # Create a dataframe from the user input
            input_df = pd.DataFrame([input_features], columns=['V{}'.format(i) for i in range(1, 29)] + ['Amount'])

            # Initialize the Isolation Forest model
            model = IsolationForest(n_estimators=100, contamination=0.1)

            # Fit the model on the dataset without the 'Time' and 'Class' columns
            model.fit(df.drop(['Time', 'Class'], axis=1))

            # Predict the anomaly for the input data
            prediction = model.predict(input_df)

            # Display results
            if prediction[0] == -1:
                st.error('This transaction is considered an anomaly.')
            else:
                st.success('This transaction is considered normal.')

            # Optionally visualize the result
            # Note that this visualization will use the entire dataset
            predictions = model.predict(df.drop(['Time', 'Class'], axis=1))
            visualize_anomalies(df.assign(predictions=predictions), predictions)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check the input format.")

if __name__ == "__main__":
    run()
