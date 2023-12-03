import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip the input values to avoid overflow in the exponential function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return [1 if i > 0.7 else 0 for i in predictions]


def run():
    st.subheader("Clustering Model")
    
    # Load the dataset (replace 'creditcard.csv' with the actual path to your dataset)
    data = pd.read_csv('creditcard.csv')
    
    # Add code for clustering model here
    st.write("Add your clustering model code here")

    # Subsample the data such that 90& of the data is fradulent and 10% is non-fraudulent
    fraudulent = data[data['Class'] == 1]
    non_fraudulent = data[data['Class'] == 0]

    # Randomly sample non-fraudulent transactions
    non_fraudulent_sample = non_fraudulent.sample(n=len(fraudulent)*9, random_state=42)

    # Combine the fraudulent and non-fraudulent samples
    subsample = pd.concat([fraudulent, non_fraudulent_sample])

    # Split the data into features and target
    X = subsample.drop(['Class','Time'], axis=1)
    y = subsample['Class'].values

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    
    