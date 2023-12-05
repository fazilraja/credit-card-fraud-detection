import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt


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

# Custom function for calculating metrics for the data
def calculate_metrics(y_true, y_pred):
    """
    Calculates accuracy, precision, recall, f1, and auc for the data using sklearn's metrics functions.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    return acc, precision, recall, f1, auc

# Custom function for plotting the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """
    Creates a confusion matrix plot using seaborn's heatmap function.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig

def scale_input(input_values, data):
    """
    Scales the input data using the same scaler used for the training data.
    """
    input_df = pd.DataFrame([input_values], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
    #scaler = MinMaxScaler()
    #scaler.fit(data.drop(['Time', 'Class'], axis=1))
    #input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    return input_df

def subsample_data(data):
    scaler = StandardScaler()
    scaler.fit(data.drop(['Time', 'Class'], axis=1))    
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

    return X_train, X_test, y_train, y_test



def run():
    st.subheader("Regression Model")
    
    # Load the dataset (replace 'creditcard.csv' with the actual path to your dataset)
    data = pd.read_csv('creditcard.csv')
    
    # Create sliders for num_iterations and learning_rate
    st.sidebar.subheader("Model Hyperparameters")
    num_iterations = st.sidebar.slider("Number of iterations", 100, 2000, 1000)
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.01)

    user_input = st.text_area("Enter transaction features separated by commas (V1, V2, ..., V28, Amount):")

    if st.button("Run Model"):

        # Check if the user entered a transaction
        try:
            # Check if the user entered exactly 29 features
            input_values = [float(val) for val in user_input.split(',')]
            if len(input_values) != 29:
                st.error("Please enter exactly 29 features.")
                return
            
            # Scale the input data
            input_df_scaled = scale_input(input_values, data)

            # subsample the data
            X_train, X_test, y_train, y_test = subsample_data(data)

            # create and fit the model
            model = LogisticRegression(n_iterations=num_iterations, learning_rate=learning_rate)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(input_df_scaled)

            if predictions[0] == 1:
                st.error("This transaction is fraudulent.")
            else:
                st.success("This transaction is not fraudulent.")

            # Calculate metrics and plot confusion matrix
            y_pred = model.predict(X_test)
            acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred)
            st.write("Metrics for the model:")
            st.write(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
            fig = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Please enter a valid input. Error: {e}")
    
if __name__ == "__main__":
    run()