import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from joblib import Parallel, delayed
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Custom node class
class Node:
    """
    A node class for a binary tree. 
    Nodes are initialized with data, left and right pointers, and split feature and value.
    """
    # Constructor
    def __init__(self, data):
        self.data = data  
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None

# Custom isolation tree class
class iTree:
    """
    A class for an isolation tree. 
    Isolation trees are initialized with a height limit.
    The fit method generates a single isolation tree.
    The generate_tree method recursively generates a single isolation tree using random splitting of the data,
      until the height limit is reached or there is only one observation left.
    The path_length method returns the path length for a single observation.
    The path_length_helper method is a helper function for the path_length method that recursively calculates the path length for a single observation.
    """
    # Constructor
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    # Fits the model to the training data
    def fit(self, X):
        self.root = self.generate_tree(X, 0)

    # Generates a single isolation tree using recursive splitting of the data until the height limit is reached or there is only one observation left
    def generate_tree(self, X, height):
        if height >= self.height_limit or len(X) <= 1:
            return Node(X)
        else:
            # Ensure the chosen column has more than one unique value
            valid_columns = [col for col in X.columns if X[col].nunique() > 1]
            if not valid_columns:
                return Node(X)

            # Randomly select a feature and value to split on
            split_feature = rand.choice(valid_columns)
            split_value = rand.choice(X[split_feature].dropna().unique())
            root = Node(X)
            root.split_feature = split_feature
            root.split_value = split_value

            # Split the data
            left = X[X[split_feature] < split_value]
            right = X[X[split_feature] >= split_value]

            # Check if either left or right is empty
            if left.empty or right.empty:
                return Node(X)

            # Recursively generate the left and right subtrees
            root.left = self.generate_tree(left, height + 1)
            root.right = self.generate_tree(right, height + 1)
            return root
        
    # Returns the path length for a single observation
    def path_length(self, X):
        return self.path_length_helper(X, self.root, 0)
    
    # Helper function for path_length method that recursively calculates the path length for a single observation 
    def path_length_helper(self, X, root, path_length):
        if root.left is None and root.right is None:
            return path_length
        else:
            split_feature = root.split_feature
            split_value = root.split_value
            if X[split_feature] < split_value:
                return self.path_length_helper(X, root.left, path_length + 1)
            else:
                return self.path_length_helper(X, root.right, path_length + 1)
    

# Custom isolation forest class
class iForest:
    """
    A class for an isolation forest. 
    Isolation forests are initialized with the number of trees, height limit, and sample size.
    The fit method generates the specified number of isolation trees.
    The anomaly_score method returns the anomaly score for each observation.
    The predict method returns the predictions for each observation.
    The c method is a helper function for calculating c as defined in the original paper (https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).
    """
    # Constructor
    def __init__(self, n_trees=100, height_limit=None, sample_size=None):
        self.n_trees = n_trees
        self.height_limit = height_limit
        self.sample_size = sample_size
        self.trees = []

    # Fits the model to the training data
    def fit(self, X):
        if self.height_limit is None:
            self.height_limit = int(np.ceil(np.log2(self.sample_size)))
        if self.sample_size is None:
            self.sample_size = int(np.ceil(len(X) ** 0.5))

        # Parallelize the tree fitting process for increased speed 
        self.trees = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.fit_tree)(X) for _ in range(self.n_trees)
        )

    # Fits a single tree to a random sample of the data
    def fit_tree(self, X):
        sample_indices = np.random.choice(len(X), size=self.sample_size, replace=False)
        sample = X.iloc[sample_indices]
        tree = iTree(self.height_limit)
        tree.fit(sample)
        return tree

    # Returns the anomaly score for each observation in X
    def anomaly_score(self, X):
        path_lengths = np.zeros(len(X))
        for tree in self.trees:
            for i in range(len(X)):
                path_lengths[i] += tree.path_length(X.iloc[i])
        avg_path_length = path_lengths / self.n_trees
        return 2 ** (-avg_path_length / self.c(self.sample_size))
    
    # Returns the predictions for each observation in X based on the specified threshold 
    # The threshold is calculated as the specified percentile of the anomaly scores for the training data 
    def predict(self, scores, threshold):
        predictions = np.array([1 if score > threshold else 0 for score in scores])
        return predictions

    # Helper function for calculating c
    def c(self, size):
        if size > 2:
            return 2 * (np.log(size - 1) + np.euler_gamma) - 2 * (size - 1) / size
        elif size == 2:
            return 1
        else:
            return 0

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

# Custom function for running the anomaly detection model
def run():
    st.title("Anomaly Detection")

    # Load the data
    file_path = 'creditcard.csv' 
    df_scaled = pd.read_csv(file_path)

    # Scale the data
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)

    # Create a sidebar menu for model selection
    threshold_slider = st.slider("Select Anomaly Score Threshold Percentile", 0, 100, 90)
    number_of_trees = st.slider("Select Number of Trees", 0, 200, 50)
    contamination_rate = st.slider("Select Contamination Rate", 0.0, 0.5, 0.1)
    height_limit = st.slider("Select Height Limit Rate", 0, 50, 10)
    user_input = st.text_area("Enter transaction features separated by commas (V1, V2, ..., V28, Amount):")

    # Detect anomalies for the entered transaction when the button is clicked
    if st.button("Detect Anomaly"):
        # Calculate the execution time
        start_time = time.time()
        # Check if the user entered a transaction
        try:
            # Check if the user entered exactly 29 features
            input_values = [float(val) for val in user_input.split(',')]
            if len(input_values) != 29:
                st.error("Please enter exactly 29 features.")
                return

            # Check if the user entered a valid transaction
            input_df = pd.DataFrame([input_values], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
            # Scale the user input and store it in input_df_scaled
            new_scaler = MinMaxScaler()
            new_scaler.fit(df_scaled.drop(['Time', 'Class'], axis=1))    
            input_df_scaled = pd.DataFrame(new_scaler.transform(input_df), columns=input_df.columns)
            
            # Display the scaled user input
            st.write(input_df_scaled)

            # Subsample the dataset to balance the classes and split into training and test data
            df_majority = df[df['Class'] == 0]
            # Sample the minority class
            df_minority = df[df['Class'] == 1]
            # Undersample the majority class
            df_majority_undersampled = df_majority.sample(n=len(df_minority)*9, random_state=42)
            # Combine the majority class with the minority class
            df_balanced = pd.concat([df_majority_undersampled, df_minority])
            
            # Store the target variable and drop the time and class columns
            y_true_balanced = df_balanced['Class']
            df_balanced = df_balanced.drop(['Time', 'Class'], axis=1)

            # Split the data into training and test sets of size 90/10 where 90 is the training set and 10 is the test set
            X_train, X_test, y_train, y_test = train_test_split(df_balanced, y_true_balanced, test_size=0.1, random_state=42)

            # Fit the model to the training data using custom isolation forest
            model = iForest(n_trees=number_of_trees, sample_size=256, height_limit=height_limit)
            model.fit(X_train)

            # Calculate the threshold for the anomaly score based on the specified percentile
            threshold = np.percentile(model.anomaly_score(X_train), threshold_slider)

            # Anomaly detection for the entered transaction
            new_score = model.anomaly_score(input_df_scaled)
            is_anomaly = model.predict(new_score, threshold=threshold)
            
            # Calculate the execution time
            end_time = time.time()

            # Output the result of the anomaly detection
            if is_anomaly:
                st.error('This transaction is considered an anomaly.')
            else:
                st.success('This transaction is considered normal.')
            st.write(f"Execution Time: {end_time - start_time:.2f} seconds")

            # Calculate and display metrics for training data using custom isolation forest which was fit to the training data
            y_pred_balanced = model.predict(model.anomaly_score(X_train), threshold=threshold)
            acc, precision, recall, f1, auc = calculate_metrics(y_train, y_pred_balanced)
            st.write(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
            fig = plot_confusion_matrix(y_train, y_pred_balanced)
            st.pyplot(fig)

            # Calculate and display metrics for test data using custom isolation forest which was fit to the training data
            y_pred_unseen = model.predict(model.anomaly_score(X_test), threshold=threshold)
            acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred_unseen)
            st.write(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
            fig = plot_confusion_matrix(y_test, y_pred_unseen)
            st.pyplot(fig)

            # Use sklearn's IsolationForest with training data for comparison with custom isolation forest
            model_sklearn = IsolationForest(n_estimators=number_of_trees, contamination=contamination_rate)
            model_sklearn.fit(X_train)
            y_pred_sklearn = model_sklearn.predict(X_train)
            y_pred_sklearn[y_pred_sklearn == 1] = 0
            y_pred_sklearn[y_pred_sklearn == -1] = 1
            acc, precision, recall, f1, auc = calculate_metrics(y_train, y_pred_sklearn)
            st.write(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
            fig = plot_confusion_matrix(y_train, y_pred_sklearn)
            st.pyplot(fig)

            # Use sklearn's IsolationForest with test data for comparison with custom isolation forest
            y_pred_sklearn_unseen = model_sklearn.predict(X_test)
            y_pred_sklearn_unseen[y_pred_sklearn_unseen == 1] = 0
            y_pred_sklearn_unseen[y_pred_sklearn_unseen == -1] = 1
            acc, precision, recall, f1, auc = calculate_metrics(y_test, y_pred_sklearn_unseen)
            st.write(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
            fig = plot_confusion_matrix(y_test, y_pred_sklearn_unseen)
            st.pyplot(fig)
           
        # Display an error message if the user did not enter a transaction 
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the anomaly detection model
if __name__ == "__main__":
    run()

# Example transaction:
#-1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62