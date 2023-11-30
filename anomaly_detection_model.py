import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from joblib import Parallel, delayed
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns


# Custom node class
class Node:
    # Constructor
    def __init__(self, data):
        self.data = data  
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None

# Custom isolation tree class
class iTree:
    # Constructor
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    # Fits the model to the training data
    def fit(self, X):
        self.root = self.generate_tree(X, 0)

    # Generates a single isolation tree
    def generate_tree(self, X, height):
        if height >= self.height_limit or len(X) <= 1:
            return Node(X)
        else:
            # Ensure the chosen column has more than one unique value
            valid_columns = [col for col in X.columns if X[col].nunique() > 1]
            if not valid_columns:
                return Node(X)

            split_feature = rand.choice(valid_columns)
            split_value = rand.choice(X[split_feature].dropna().unique())
            root = Node(X)
            root.split_feature = split_feature
            root.split_value = split_value

            left = X[X[split_feature] < split_value]
            right = X[X[split_feature] >= split_value]

            # Check if either left or right is empty
            if left.empty or right.empty:
                return Node(X)

            root.left = self.generate_tree(left, height + 1)
            root.right = self.generate_tree(right, height + 1)
            return root
        
    # Returns the path length for a single observation
    def path_length(self, X):
        return self.path_length_helper(X, self.root, 0)
    
    # Helper function for path_length
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
    # Constructor
    def __init__(self, n_trees=100, height_limit=None, sample_size=None, contamination=0.1):
        self.n_trees = n_trees
        self.height_limit = height_limit
        self.sample_size = sample_size
        self.trees = []
        self.contamination = contamination

    # Fits the model to the training data
    def fit(self, X):
        if self.height_limit is None:
            self.height_limit = int(np.ceil(np.log2(self.sample_size)))
        if self.sample_size is None:
            self.sample_size = int(np.ceil(len(X) ** 0.5))

        # Parallelize the tree fitting process
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
        return 2 ** (-avg_path_length / c(self.sample_size))
    
    # Returns the predictions for each observation in X
    def predict(self, X, scores):
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        predictions = np.array([1 if score > threshold else 0 for score in scores])
        return predictions

def c(size):
    if size > 2:
        return 2 * (np.log(size - 1) + np.euler_gamma) - 2 * (size - 1) / size
    elif size == 2:
        return 1
    else:
        return 0

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    return acc, precision, recall, f1, auc

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig


def run():
    st.title("Anomaly Detection")

    # Load and display dataset
    file_path = 'creditcard.csv'  # Update the file path if needed
    df = pd.read_csv(file_path)
    if st.checkbox("Show raw data"):
        st.write(df)

    threshold_slider = st.slider("Select Anomaly Score Threshold Percentile", 0, 100, 95)
    number_of_trees = st.slider("Select Number of Trees", 0, 1000, 50)
    contamination_rate = st.slider("Select Contamination Rate", 0.0, 0.5, 0.1)
    user_input = st.text_area("Enter transaction features separated by commas (V1, V2, ..., V28, Amount):")

    if st.button("Detect Anomaly"):
        start_time = time.time()

        try:
            input_values = [float(val) for val in user_input.split(',')]
            if len(input_values) != 29:
                st.error("Please enter exactly 29 features.")
                return

            input_df = pd.DataFrame([input_values], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])

            # Subsample the dataset
            df_majority = df[df['Class'] == 0]
            df_minority = df[df['Class'] == 1]
            df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)
            df_balanced = pd.concat([df_majority_undersampled, df_minority])

            y_true_balanced = df_balanced['Class']
            df_balanced = df_balanced.drop(['Time', 'Class'], axis=1)

            # Initialize and fit the model
            model = iForest(n_trees=number_of_trees, sample_size=256, contamination=contamination_rate)
            model.fit(df_balanced)

            # Prediction
            scores = model.anomaly_score(df_balanced)
            threshold = np.percentile(scores, threshold_slider)

            new_score = model.anomaly_score(input_df)
            is_anomaly = new_score > threshold

            end_time = time.time()

            # Output the result
            if is_anomaly:
                st.error('This transaction is considered an anomaly.')
            else:
                st.success('This transaction is considered normal.')
            st.write(f"Anomaly Score: {new_score}")
            st.write(f"Execution Time: {end_time - start_time} seconds")

            y_pred_balanced = model.predict(df_balanced, scores)
            # Calculate metrics
            acc, precision, recall, f1, auc = calculate_metrics(y_true_balanced, y_pred_balanced)
            st.write(f"Accuracy: {acc}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")
            st.write(f"AUC-ROC: {auc}")

            # Plot and display confusion matrix
            st.pyplot(plot_confusion_matrix(y_true_balanced, y_pred_balanced))

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run()