import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class Node:
    """A Node in the Decision Tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initialization of the decision tree node.

        Parameters:
        feature_index (int): Index of the feature used for splitting.
        threshold (float): Threshold value for splitting.
        left (Node): Left child node.
        right (Node): Right child node.
        value (int): Value if the node is a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf node."""
        return self.value is not None

class DecisionTree:
    """The CART Decision Tree."""

    def __init__(self, min_samples_split=2, max_depth=100, class_weight=None):
        """
        Initialization of the Decision Tree.

        Parameters:
        min_samples_split (int): Minimum number of samples required to split a node.
        max_depth (int): Maximum depth of the tree.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.class_weight = class_weight or {0: 1, 1: 1}  # Default weights

    def fit(self, X, y):
        """Fit the decision tree model.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Target values.
        """
        default_class_label = Counter(y).most_common(1)[0][0]
        self.root = self._build_tree(X, y, 0,default_class_label)
        self._prune(self.root, X, y)  # Pruning after building the tree

    def _prune(self, node, X, y):
            """
            Apply post-pruning to the decision tree.

            Parameters:
            node (Node): Current node in the decision tree.
            X (numpy.ndarray): Training features.
            y (numpy.ndarray): Target values.
            """
            # Base case: if the node is None or a leaf, return
            if node is None or node.is_leaf_node():
                return

            # Recursively prune left and right children
            if node.left is not None:
                self._prune(node.left, X, y)
            if node.right is not None:
                self._prune(node.right, X, y)

            # Ensure both children are not None before accessing their properties
            if node.left is not None and node.right is not None:
                # If both children are leaf nodes, check for pruning possibility
                if node.left.is_leaf_node() and node.right.is_leaf_node():
                    self._cost_complexity_prune(node, X, y)

    def _cost_complexity_prune(self, node, X, y):
        """
        Apply cost complexity pruning on a specific node.
        Parameters:
        node (Node): Node to evaluate for pruning.
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Target values.
        """
        # Predictions without the split
        predictions_without_split = [self._traverse_tree(x, node) for x in X]
        error_without_split = np.sum(y != predictions_without_split)

        # Predictions with the split
        predictions_left = [self._traverse_tree(x, node.left) for x in X]
        predictions_right = [self._traverse_tree(x, node.right) for x in X]
        error_with_split = np.sum(y != predictions_left) + np.sum(y != predictions_right)

        # Prune if error without split is less or equal
        if error_without_split <= error_with_split:
            node.left = None
            node.right = None
            node.feature_index = None
            node.threshold = None
            node.value = Counter(y).most_common(1)[0][0]

    def _build_tree(self, X, y, depth=0,default_class_label=None):
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None

        # If y is empty, return a leaf node with a default value or the parent class
        if len(y) == 0:
            return Node(value=default_class_label) 

        # Check for stopping criteria
        if num_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Find the best split
        best_gini = float("inf")
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._gini_index(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        # Check if a split is found
        if best_feature is not None:
            left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
            X_left, y_left = X[left_idxs], y[left_idxs]
            X_right, y_right = X[right_idxs], y[right_idxs]

            print(f"Depth: {depth}, Splitting: {len(X_left)} left, {len(X_right)} right")  # Debugging line

            # Recursively build the left and right subtrees
            left_subtree = self._build_tree(X_left, y_left, depth + 1)
            right_subtree = self._build_tree(X_right, y_right, depth + 1)

            return Node(best_feature, best_threshold, left_subtree, right_subtree)

        # If no split is found, return a leaf node
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)


    def predict(self, X):
        """Make predictions using the decision tree.

        Parameters:
        X (numpy.ndarray): Input features.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the tree for making a prediction.

        Parameters:
        x (numpy.ndarray): Single input feature.
        node (Node): Current node of the tree.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _gini_index(self, X, y, feature_index, threshold):
        left_idxs, right_idxs = self._split(X[:, feature_index], threshold)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0

        weights = self.class_weight
        gini_left = 1.0 - sum([(np.sum(y[left_idxs] == c) / n_left) ** 2 * weights.get(c, 1) for c in np.unique(y[left_idxs])])
        gini_right = 1.0 - sum([(np.sum(y[right_idxs] == c) / n_right) ** 2 * weights.get(c, 1) for c in np.unique(y[right_idxs])])

        # Calculate the weighted Gini index
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return weighted_gini
    
    def _split(self, feature_values, threshold):
        """Split the dataset on a feature and threshold.

        Parameters:
        feature_values (numpy.ndarray): Values of the feature to split on.
        threshold (float): Threshold value for splitting.
        """
        left_idxs = np.argwhere(feature_values <= threshold).flatten()
        right_idxs = np.argwhere(feature_values > threshold).flatten()
        return left_idxs, right_idxs


# Load the data
#data = pd.read_csv('credit-card-fraud-detection\creditcard.csv')
data = pd.read_csv('credit-card-fraud-detection\subsample.csv')
X = data.drop('Class', axis=1)
y = data['Class'].values

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X to NumPy arrays
X_train = X_train.values
X_test = X_test.values

# Calculate class weights
non_fraudulent_weight = 1 / (2 * 0.1)
fraudulent_weight = 1 / (2 * 0.9)

# Initialize the Decision Tree model
tree = DecisionTree(min_samples_split=2, max_depth=5, class_weight={0: non_fraudulent_weight, 1: fraudulent_weight})  # Adjust parameters as needed

# Train the model
tree.fit(X_train, y_train)

# Predictions
predictions = tree.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# def run():
#     st.subheader("Classification Model")
    
#     # Load the dataset (replace 'creditcard.csv' with the actual path to your dataset)
#     df = pd.read_csv('creditcard.csv')
    
#     # Add code for classification model here
#     st.write("Add your classification model code here")
