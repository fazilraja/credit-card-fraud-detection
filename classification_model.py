import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

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

    def __init__(self, min_samples_split=2, max_depth=100):
        """
        Initialization of the Decision Tree.

        Parameters:
        min_samples_split (int): Minimum number of samples required to split a node.
        max_depth (int): Maximum depth of the tree.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Fit the decision tree model.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Target values.
        """
        self.root = self._build_tree(X, y)
        self._prune(self.root, X, y)  # Pruning after building the tree

    def _prune(self, node, X, y):
            """
            Apply post-pruning to the decision tree.

            Parameters:
            node (Node): Current node in the decision tree.
            X (numpy.ndarray): Training features.
            y (numpy.ndarray): Target values.
            """
            if node.is_leaf_node():
                return

            # Prune left and right children first
            if node.left is not None:
                self._prune(node.left, X, y)
            if node.right is not None:
                self._prune(node.right, X, y)

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
        # Making predictions with and without the split
        error_without_split = np.sum(y != self._traverse_tree(X, node))
        error_with_split = np.sum(y != self._traverse_tree(X, node.left)) + np.sum(y != self._traverse_tree(X, node.right))

        # Prune (replace with a leaf node) if error without split is less or equal
        if error_without_split <= error_with_split:
            node.left = None
            node.right = None
            node.feature_index = None
            node.threshold = None
            node.value = Counter(y).most_common(1)[0][0]

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Target values.
        depth (int): Current depth of the tree.
        """
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None

        # Check if the node is a leaf node
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            best_gini = float("inf")
            for feature_index in range(num_features):
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    gini = self._gini_index(X, y, feature_index, threshold)
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature_index
                        best_threshold = threshold

        # Create a leaf node if no split is found
        if best_feature is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)

        # Split the data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

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
        return self._traverse_tree(x, node.right)

    def _gini_index(self, X, y, feature_index, threshold):
        """Calculate the Gini index for a split.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Target values.
        feature_index (int): Feature index.
        threshold (float): Threshold value.
        """
        left_idxs, right_idxs = self._split(X[:, feature_index], threshold)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0

        gini_left = 1.0 - sum([(np.sum(y[left_idxs] == c) / n_left) ** 2 for c in np.unique(y[left_idxs])])
        gini_right = 1.0 - sum([(np.sum(y[right_idxs] == c) / n_right) ** 2 for c in np.unique(y[right_idxs])])

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

# Example usage
# Note: This is just a demonstration. Actual usage would require a dataset.
# X, y are assumed to be numpy arrays with the appropriate dimensions.

# decision_tree = DecisionTree()
# decision_tree.fit(X, y)
# predictions = decision_tree.predict(X_test)





# def run():
#     st.subheader("Classification Model")
    
#     # Load the dataset (replace 'creditcard.csv' with the actual path to your dataset)
#     df = pd.read_csv('creditcard.csv')
    
#     # Add code for classification model here
#     st.write("Add your classification model code here")
