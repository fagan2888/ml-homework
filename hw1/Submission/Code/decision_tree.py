from collections import Counter, namedtuple
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

def get_split(xs, ys):
    """
    Function to generate the split which results in
        the best gini_index

    Params
    ------
    xs : the input data (n_samples, n_features)
    ys : the input label (n_samples,)

    Return
    ------
    {split_index, split_value}
    """
    # we'll assume that there are only 2 classes, 0 and 1.
    # we want to find the split that minimizes the gini cost (proportional to the expected change in gini impurity after making the split):
    # gini cost = nl0*nl1/nl + nr0*nr1/nr
    # steps: for each feature, sort the values for that feature. sort y correspondingly. calculate nl1. calculate the other vecs from that.

    xs.reset_index(drop=True, inplace=True)
    ys.reset_index(drop=True, inplace=True)

    best_cost = np.inf
    best_split = None

    n_samples, n_features = xs.shape

    for split_index in range(n_features):
        split_values = xs.iloc[:, split_index]
        sort_indices = np.argsort(split_values)
        sorted_split_values = split_values[sort_indices]
        sorted_ys = ys[sort_indices]

        # nl1 tells us, given a split value, how many instances with <= that split value were classified as 1s.
        # we find the index of the value within sorted_split_values (eg. using binary search), then look at the corresponding entry in nl1.
        nl1 = np.cumsum(sorted_ys)
        nl = np.arange(1, n_samples+1)
        nl0 = nl - nl1

        nr = np.arange(n_samples-1, -1, step=-1)
        nr1 = sum(ys) - nl1
        nr0 = nr - nr1

        gini_costs = (nl0*nl1)/nl + (nr0*nr1)/nr
        best_index = np.argmin(gini_costs)
        best_cost_for_this_feature = gini_costs[best_index]
        if best_cost_for_this_feature < best_cost:
            best_cost = best_cost_for_this_feature
            best_split = split_index, sorted_split_values[best_index]
    
    assert(best_split is not None)
    return best_split

class DecisionNode:
    def __init__(self, left_node, right_node, split_index, split_value):
        self.left_node = left_node
        self.right_node = right_node
        self.split_index = split_index
        self.split_value = split_value
    
    def predict(self, x):
        succ = self.left_node if x[self.split_index] < self.split_value else self.right_node
        return succ.predict(x)

class LeafNode:
    def __init__(self, ys):
        self.prediction = Counter(ys).most_common(1)[0][0]

    def predict(self, x):
        return self.prediction

##########################################################
# Alternatively combine gini_index into get_split and 
# find the split point using an array instead of a for
# loop, would speed up run time 
##########################################################

class DecisionTree(BaseEstimator):
    """
    The Decision Tree classifier
    """
    def __init__(self, max_depth, min_size):
        """
        Params
        ------
        max_depth   : the maximum depth of the decision tree
        min_size    : the minimum observation points on a 
                        leaf/terminal node
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.root_ = None

    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier
        
        Hint: Build the decision tree using 
                splits recursively until a leaf
                node is reached

        """
        self.root_ = self._fit(x_train, y_train, depth=0)
        return self

    def _fit(self, xs, ys, depth):
        assert(xs.shape[0] >= self.min_size)

        # Cut off the tree at max_depth
        if depth >= self.max_depth:
            return LeafNode(ys)

        # Make the split based on Gini impurity
        split_index, split_value = get_split(xs, ys)

        splitter = xs.iloc[:, split_index] <= split_value
        xs_left, ys_left = xs[splitter], ys[splitter]
        xs_right, ys_right = xs[~splitter], ys[~splitter]

        # Don't make it if either child node falls below the min size
        n_left, n_right = xs_left.shape[0], xs_right.shape[0]
        if n_left < self.min_size or n_right < self.min_size:
            return LeafNode(ys)

        return DecisionNode(left_node=self._fit(xs_left, ys_left, depth+1),
                            right_node=self._fit(xs_right, ys_right, depth+1),
                            split_index=split_index,
                            split_value=split_value)

    def predict(self, x_test):
        """
        Predicting the test data

        Hint: Run the test data through the decision tree built
                during training (self.tree)
        """
        return np.array([self.root_.predict(x) for _, x in x_test.iterrows()])
