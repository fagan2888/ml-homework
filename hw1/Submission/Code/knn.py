from collections import Counter
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

class KNN(object):
    """
    The KNN classifier
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.tree_ = None

    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier

        Hint:   Build a tree to get neighbors 
                faster at test time
        """
        self.tree_ = BallTree(x_train)
        self.y_train_ = y_train.reset_index(drop=True)
        return self

    def predict(self, x_test):
        """
        Predicting the test data
        Hint:   Get the K-Neighbors, then generate
                predictions using the labels of the
                neighbors
        """
        n_test = x_test.shape[0]

        y_pred = []
        _, indices = self.tree_.query(x_test, k=self.n_neighbors)
        assert(indices.shape[1] == self.n_neighbors)

        for i in range(n_test):
            neighbor_classes = self.y_train_[indices[i]]
            cl = Counter(neighbor_classes).most_common(1)[0][0]
            y_pred.append(cl)
        return np.array(y_pred)
