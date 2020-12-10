from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    classes = set(np.concatenate([y_true, y_pred]))
    result = pd.DataFrame(index=classes, columns=classes)
    for class_true in classes:
        for class_pred in classes:
            result.loc[class_true, class_pred] = ((y_true == class_true) & (y_pred == class_pred)).sum() / len(y_true)
    return result

class LogisticRegression:
    def __init__(self, input_size, alpha, std, learning_rate, epsilon, num_epochs):
        self.W = std * np.random.randn(input_size + 1)
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1) # bias term
        N, D = X.shape

        self.loss_hist_ = []
        for epoch in range(self.num_epochs):
            loss, loss_grads = self.loss(X, y)
            self.loss_hist_.append(loss)
            print('Epoch', epoch+1, 'loss', loss)
            if loss < self.epsilon:
                break
            self.W -= self.learning_rate * loss_grads # why isn't -= working???

        return self

    def predict(self, X):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1) # bias term
        probs = self.sigmoid_Wx(X)
        y_pred = (probs >= 0.5)
        return probs, y_pred

    def sigmoid_Wx(self, X):
        scores = np.matmul(X, self.W) # (N, D) x (D,) = (N,), one score per datapoint
        return 1 / (1 + np.exp(-scores))

    def loss(self, X, y):
        N, D = X.shape

        probs = self.sigmoid_Wx(X)
        data_loss = (-1 / N) * sum(y * np.log(probs) + (1 - y) * np.log(1. - probs))
        reg_loss = self.alpha * self.W.dot(self.W)
        loss = data_loss + reg_loss

        loss_grads = (1 / N) * np.matmul(X.T, probs - y) # (D, N) x (N,) = (D,)
        loss_grads += 2*self.alpha*self.W

        return loss, loss_grads

DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'Data'))
FIGURES_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Figures'))
PREDICTIONS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Predictions'))

def main():
    X_train, y_train, X_val, y_val, X_test = map(lambda fn: pd.read_csv(os.path.join(DATA_DIR, fn)), ['X_train.csv', 'Y_train.csv', 'X_val.csv', 'Y_val.csv', 'X_test.csv'])
    test_ids = X_test['ID']

    X_train = X_train['Review Text'] # ignore all feats besides review text
    X_val = X_val['Review Text']
    X_test = X_test['Review Text']
    y_train = (y_train['Sentiment'] == 'Positive').values.astype(int) # binarize labels
    y_val = (y_val['Sentiment'] == 'Positive').values.astype(int)

    print('Transforming into bow representation')
    vocab = sorted(CountVectorizer().fit(X_train).vocabulary_.keys())
    # todo: don't use dense arrays here
    X_train = CountVectorizer(vocabulary=vocab).fit_transform(X_train).toarray()
    X_val = CountVectorizer(vocabulary=vocab).fit_transform(X_val).toarray()
    X_test = CountVectorizer(vocabulary=vocab).fit_transform(X_test).toarray()

    param_grid = {
        'input_size': [len(vocab)],
        'std': [1e-4],
        'epsilon': [1e-3],

        'alpha': [0., 0.05, 0.1, 0.2],
        'num_epochs': [10, 50, 250, 1000],
        'learning_rate': [1e-3, 1e-2, 1e-1, 1.]
    }

    # Q3.1
    best_params, best_val_auc = None, -np.inf
    for params in ParameterGrid(param_grid):
        lr = LogisticRegression(**params)
        print('Fitting training set with params:', params)
        lr.fit(X_train, y_train)
        print('Making predictions for validation set')
        y_val_probs, y_val_pred = lr.predict(X_val)

        val_auc = roc_auc_score(y_val, y_val_probs)
        print('Val ROC/AUC:', val_auc)
        print('Val accuracy:', accuracy_score(y_val, y_val_pred))
        print()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params
    
    lr = LogisticRegression(**best_params)
    lr.fit(X_train, y_train)
    best_train_auc = roc_auc_score(y_train, lr.predict(X_train)[0])
    print('Finished grid search, best parameters are', best_params, 'with validation AUC', best_val_auc, 'and train AUC', best_train_auc)

    #best_params = {'alpha': 0.0, 'epsilon': 0.001, 'input_size': len(vocab), 'learning_rate': 1.0, 'num_epochs': 1000, 'std': 0.0001}

    # Q3.2
    for param_name, param_values in param_grid.items():
        if len(param_values) == 1:
            continue

        print(f'Graphing ROC/AUC score as {param_name} changes')
        train_aucs, val_aucs = [], []
        for param_value in param_values:
            new_params = best_params.copy()
            new_params[param_name] == param_value
            print('Using parameters:', new_params)

            lr = LogisticRegression(**new_params)
            lr.fit(X_train, y_train)
            train_aucs.append(roc_auc_score(y_train, lr.predict(X_train)[0]))
            val_aucs.append(roc_auc_score(y_val, lr.predict(X_val)[0]))
        
        xs, ys = param_values, train_aucs
        plt.scatter(xs, ys)
        plt.xlabel(param_name)
        plt.ylabel('Train ROC/AUC score')
        plt.ylim(bottom=0., top=1.)
        plt.savefig(os.path.join(FIGURES_DIR, f'q_3_2_train_auc_vs_{param_name}.png'))
        plt.clf()

        xs, ys = param_values, val_aucs
        plt.scatter(xs, ys)
        plt.xlabel(param_name)
        plt.ylabel('Validation ROC/AUC score')
        plt.ylim(bottom=0., top=1.)
        plt.savefig(os.path.join(FIGURES_DIR, f'q_3_2_val_auc_vs_{param_name}.png'))
        plt.clf()

    # Q3.3
    lr = LogisticRegression(**best_params)
    lr.fit(X_train, y_train)
    y_val_probs, y_val_pred = lr.predict(X_val)
    print('Confusion matrix for validation set:')
    print(confusion_matrix(y_val, y_val_pred))

    # Q4
    lr = LogisticRegression(**best_params)
    lr.fit(X_train, y_train)
    y_val_probs, y_val_pred = lr.predict(X_val)
    print('Precision:', precision_score(y_val, y_val_pred))
    print('Recall:', recall_score(y_val, y_val_pred))
    print('F1:', f1_score(y_val, y_val_pred))
    print('ROC/AUC:', roc_auc_score(y_val, y_val_probs))

    # Fit on the full training data
    lr.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_test_probs, y_test_pred = lr.predict(X_test)
    pd.DataFrame([test_ids, y_test_pred]).to_csv(os.path.join(PREDICTIONS_DIR, 'predictions.csv'), index=False)

if __name__ == '__main__':
    main()
