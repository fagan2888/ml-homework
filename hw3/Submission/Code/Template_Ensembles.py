# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 63
    # of testing samples: 20
    ------
    """
    train_X = np.genfromtxt("../../Data/gene_data/gene_train_x.csv", delimiter= ",")
    train_y = np.genfromtxt("../../Data/gene_data/gene_train_y.csv", delimiter= ",")
    test_X = np.genfromtxt("../../Data/gene_data/gene_test_x.csv", delimiter= ",")
    test_y = np.genfromtxt("../../Data/gene_data/gene_test_y.csv", delimiter= ",")

    return train_X, train_y, test_X, test_y

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')

def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    
    print('Q2.4')

    N = 150 # Each part will be tried with 1 to 150 estimators

    errs1 = []
    errs2 = []
    errs3 = []

    for n_estimators in range(1, N+1):
        # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)
        rfc1 = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt')
        rfc1.fit(train_X, train_y)
        test_preds_1 = rfc1.predict(test_X)
        errs1.append(1. - metrics.accuracy_score(test_y, test_preds_1))

        # Train RF with m = n_features recording the errors (errors will be of size 150)
        rfc2 = RandomForestClassifier(n_estimators=n_estimators, max_features=None)
        rfc2.fit(train_X, train_y)
        test_preds_2 = rfc2.predict(test_X)
        errs2.append(1. - metrics.accuracy_score(test_y, test_preds_2))

        # Train RF with m = n_features/3 recording the errors (errors will be of size 150)
        rfc3 = RandomForestClassifier(n_estimators=n_estimators, max_features=1/3)
        rfc3.fit(train_X, train_y)
        test_preds_3 = rfc3.predict(test_X)
        errs3.append(1. - metrics.accuracy_score(test_y, test_preds_3))
        
    #plot the Random Forest results
    fig, ax = plt.subplots()

    ax.plot(np.arange(1, N+1), errs1, c='red', label='sqrt(p)')
    ax.plot(np.arange(1, N+1), errs2, c='green', label='p')
    ax.plot(np.arange(1, N+1), errs3, c='blue', label='p/3')

    ax.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('training error')
    plt.title('RandomForestClassifier')
    plt.savefig(os.path.join(FIGURES_DIR, 'rfc_results.png'))
    
    print('Q2.6')

    errs1 = []
    errs2 = []
    errs3 = []

    for n_estimators in range(1, N+1):
        # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)
        ada1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, learning_rate=.1, random_state=42)
        ada1.fit(train_X, train_y)
        test_preds_1 = ada1.predict(test_X)
        errs1.append(1. - metrics.accuracy_score(test_y, test_preds_1))

        # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)
        ada2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators, learning_rate=.1, random_state=42)
        ada2.fit(train_X, train_y)
        test_preds_2 = ada2.predict(test_X)
        errs2.append(1. - metrics.accuracy_score(test_y, test_preds_2))

        # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)
        ada3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=n_estimators, learning_rate=.1, random_state=42)
        ada3.fit(train_X, train_y)
        test_preds_3 = ada3.predict(test_X)
        errs3.append(1. - metrics.accuracy_score(test_y, test_preds_3))

    # plot the adaboost results
    fig, ax = plt.subplots()

    ax.plot(np.arange(1, N+1), errs1, c='red', label='max_depth=1')
    ax.plot(np.arange(1, N+1), errs2, c='green', label='max_depth=3')
    ax.plot(np.arange(1, N+1), errs3, c='blue', label='max_depth=5')

    ax.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('training error')
    plt.title('AdaBoostClassifier')
    plt.savefig(os.path.join(FIGURES_DIR, 'ada_results.png'))
    plt.show()

if __name__ == '__main__':
    main()
