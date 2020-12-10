# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
import pandas as pd
# feel free to import any sklearn model here
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'breast_cancer_data', 'data.csv')

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    df = pd.read_csv(DATA_PATH)

    cols = df.columns
    X = df[cols[2:-1]].to_numpy()
    y = df[cols[1]].to_numpy()
    y = (y=='M').astype(np.int) * 2 - 1

    train_X = X[:-150]
    train_y = y[:-150]

    test_X = X[-150:]
    test_y = y[-150:]

    return train_X, train_y, test_X, test_y

def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()

    '''
    # find the best params for rfc
    param_grid = {
        'n_estimators': np.arange(10, 21),
        'max_features': ['sqrt']
    }

    g = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1').fit(train_X, train_y)
    print('best random forest params:', g.best_params_)
    rfc = g.best_estimator_
    '''
    rfc = RandomForestClassifier(**{'max_features': 'sqrt', 'n_estimators': 13})

    '''
    # find the best params for ada
    param_grid = {
        'base_estimator': [DecisionTreeClassifier(max_depth=1)],
        'n_estimators': np.arange(30, 41)
    }

    g = GridSearchCV(AdaBoostClassifier(), param_grid, scoring='f1').fit(train_X, train_y)
    print('best adaboost params:', g.best_params_)
    ada = g.best_estimator_
    '''
    ada = AdaBoostClassifier(**{'base_estimator': DecisionTreeClassifier(max_depth=1), 'n_estimators': 32})

    '''
    # find the best params for lr
    param_grid = {
        'C': 10. ** np.arange(-4, 5),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

    g = GridSearchCV(LogisticRegression(), param_grid, scoring='f1').fit(train_X, train_y)
    print('best lr params:', g.best_params_)
    lr = g.best_estimator_
    '''
    lr = LogisticRegression(**{'C': 10.0, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'})

    '''
    # find the best params for svc
    param_grid = {
        'C': 10. ** np.arange(-4, 5),
        'kernel': ['rbf']
    }
    
    g = GridSearchCV(SVC(), param_grid, scoring='f1').fit(train_X, train_y)
    print('best svc params:', g.best_params_)
    svc = g.best_estimator_
    '''
    svc = SVC(**{'C': 1000.0, 'kernel': 'rbf'})

    print('Q3.1')

    stac = StackingClassifier(estimators=[
        ('random forest', rfc),
        ('adaboost', ada),
        ('logistic regression', lr),
        ('svc', svc)
    ], final_estimator=LogisticRegression())
    kf = KFold(n_splits=10)
    f1s = []
    for learn_ix, val_ix in kf.split(train_X, train_y):
        learn_X, learn_y, val_X, val_y = train_X[learn_ix, :], train_y[learn_ix], train_X[val_ix, :], train_y[val_ix]
        stac.fit(learn_X, learn_y)
        val_preds = stac.predict(val_X)
        f1s.append(metrics.f1_score(val_y, val_preds))
    print('average validation f1 score:', np.mean(f1s))

    '''
    stac.fit(train_X, train_y)
    test_preds = stac.predict(test_X)
    print('stacked f1 score on test set:', metrics.f1_score(test_y, test_preds))
    '''

if __name__ == '__main__':
    main()
