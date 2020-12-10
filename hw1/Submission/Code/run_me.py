#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold

from decision_tree import DecisionTree
from knn import KNN
from metrics import get_metrics

CODE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..', 'Data'))
PREDICTIONS_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', 'Predictions'))

def main():
    parser = ArgumentParser()
    parser.add_argument('--decision_tree', action='store_true')
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--knn_cv', action='store_true')
    parser.add_argument('--decision_tree_cv', action='store_true')
    parser.add_argument('--make_predictions', action='store_true')
    parser.add_argument('--add_index', action='store_true')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        args.make_predictions = True

    x_train = pd.read_csv(os.path.join(DATA_DIR, 'x_train.csv'), header=None)
    x_test = pd.read_csv(os.path.join(DATA_DIR, 'x_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'), header=None)
    y_train = y_train.T.squeeze() # make it a Series

    # decision tree
    if args.decision_tree:
        print('running decision tree, max_depth=5 min_size=5')
        tree = DecisionTree(max_depth=5, min_size=5)
        tree.fit(x_train, y_train)
        y_pred = tree.predict(x_test)
        pd.Series(y_pred).to_csv(os.path.join(PREDICTIONS_DIR, 'decision_tree_predictions.csv'), index=False, header=False)

    # knn
    if args.knn:
        print('running knn, n_neighbors=5')
        knn = KNN(n_neighbors=5)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        pd.Series(y_pred).to_csv(os.path.join(PREDICTIONS_DIR, 'knn_predictions.csv'), index=False, header=False)

    if args.knn_cv:
        for n_neighbors in [3, 5, 10, 20, 25]:
            knn = KNN(n_neighbors=n_neighbors)

            kfold = KFold(n_splits=5)
            metrics = []
            for fit_index, val_index in kfold.split(x_train, y_train):
                x_fit, y_fit = x_train.iloc[fit_index], y_train[fit_index]
                x_val, y_val = x_train.iloc[val_index], y_train[val_index]

                print(f'running knn cv, n_neighbors={n_neighbors}')
                knn.fit(x_fit, y_fit)
                y_pred_val = knn.predict(x_val)
                metrics.append(get_metrics(y_val, y_pred_val))
            
            a, p, r, f = zip(*metrics)
            mean_val_accuracy = np.mean(a)
            mean_val_precision = np.mean(p)
            mean_val_recall = np.mean(r)
            mean_val_f1 = np.mean(f)
            print(f'knn cv metrics for n_neighbors={n_neighbors}:')
            print('avg accuracy:', mean_val_accuracy)
            print('avg precision:', mean_val_precision)
            print('avg recall:', mean_val_recall)
            print('avg f1 score:', mean_val_f1)
    
    if args.decision_tree_cv:
        for max_depth in [3, 6, 9, 12, 15]:
            tree = DecisionTree(max_depth=max_depth, min_size=5)

            kfold = KFold(n_splits=5)
            metrics = []
            for fit_index, val_index in kfold.split(x_train, y_train):
                x_fit, y_fit = x_train.iloc[fit_index], y_train[fit_index]
                x_val, y_val = x_train.iloc[val_index], y_train[val_index]

                print(f'running decision tree cv, max_depth={n_neighbors} min_size=5')
                tree.fit(x_fit, y_fit)
                y_pred_val = tree.predict(x_val)
                metrics.append(get_metrics(y_val, y_pred_val))
            
            a, p, r, f = zip(*metrics)
            mean_val_accuracy = np.mean(a)
            mean_val_precision = np.mean(p)
            mean_val_recall = np.mean(r)
            mean_val_f1 = np.mean(f)
            print(f'decision tree cv metrics for max_depth={max_depth} min_size=5:')
            print('avg accuracy:', mean_val_accuracy)
            print('avg precision:', mean_val_precision)
            print('avg recall:', mean_val_recall)
            print('avg f1 score:', mean_val_f1)

    if args.make_predictions:
        '''
        # using k-fold cross validation to select the best hyperparameters for the decision tree model
        param_grid = {
            'max_depth': np.arange(3, 11),
            'min_size': [2, 3, 5, 8, 10]
        }
        final_model = GridSearchCV(tree, param_grid, cv=5, scoring='f1', refit=True, verbose=10)
        final_model.fit(x_train, y_train)
        print(final_model.best_params_)
        '''
        # {'max_depth': 10, 'min_size': 3}

        final_model = DecisionTree(max_depth=10, min_size=3)

        kfold = KFold(n_splits=5)
        metrics = []
        aucs = []
        for fit_index, val_index in kfold.split(x_train, y_train):
            x_fit, y_fit = x_train.iloc[fit_index], y_train[fit_index]
            x_val, y_val = x_train.iloc[val_index], y_train[val_index]

            print('running best model cv, decision tree with max_depth=10 min_size=3')
            final_model.fit(x_fit, y_fit)
            y_pred_val = final_model.predict(x_val)
            metrics.append(get_metrics(y_val, y_pred_val))
            aucs.append(roc_auc_score(y_val, y_pred_val))
        
        a, p, r, f = zip(*metrics)
        mean_val_accuracy = np.mean(a)
        mean_val_precision = np.mean(p)
        mean_val_recall = np.mean(r)
        mean_val_f1 = np.mean(f)
        mean_val_auc = np.mean(aucs)
        print('best model cv metrics:')
        print('avg accuracy:', mean_val_accuracy)
        print('avg precision:', mean_val_precision)
        print('avg recall:', mean_val_recall)
        print('avg f1 score:', mean_val_f1)
        print('avg auc:', mean_val_auc)

        final_model.fit(x_train, y_train)
        y_pred_train = final_model.predict(x_train)
        accuracy, precision, recall, f1 = get_metrics(y_train, y_pred_train)
        auc = roc_auc_score(y_train, y_pred_train)
        print('best model training set metrics:')
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)
        print('f1 score:', f1)
        print('auc:', auc)
        
        y_pred = final_model.predict(x_test)
        pd.Series(y_pred).to_csv(os.path.join(PREDICTIONS_DIR, 'best.csv'), index=False, header=False)
    
    if args.add_index:
        y_pred = pd.read_csv(os.path.join(PREDICTIONS_DIR, 'best.csv'), header=None)
        y_pred.to_csv(os.path.join(PREDICTIONS_DIR, 'best.csv'), header=None) # this adds the index by default
        print('finished adding index')

if __name__ == '__main__':
    main()
