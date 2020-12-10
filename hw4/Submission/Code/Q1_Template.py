import pandas as pd
import numpy as np
import os

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

HOUSING_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'housing_data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')
PREDS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Predictions')

def load_data(dataset):
    """
    Load a pair of data X,y 

    Params
    ------
    dataset:    train/valid/test

    Return
    ------
    X:          shape (N, 240)
    y:          shape (N, 1)
    """
    X = pd.read_csv(os.path.join(HOUSING_DATA_DIR, f'{dataset}_x.csv'), header=None).to_numpy()
    y = pd.read_csv(os.path.join(HOUSING_DATA_DIR, f'{dataset}_y.csv'), header=None).to_numpy()

    return X,y

def hyper_parameter_tuning(model_class, param_grid, train, valid):
    """
    Tune the hyper-parameter using training and validation data

    Params
    ------
    model_class:    the model class
    param_grid:     the hyper-parameter grid, dict
    train:          the training data (train_X, train_y)
    valid:          the validatation data (valid_X, valid_y)

    Return
    ------
    model:          model fit with best params
    best_param:     the best params
    """
    train_X, train_y = train
    valid_X, valid_y = valid

    # Set up the parameter grid
    param_grid = list(ParameterGrid(param_grid))

    # train the model with each parameter setting in the grid

    maes = []
    for params in param_grid:
        print(f"Fitting {model_class.__name__} with params {params}")
        model = model_class(**params)
        model.fit(train_X, train_y)
        pred_y = model.predict(valid_X)
        mae = mean_absolute_error(y_true=valid_y, y_pred=pred_y)
        print(f"Got validation MAE {mae}")
        maes.append(mae)
        print()

    # choose the model with lowest MAE on validation set
    # then fit the model with the training and validation set (refit)
    best_params = param_grid[np.argmin(maes)]
    print(f"Best params for {model_class.__name__} are {best_params} with validation MAE {np.min(maes)}")
    best_model = model_class(**best_params)
    full_X = np.concatenate([train_X, valid_X], axis=0)
    full_y = np.concatenate([train_y, valid_y], axis=0)
    model.fit(full_X, full_y)

    print("============================================")

    # return the fitted model and the best parameter setting
    return best_model, best_params

def plot_mae_alpha(model_class, params, train, valid, test, title="Model"):
    """
    Plot the model MAE vs Alpha (regularization constant)

    Params
    ------
    model_class:    The model class to fit and plot
    params:         The best params found 
    train:          The training dataset
    valid:          The validation dataset
    test:           The testing dataset
    title:          The plot title

    Return
    ------
    None
    """
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)
    test_X, test_y = test

    # set up the list of alphas to train on
    alphas = 10.**np.arange(-4, 4)

    # train the model with each alpha, log MAE
    maes = []
    for alpha in alphas:
        new_params = params.copy()
        new_params['alpha'] = alpha
        print(f"Fitting {model_class.__name__} with params {new_params}")
        model = model_class(**new_params)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        mae = mean_absolute_error(y_true=test_y, y_pred=pred_y)
        maes.append(mae)

    # plot the MAE - Alpha
    xs = alphas
    ys = maes
    plt.xlabel('alpha')
    plt.ylabel('MAE')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(xs, ys)

    plt.savefig(os.path.join(FIGURES_DIR, f'{model_class.__name__}_mae_vs_alpha.png'))
    plt.show()
    

def main():
    print('Question 1.4')
    print()

    """
    Load in data
    """
    train = load_data('train')
    valid = load_data('valid')
    test = load_data('test')

    ols_grid = {} # no hyperparameters to tune???
    lasso_grid = {
        'alpha': 10.**np.arange(-4, 4),
        'max_iter': [100000],
        'tol': [1e-2]
    }
    ridge_grid = {
        'alpha': 10.**np.arange(-4, 4),
        'max_iter': [100000],
        'tol': [1e-2]
    }

    ols_model, ols_params = hyper_parameter_tuning(LinearRegression, ols_grid, train, valid)
    lasso_model, lasso_params = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    ridge_model, ridge_params = hyper_parameter_tuning(Ridge, ridge_grid, train, valid)

    plot_mae_alpha(Lasso, lasso_params, train, valid, test, "Lasso")
    plot_mae_alpha(Ridge, ridge_params, train, valid, test, "Ridge")

    ols_preds = ols_model.predict(test[0])
    lasso_preds = lasso_model.predict(test[0])
    ridge_preds = ridge_model.predict(test[0])

    np.savetxt(os.path.join(PREDS_DIR, 'q1_ols_preds.csv'), ols_preds, delimiter=',')
    np.savetxt(os.path.join(PREDS_DIR, 'q1_lasso_preds.csv'), lasso_preds, delimiter=',')
    np.savetxt(os.path.join(PREDS_DIR, 'q1_ridge_preds.csv'), ridge_preds, delimiter=',')


if __name__ == '__main__':
    main()
