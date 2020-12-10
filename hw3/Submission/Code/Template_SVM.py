import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn import metrics
from sklearn import svm

class SVM(object):
    """
    SVM Support Vector Machine
    with sub-gradient descent training.
    """
    def __init__(self, C):
        """
        Params
        ------
        n_features  : number of features (D)
        C           : regularization parameter (default: 1)
        """
        self.C = C

    def fit(self, X, y, lr, iterations):
        """
        Fit the model using the training data.

        Params
        ------
        X           :   (ndarray, shape = (n_samples, n_features)):
                        Training input matrix where each row is a feature vector.
        y           :   (ndarray, shape = (n_samples,)):
                        Training target. Each entry is either -1 or 1.
        lr          :   learning rate
        iterations  :   number of iterations
        """
        n_samples, n_features = X.shape
        self.objective_hist_ = []

        # Initialize the parameters wb (is this right?)
        wb = np.random.rand(n_features + 1)

        # initialize any container needed for save results during training
        best_obj = np.inf
        best_wb = None

        for i in range(iterations):
            # calculate learning rate with iteration number i
            lr_t = lr / np.sqrt(i + 1)

            # calculate the subgradients
            subgrad = self.subgradient(wb, X, y)

            # update the parameter wb with the gradients
            wb -= lr_t * subgrad

            # calculate the new objective function value
            obj = self.objective(wb, X, y)

            # record objective function values (for debugging purpose) to see
            # if the model is converging
            self.objective_hist_.append(obj)

            # compare the current objective function value with the saved best value
            # update the best value if the current one is better
            if obj < best_obj:
                best_obj = obj
                best_wb = wb

            # Logging
            if (i+1)%1000 == 0:
                print(f"Training step {i+1:6d}: LearningRate[{lr_t:.7f}], Objective[{obj:.7f}]")

        # Save the best parameter found during training
        self.w_, self.b_ = self.unpack_wb(best_wb, n_features)

        return self

    @staticmethod
    def unpack_wb(wb, n_features):
        """
        Unpack wb into w and b
        """
        w = wb[:n_features]
        b = wb[-1]

        return (w,b)
    
    @staticmethod
    def pack_wb(w, b):
        return np.append(w, b)

    def g(self, X, wb):
        """
        Helper function for g(x) = WX+b
        """
        n_samples, n_features = X.shape

        w,b = self.unpack_wb(wb, n_features)
        gx = np.dot(w, X.T) + b

        return gx

    def hinge_loss(self, X, y, wb):
        """
        Hinge loss for max(0, 1 - y(Wx+b))

        Params
        ------

        Return
        ------
        hinge_loss
        hinge_loss_mask
        """
        hinge = 1 - y*(self.g(X, wb))
        hinge_mask = (hinge > 0).astype(np.int)
        hinge = hinge * hinge_mask

        return hinge, hinge_mask


    def objective(self, wb, X, y):
        """
        Compute the objective function for the SVM.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        obj (float): value of the objective function evaluated on X and y.
        """
        n_samples, n_features = X.shape
        
        hinge, _ = self.hinge_loss(X, y, wb)
        obj = self.C * sum(hinge) + wb.dot(wb)
        return obj

    def subgradient(self, wb, X, y):
        """
        Compute the subgradient of the objective function.

        Params
        ------
        X   :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
        y   :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        Return
        ------
        subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model 
        """
        n_samples, n_features = X.shape
        w, b = self.unpack_wb(wb, n_features)

        # Retrieve the hinge mask
        hinge, hinge_mask = self.hinge_loss(X, y, wb)

        ## Cast hinge_mask on y to make y to be 0 where hinge loss is 0
        cast_y = - hinge_mask * y

        # Cast the X with an addtional feature with 1s for b gradients: -y
        cast_X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

        # Calculate the gradients for w and b in hinge loss term
        grad = self.C * np.dot(cast_y, cast_X)

        # Calculate the gradients for regularization term
        grad_add = np.append(2*w,0)
        
        # Add the two terms together
        subgrad = grad+grad_add

        return subgrad

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Params
        ------
        X   :    (ndarray, shape = (n_samples, n_features)): test data

        Return
        ------
        y   :   (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        # retrieve the parameters wb
        wb = self.pack_wb(self.w_, self.b_)

        # calculate the predictions
        y = np.sign(self.g(X, wb))

        # return the predictions
        return y

    def get_params(self):
        """
        Get the model parameters.

        Params
        ------
        None

        Return
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        return (self.w_, self.b_)

    def set_params(self, w, b):
        """
        Set the model parameters.

        Params
        ------
        w       (ndarray, shape = (n_features,)):
                coefficient of the linear model.
        b       (float): bias term.
        """
        self.w_ = w
        self.b_ = b


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'breast_cancer_data', 'data.csv')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')

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

def plot_decision_boundary(clf, X, y, title='SVM'):
    """
    Helper function for plotting the decision boundary

    Params
    ------
    clf     :   The trained SVM classifier
    X       :   (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
    y       :   (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
    title   :   The title of the plot

    Return
    ------
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    
    meshed_data = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(meshed_data)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, linewidth=1, edgecolor='black')
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.xlim(xx.min(), xx.max())
    plt.title(title)

    plt.savefig(os.path.join(FIGURES_DIR, title + '_decision_boundary.png'))
    plt.show()

def main():
    # Set the seed for numpy random number generator
    # so we'll have consistent results at each run
    np.random.seed(0)

    # Load in the training data and testing data
    train_X, train_y, test_X, test_y = load_data()

    # For using the first two dimensions of the data
    #train_X = train_X[:,:2]
    #test_X = test_X[:,:2]

    print('Q1.3')

    clf = SVM(C=1.)
    # increasing the number of iterations is... giving us a worse f1 score???
    objs = clf.fit(train_X, train_y, lr=2e-3, iterations=10000)

    train_preds  = clf.predict(train_X)
    test_preds  = clf.predict(test_X)

    print_metrics(train_y, train_preds, 'train')
    print_metrics(test_y, test_preds, 'test')
    #plot_decision_boundary(clf, train_X, train_y)

    print('Q1.4')

    svc = svm.SVC(kernel='linear')
    svc.fit(train_X, train_y)
    train_preds = svc.predict(train_X)
    test_preds = svc.predict(test_X)
    print("[linear svc]")
    print_metrics(train_y, train_preds, 'train')
    print_metrics(test_y, test_preds, 'test')

    svc = svm.SVC(kernel='poly')
    svc.fit(train_X, train_y)
    train_preds = svc.predict(train_X)
    test_preds = svc.predict(test_X)
    print("[poly svc]")
    print_metrics(train_y, train_preds, 'train')
    print_metrics(test_y, test_preds, 'test')

    svc = svm.SVC(kernel='rbf')
    svc.fit(train_X, train_y)
    train_preds = svc.predict(train_X)
    test_preds = svc.predict(test_X)
    print("[rbf svc]")
    print_metrics(train_y, train_preds, 'train')
    print_metrics(test_y, test_preds, 'test')

    print('Q1.5')

    train_X_2feat = train_X[:, :2]
    svc = svm.SVC(kernel='linear')
    svc.fit(train_X_2feat, train_y)
    plot_decision_boundary(svc, train_X_2feat, train_y, title='svc_linear')

    train_X_2feat = train_X[:, :2]
    svc = svm.SVC(kernel='poly')
    svc.fit(train_X_2feat, train_y)
    plot_decision_boundary(svc, train_X_2feat, train_y, title='svc_poly')

    train_X_2feat = train_X[:, :2]
    svc = svm.SVC(kernel='rbf')
    svc.fit(train_X_2feat, train_y)
    plot_decision_boundary(svc, train_X_2feat, train_y, title='svc_rbf')

def print_metrics(y_true, y_pred, set_name):
    print(f"metrics for {set_name} set")
    print(f"f1 score: {metrics.f1_score(y_true, y_pred)}")
    print(f"precision: {metrics.precision_score(y_true, y_pred)}")
    print(f"recall: {metrics.recall_score(y_true, y_pred)}")
    print()

if __name__ == '__main__':
    main()
