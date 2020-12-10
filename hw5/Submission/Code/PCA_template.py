# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Load data
    X = np.load('../../Data/X_train.npy')
    Y = np.load('../../Data/y_train.npy')
    #%% Plotting mean of the whole dataset
    X_mean = np.mean(X, axis=0)
    plt.imshow(X_mean.reshape(28,28))
    plt.savefig('../Figures/mean.png')
    plt.show()

    #%% Plotting each digit
    classes = sorted(set(Y))
    for cl in classes:
        cls_mean = np.mean(X[Y == cl, :], axis=0)
        print(f'Showing mean for class {cl}')
        plt.imshow(cls_mean.reshape(28,28))
        plt.savefig(f'../Figures/mean_cls_{cl}.png')
        plt.show()

    #%% Center the data (subtract the mean)
    X = X - X_mean

    #%% Calculate Covariate Matrix
    print('Computing covariance matrix')
    Sigma = np.cov(X, rowvar=False)
    #np.savetxt('./PCA_Sigma.csv', Sigma, delimiter=',')
    print('Done')
    #print('Shape:', Sigma.shape)

    #%% Calculate eigen values and vectors
    print('Computing eigenvalues/vectors of covariance matrix')
    '''
    try:
        Lambda = np.load('./PCA_eigenvalues.npy')
        V = np.load('./PCA_eigenvectors.npy')
    except IOError:
        Lambda, V = np.linalg.eig(Sigma)
        np.save('./PCA_eigenvalues.npy', Lambda)
        np.save('./PCA_eigenvectors.npy', V)
    '''
    Lambda, V = np.linalg.eig(Sigma)
    #np.savetxt('./PCA_eigenvalues.csv', Lambda, delimiter=',')
    #np.savetxt('./PCA_eigenvectors.csv', V, delimiter=',')
    print('Done')
    #print('Eigenvectors shape:', Lambda.shape)
    #print('Eigenvalues shape:', V.shape)

    #%% Plot eigen values
    xs = range(1, len(Lambda)+1)
    ys = np.real(Lambda)
    plt.xlabel('k')
    plt.ylabel('lambda_k')
    plt.plot(xs, ys)
    plt.savefig('../Figures/eigenvalues.png')
    plt.show()

    #%% Plot the 5 first eigen vectors as images
    vectors = V[:, :5].T # they're in the columns, not the rows
    vectors = np.real(vectors)
    for i, v in enumerate(vectors):
        plt.imshow(v.reshape(28,28))
        plt.savefig(f'../Figures/pc_{i+1}.png')
        plt.show()

    #%% Project to two first bases
    v1, v2 = V[:, 0], V[:, 1]
    #v1 /= np.linalg.norm(v1)
    #v2 /= np.linalg.norm(v2)

    xs = np.matmul(X, v1)
    ys = np.matmul(X, v2)
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    color = np.array(['']*10000, dtype=object)
    color_mapping = {
        0: 'aqua',
        1: 'azure',
        2: 'beige',
        3: 'black',
        4: 'chartreuse',
        5: 'coral',
        6: 'crimson',
        7: 'cyan',
        8: 'fuchsia',
        9: 'green'
    }
    for cl in classes:
        color[Y == cl] = color_mapping[cl]
    assert(len(np.where(color == '')[0]) == 0)
    #print(color)
    #plt.legend(range(10))
    plt.scatter(xs, ys, c=color)
    '''
    ax = plt.gca()
    leg = ax.get_legend()
    for cl in classes:
        leg.legendHandles[cl].set_color(color_mapping[cl])
    '''
    plt.savefig('../Figures/first_2_pcs.png')
    plt.show()

    #%% Plotting the projected data as scatter plot

if __name__ == '__main__':
    main()
