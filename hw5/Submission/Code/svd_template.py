import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def SVD(A, s, k):
    n,m = A.shape

    p = np.zeros((n,))
    A_norm_sq = np.linalg.norm(A)**2
    for i in range(n):
        row_i = A[i, :]
        row_i_norm_sq = np.linalg.norm(row_i)**2
        p[i] = row_i_norm_sq / A_norm_sq

    S = np.zeros((s,m))

    for i in range(s):
        j = np.random.choice(range(n), p=p)
        S[i, :] = A[j, :]

    SST = np.matmul(S, S.T)

    U, Sigma_sq, _ = np.linalg.svd(SST)
    Sigma = np.sqrt(Sigma_sq)
    #print(A.shape, S.shape, SST.shape, U.shape, Sigma.shape)

    H = np.zeros((m,k))
    for t in range(k):
        h_t = np.matmul(S.T, U[:, t])
        h_t /= np.linalg.norm(h_t)
        H[:, t] = h_t

    return H, Sigma[:k]
    

def main():
    np.random.seed(42)

    im = Image.open("../../Data/baboon.tiff")
    A = np.array(im)
    plt.imshow(A)
    plt.savefig('../Figures/original.png')
    plt.show()

    s, k = 80, 60

    U, Sigma, V_T = np.linalg.svd(A)
    V = V_T.T
    V_k = V[:, :k] # trim the last (D-K) columns to get a rank-K approx
    A_k = np.matmul(A, np.matmul(V_k, V_k.T))
    plt.imshow(A_k)
    plt.savefig('../Figures/optimal_rank_60_approx.png')
    plt.show()

    # TO DO: Use H to compute sub-optimal k rank approximation for A
    H, Sigma = SVD(A, s, k)
    A_k_hat = np.matmul(A, np.matmul(H, H.T))
    plt.imshow(A_k_hat)
    plt.savefig('../Figures/suboptimal_rank_60_approx.png')
    plt.show()

    # Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    # sub-optimal k-rank approximation for A using H.
    err_A_k = np.linalg.norm(A - A_k)
    print('A_k error:', err_A_k)
    err_A_k_hat = np.linalg.norm(A - A_k_hat)
    print('A_k_hat error:', err_A_k_hat)


if __name__ == "__main__":
    main()
