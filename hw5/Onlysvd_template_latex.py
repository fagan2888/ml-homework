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
