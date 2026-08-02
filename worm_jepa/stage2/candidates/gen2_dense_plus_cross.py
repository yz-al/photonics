# Claude-as-operator, gen 2.
# Gen1 lesson: a low-rank latent bottleneck HURT both prediction and structure.
# The full-dimensional linear term is what carries connectome recovery, so keep
# it and AUGMENT. Two changes vs SINDy: (1) dense tuned ridge instead of hard
# thresholding (thresholding discards predictive signal); (2) add pairwise
# cross-terms among the top-k latent projections as EXTRA columns (interactions)
# on top of the full elementwise library. Standardize columns for a stable fit.

def build(X_train, D_train):
    mu = X_train.mean(0)
    Xc = X_train - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = 6
    P = Vt[:k].T
    zsd = (Xc @ P).std(0) + 1e-8

    def raw(X):
        Xc = X - mu
        Z = (Xc @ P) / zsd
        B = X.shape[0]
        cross = []
        for i in range(k):
            for j in range(i + 1, k):
                cross.append((Z[:, i] * Z[:, j])[:, None])
        cols = [X, softplus(X), np.tanh(X), X ** 2]     # full elementwise library
        if cross:
            cols.append(np.concatenate(cross, axis=1))  # latent interactions
        return np.concatenate(cols, axis=1)

    T = raw(X_train)
    cmu = T.mean(0)
    csd = T.std(0) + 1e-8
    Ts = (T - cmu) / csd
    ridge = 3.0
    C = np.linalg.solve(Ts.T @ Ts + ridge * np.eye(Ts.shape[1]), Ts.T @ D_train)

    def predict(X):
        return ((raw(X) - cmu) / csd) @ C
    return predict
