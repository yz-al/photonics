# Claude-as-operator, gen 1.
# Hypothesis: the dynamics live in a low-dim latent space (activity = nonlinear
# readout of a few latents), and the delta depends on INTERACTIONS between
# latents -- which the elementwise SINDy/linear baselines cannot express.
# Program: recover a k-dim latent projection via SVD of the (centered) state,
# then regress the delta on [latents, squares, PAIRWISE PRODUCTS, softplus] with
# ridge. The pairwise cross-terms are the new ingredient.

def build(X_train, D_train):
    k = 8
    mu = X_train.mean(0)
    Xc = X_train - mu
    # principal directions of the state -> latent projection P (N, k)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:k].T
    zsd = (Xc @ P).std(0) + 1e-8

    def feats(X):
        Z = ((X - mu) @ P) / zsd            # (B, k) standardized latents
        B, kk = Z.shape
        cols = [np.ones((B, 1)), Z, Z ** 2, softplus(Z)]
        # pairwise products z_i * z_j (i<j)
        pair = []
        for i in range(kk):
            for j in range(i + 1, kk):
                pair.append((Z[:, i] * Z[:, j])[:, None])
        if pair:
            cols.append(np.concatenate(pair, axis=1))
        return np.concatenate(cols, axis=1)

    T = feats(X_train)
    ridge = 1e-2
    C = np.linalg.solve(T.T @ T + ridge * np.eye(T.shape[1]), T.T @ D_train)

    def predict(X):
        return feats(X) @ C
    return predict
