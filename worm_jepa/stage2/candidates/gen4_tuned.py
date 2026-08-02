# Claude-as-operator, gen 4.
# Gen2 is the champion; gen3's reduced-rank hurt. So refine the WINNER rather
# than change regime: gen2's full-elementwise + latent-cross library, but (1)
# auto-select the ridge on a held-out slice of train instead of hardcoding it,
# and (2) add a cubic elementwise term (softplus/tanh/x^2 miss odd curvature).

def build(X_train, D_train):
    mu = X_train.mean(0)
    Xc = X_train - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = 6
    P = Vt[:k].T
    zsd = (Xc @ P).std(0) + 1e-8

    def raw(X):
        Xc_ = X - mu
        Z = (Xc_ @ P) / zsd
        cross = []
        for i in range(k):
            for j in range(i + 1, k):
                cross.append((Z[:, i] * Z[:, j])[:, None])
        cols = [X, softplus(X), np.tanh(X), Xc_ ** 2, Xc_ ** 3]
        if cross:
            cols.append(np.concatenate(cross, axis=1))
        return np.concatenate(cols, axis=1)

    T = raw(X_train)
    cmu = T.mean(0); csd = T.std(0) + 1e-8
    Ts = (T - cmu) / csd

    cut = int(0.8 * len(X_train))
    Ta, Da, Tv, Dv = Ts[:cut], D_train[:cut], Ts[cut:], D_train[cut:]

    def r2(Y, Yh):
        ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
        return (1 - ss / st).mean()

    best_ridge, best = 3.0, -1e9
    for ridge in (0.3, 1.0, 3.0, 10.0, 30.0):
        C = np.linalg.solve(Ta.T @ Ta + ridge * np.eye(Ta.shape[1]), Ta.T @ Da)
        s = r2(Dv, Tv @ C)
        if s > best:
            best, best_ridge = s, ridge
    C = np.linalg.solve(Ts.T @ Ts + best_ridge * np.eye(Ts.shape[1]), Ts.T @ D_train)

    def predict(X):
        return ((raw(X) - cmu) / csd) @ C
    return predict
