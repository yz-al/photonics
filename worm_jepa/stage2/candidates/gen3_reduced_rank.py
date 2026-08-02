# Claude-as-operator, gen 3.
# Gen2 beat the bar on prediction but structure plateaued (~0.24). Insight: the
# true neuron coupling W@W.T is LOW RANK (rank = #latents), yet every method fits
# a full-rank operator, diluting connectome recovery. Fit the linear core by
# reduced-rank regression (SVD-truncate the ridge operator) so it concentrates on
# the true coupling directions, then add a small elementwise nonlinear residual
# for prediction. Auto-select the rank on a held-out slice of train.

def build(X_train, D_train):
    mu = X_train.mean(0)
    Xc = X_train - mu
    cut = int(0.8 * len(X_train))
    Xa, Da, Xv, Dv = Xc[:cut], D_train[:cut], Xc[cut:], D_train[cut:]
    N = X_train.shape[1]

    A_full = np.linalg.solve(Xa.T @ Xa + 1.0 * np.eye(N), Xa.T @ Da)
    Ua, Sa, Vta = np.linalg.svd(A_full, full_matrices=False)

    def rr(r):
        return (Ua[:, :r] * Sa[:r]) @ Vta[:r]

    def r2(Y, Yh):
        ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
        return (1 - ss / st).mean()

    best_r, best = None, -1e9
    for r in (4, 6, 8, 10, 14, 20):
        s = r2(Dv, Xv @ rr(r))
        if s > best:
            best, best_r = s, r
    A_r = rr(best_r)

    # nonlinear residual on the low-rank linear fit
    R = Da - Xa @ A_r

    def nonlin(Xc_):
        return np.concatenate([softplus(Xc_), np.tanh(Xc_), Xc_ ** 2], axis=1)

    G = nonlin(Xa); gmu = G.mean(0); gsd = G.std(0) + 1e-8
    Gs = (G - gmu) / gsd
    Cn = np.linalg.solve(Gs.T @ Gs + 3.0 * np.eye(Gs.shape[1]), Gs.T @ R)

    def predict(X):
        Xc_ = X - mu
        return Xc_ @ A_r + ((nonlin(Xc_) - gmu) / gsd) @ Cn
    return predict
