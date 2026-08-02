# AlphaEvolve winner from the MAP-Elites run (best structure recovery).
# Genome: coupling=corr, couple_vel=True, ridge=10, no elementwise nonlinearities.
# Key idea: a data-driven neuron-coupling matrix M = correlation of activity
# (which reflects which neurons share latents) used as a MESSAGE-PASSING term, so
# the model's effective operator carries the real coupling structure. Velocity is
# injected only THROUGH the coupling (vel @ M.T). Result on the temporal problem:
# struct_corr 0.336 (vs 0.248 for the non-coupling temporal model), pred_r2 0.780.

def build(X_train, D_train):
    N = D_train.shape[1]
    xt_tr = X_train[:, :N]
    M = np.corrcoef(xt_tr.T)
    M = np.nan_to_num(M)
    for i in range(N):
        M[i, i] = 0.0

    def raw(X):
        xt, xt1 = X[:, :N], X[:, N:2 * N]
        vel = xt - xt1
        return np.concatenate([xt, xt @ M.T, vel @ M.T], axis=1)   # coupled activity + coupled velocity

    T = raw(X_train)
    cmu, csd = T.mean(0), T.std(0) + 1e-8
    Ts = (T - cmu) / csd
    ridge = 10.0
    C = np.linalg.solve(Ts.T @ Ts + ridge * np.eye(Ts.shape[1]), Ts.T @ D_train)

    def predict(X):
        return ((raw(X) - cmu) / csd) @ C
    return predict
