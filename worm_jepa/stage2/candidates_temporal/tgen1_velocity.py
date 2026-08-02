# AlphaEvolve (Claude-as-operator), TEMPORAL gen 1.
# Fair input: X = [x[t], x[t-1], x[t-2]] stacked (in_dim = 3N). The diagnostic
# showed a single frame is phase-ambiguous; three frames carry velocity and
# acceleration, which is what predicting the delta actually needs. This program
# reconstructs velocity/acceleration explicitly from the stacked frames, then
# regresses the delta on [x_t, vel, acc + elementwise nonlinearities] with ridge.

def build(X_train, D_train):
    N = D_train.shape[1]

    def raw(X):
        xt, xt1, xt2 = X[:, :N], X[:, N:2 * N], X[:, 2 * N:3 * N]
        vel = xt - xt1                 # velocity
        acc = xt - 2 * xt1 + xt2       # acceleration
        return np.concatenate(
            [xt, vel, acc, softplus(xt), np.tanh(vel), vel * xt], axis=1)

    T = raw(X_train)
    cmu, csd = T.mean(0), T.std(0) + 1e-8
    Ts = (T - cmu) / csd
    ridge = 3.0
    C = np.linalg.solve(Ts.T @ Ts + ridge * np.eye(Ts.shape[1]), Ts.T @ D_train)

    def predict(X):
        return ((raw(X) - cmu) / csd) @ C
    return predict
