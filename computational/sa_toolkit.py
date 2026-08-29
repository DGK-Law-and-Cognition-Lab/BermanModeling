"""
Hand-implemented global sensitivity analysis toolkit (no SALib).
Sobol (Saltelli sampling, Saltelli-2010 first-order + Jansen-1999 total) with
bootstrap CIs, and Morris elementary-effects screening (mu, mu*, sigma).
All inputs are in the unit hypercube [0,1]^D; callers map to real ranges.
"""
import numpy as np


# ===========================================================================
# SOBOL
# ===========================================================================
def _base_sample(N, D, seed, qmc=True):
    """Return two (N,D) base matrices A,B in [0,1]. Sobol' QMC if available."""
    if qmc:
        try:
            from scipy.stats import qmc
            sob = qmc.Sobol(d=2 * D, scramble=True, seed=seed)
            m = int(round(np.log2(N)))
            X = sob.random_base2(m=m) if 2 ** m == N else sob.random(N)
            return X[:, :D], X[:, D:]
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    return rng.random((N, D)), rng.random((N, D))


def saltelli_design(N, D, seed=0, qmc=True):
    """Saltelli matrices. Returns (X, N, D). X has N*(D+2) rows in block order
    A, B, AB_0, ..., AB_{D-1}; AB_i = A with column i replaced by B's column i."""
    A, B = _base_sample(N, D, seed, qmc=qmc)
    blocks = [A, B]
    for i in range(D):
        ab = A.copy()
        ab[:, i] = B[:, i]
        blocks.append(ab)
    return np.vstack(blocks), N, D


def sobol_indices(Y, N, D, bootstrap=1000, seed=0, conf=0.95, clip=False):
    """First-order S_i (Saltelli 2010) and total-order S_T (Jansen 1999) from
    outputs Y of saltelli_design's X (same row order). Bootstrap CIs over the N
    base rows. If clip, negative point estimates are floored at 0 (report both)."""
    Y = np.asarray(Y, float).ravel()
    f_A = Y[:N]
    f_B = Y[N:2 * N]
    f_AB = np.array([Y[(2 + i) * N:(3 + i) * N] for i in range(D)])  # (D,N)

    def _idx(fA, fB, fAB):
        var = np.var(np.concatenate([fA, fB]), ddof=1)
        S1 = np.array([np.mean(fB * (fAB[i] - fA)) for i in range(D)]) / var
        ST = np.array([0.5 * np.mean((fA - fAB[i]) ** 2) for i in range(D)]) / var
        return S1, ST

    S1, ST = _idx(f_A, f_B, f_AB)
    rng = np.random.default_rng(seed)
    S1b = np.empty((bootstrap, D)); STb = np.empty((bootstrap, D))
    for b in range(bootstrap):
        idx = rng.integers(0, N, N)
        S1b[b], STb[b] = _idx(f_A[idx], f_B[idx], f_AB[:, idx])
    lo, hi = 100 * (1 - conf) / 2, 100 * (1 + conf) / 2
    out = {
        "S1": np.clip(S1, 0, None) if clip else S1,
        "ST": np.clip(ST, 0, None) if clip else ST,
        "S1_raw": S1, "ST_raw": ST,
        "S1_conf": np.percentile(S1b, [lo, hi], axis=0).T,
        "ST_conf": np.percentile(STb, [lo, hi], axis=0).T,
        "S1_se": S1b.std(axis=0, ddof=1),
        "ST_se": STb.std(axis=0, ddof=1),
    }
    return out


# ===========================================================================
# MORRIS elementary effects (Morris 1991 trajectories; Campolongo mu*)
# ===========================================================================
def morris_sample(D, r, levels=4, seed=0):
    """r trajectories, each D+1 points, one factor changed per step.
    Returns (X, delta) with X shape (r*(D+1), D) on a p-level grid in [0,1]."""
    rng = np.random.default_rng(seed)
    p = levels
    delta = p / (2.0 * (p - 1))          # delta in [0,1] units (Morris standard)
    grid = np.arange(p) / (p - 1.0)
    traj = []
    for _ in range(r):
        # base x* chosen so that x* + delta stays in [0,1]
        x_star = rng.choice(grid[grid <= 1.0 - delta + 1e-12], size=D)
        B = np.tril(np.ones((D + 1, D)), -1)          # strictly lower triangular
        J = np.ones((D + 1, D))
        Dstar = np.diag(rng.choice([-1.0, 1.0], size=D))
        perm = rng.permutation(D)
        Pstar = np.eye(D)[:, perm]
        Bstar = (J * x_star + (delta / 2.0) * ((2.0 * B - J) @ Dstar + J)) @ Pstar
        Bstar = np.clip(Bstar, 0.0, 1.0)
        traj.append(Bstar)
    return np.vstack(traj), delta


def morris_analyze(X, Y, D, r):
    """Elementary effects -> mu, mu*, sigma per factor (auto-detect step delta)."""
    Y = np.asarray(Y, float).ravel()
    EE = [[] for _ in range(D)]
    block = D + 1
    for t in range(r):
        xs = X[t * block:(t + 1) * block]
        ys = Y[t * block:(t + 1) * block]
        for s in range(D):
            dx = xs[s + 1] - xs[s]
            j = int(np.argmax(np.abs(dx)))
            if dx[j] == 0:
                continue
            EE[j].append((ys[s + 1] - ys[s]) / dx[j])
    mu = np.array([np.mean(e) if e else np.nan for e in EE])
    mu_star = np.array([np.mean(np.abs(e)) if e else np.nan for e in EE])
    sigma = np.array([np.std(e, ddof=1) if len(e) > 1 else np.nan for e in EE])
    return {"mu": mu, "mu_star": mu_star, "sigma": sigma}
