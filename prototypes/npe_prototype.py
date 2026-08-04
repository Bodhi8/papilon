"""
Prototype 3: A trained amortized meta-learner (neural posterior estimation),
validated through the ESS gate of prototype 1.

Papilon RFC 001 — this is the first LEARNED component: an MLP trained on
PyCausalSim-style simulator draws that maps panel summary statistics to a
full-covariance Gaussian posterior over structural parameters
(mu(s), L(s) with Sigma = L L^T). Pure numpy, manual backprop — no
framework dependencies, runs anywhere.

Pipeline
--------
  1. Draw theta ~ prior, simulate a panel with the mini-MMM simulator
     (imported from prototype 1 so the DGP is identical).
  2. Compute ~30 summary statistics per panel (cross-correlograms,
     autocorrelations, moments, adstock-probe regressions).
  3. Train NPE: minimize  -log N(z_true; mu(s), L(s) L(s)^T)  over sims.
     Full Cholesky head — prototype 1 showed the k–beta correlation
     (0.93) makes diagonal heads ~3x worse as proposals.
  4. Evaluate: held-out log prob + per-parameter 80% coverage (SBC-lite).
  5. THE test: feed the 'real' panel from prototype 1 through the trained
     network, use its output as the importance proposal, and check the
     gate verdict. Success = the learned proposal earns ACCEPT (or a
     high-ESS REFINE) with zero hand-tuning.
"""

import numpy as np
from scipy import stats

# Reuse the exact DGP + gate machinery from prototype 1
from refinement_gate import (TRUE, Z_TRUE, N_WEEKS, simulate_panel,
                             adstock, hill, log_posterior, to_constrained,
                             metropolis)

rng = np.random.default_rng(99)

# ----------------------------------------------------------------------
# 1. Prior + simulation set
# ----------------------------------------------------------------------


def sample_prior(r):
    return dict(alpha=r.beta(2, 2),
                k=r.gamma(3, 1) + 0.2,
                beta=abs(r.normal(0, 3)) + 0.1,
                sigma=abs(r.normal(0, 1)) + 0.05)


def z_of_dict(t):
    return np.array([np.log(t["alpha"] / (1 - t["alpha"])),
                     np.log(t["k"]), np.log(t["beta"]), np.log(t["sigma"])])


# ----------------------------------------------------------------------
# 2. Summary statistics (the poor man's panel encoder)
# ----------------------------------------------------------------------


def _xcorr(a, b, lag):
    if lag > 0:
        a, b = a[lag:], b[:-lag]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summaries(spend, rev):
    s = []
    # cross-correlograms spend->rev, raw + differenced (lag structure)
    for lag in range(9):
        s.append(_xcorr(rev, spend, lag))
    dr, ds = np.diff(rev), np.diff(spend)
    for lag in range(9):
        s.append(_xcorr(dr, ds, lag))
    # marginal moments
    s += [rev.mean(), rev.std(), stats.skew(rev), spend.mean(), spend.std()]
    # revenue autocorrelation (carryover signature)
    for lag in range(1, 5):
        s.append(_xcorr(rev, rev, lag))
    # adstock probes: R^2 of rev ~ hill(adstock(spend, a), k) on a small grid
    for a in (0.2, 0.5, 0.8):
        for k in (1.5, 4.0):
            x = hill(adstock(spend, a), k)
            s.append(_xcorr(rev, x, 0))
    return np.array(s)


N_SUM = 33

print("Simulating training set...")
N_TRAIN, N_VAL = 9000, 1000
X = np.empty((N_TRAIN + N_VAL, N_SUM))
Zt = np.empty((N_TRAIN + N_VAL, 4))
for i in range(N_TRAIN + N_VAL):
    th = sample_prior(rng)
    sp, rv = simulate_panel(th, n=N_WEEKS, seed=int(rng.integers(1 << 30)))
    X[i] = summaries(sp, rv)
    Zt[i] = z_of_dict(th)

x_mean, x_std = X[:N_TRAIN].mean(0), X[:N_TRAIN].std(0) + 1e-8
Xn = (X - x_mean) / x_std
Xtr, Ztr = Xn[:N_TRAIN], Zt[:N_TRAIN]
Xva, Zva = Xn[N_TRAIN:], Zt[N_TRAIN:]

# ----------------------------------------------------------------------
# 3. NPE network: MLP -> (mu, Cholesky L), trained by max log-likelihood
# ----------------------------------------------------------------------

H = 64
D_OUT = 4 + 4 + 6           # mu, softplus-diag, off-diag
TRIL = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]


def init(r):
    return {
        "W1": r.normal(0, np.sqrt(2 / N_SUM), (H, N_SUM)), "b1": np.zeros(H),
        "W2": r.normal(0, np.sqrt(2 / H), (H, H)),         "b2": np.zeros(H),
        "W3": r.normal(0, 0.01, (D_OUT, H)),               "b3": np.zeros(D_OUT),
    }


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def build_L(out):
    d = softplus(out[4:8]) + 1e-3
    L = np.diag(d)
    for j, (r_, c_) in enumerate(TRIL):
        L[r_, c_] = out[8 + j]
    return L, d


def forward_backward(p, x, z_true):
    """Returns (loss, grads) for one sample. Analytic gradients through
    the Gaussian log-density with Cholesky parameterization:
        v = L^{-1}(z - mu)
        loss = 0.5 v.v + sum(log diag L)
        d/d mu   = -L^{-T} v
        d/d L    = -(L^{-T} v) v^T  (lower tri)  + diag(1/L_ii)
    """
    a1 = p["W1"] @ x + p["b1"]; h1 = np.tanh(a1)
    a2 = p["W2"] @ h1 + p["b2"]; h2 = np.tanh(a2)
    out = p["W3"] @ h2 + p["b3"]

    mu = out[:4]
    L, d = build_L(out)
    r_ = z_true - mu
    v = np.linalg.solve(L, r_)
    Ltv = np.linalg.solve(L.T, v)                 # L^{-T} v
    loss = 0.5 * v @ v + np.log(d).sum()

    g_out = np.zeros(D_OUT)
    g_out[:4] = -Ltv                               # d/d mu
    G_L = -np.outer(Ltv, v)                        # d/d L (dense)
    # diag entries: chain through softplus (+ the log-det term)
    diag_grad = np.diag(G_L) + 1.0 / d
    g_out[4:8] = diag_grad * (1 / (1 + np.exp(-out[4:8])))   # softplus'
    for j, (r__, c__) in enumerate(TRIL):
        g_out[8 + j] = G_L[r__, c__]

    # backprop MLP
    gW3 = np.outer(g_out, h2); gb3 = g_out
    dh2 = p["W3"].T @ g_out; da2 = dh2 * (1 - h2 ** 2)
    gW2 = np.outer(da2, h1); gb2 = da2
    dh1 = p["W2"].T @ da2; da1 = dh1 * (1 - h1 ** 2)
    gW1 = np.outer(da1, x); gb1 = da1
    return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                  "W3": gW3, "b3": gb3}


def gradient_check(p, x, z):
    """Numeric check on a few entries — trust but verify manual backprop."""
    _, g = forward_backward(p, x, z)
    eps, worst = 1e-5, 0.0
    for key, idx in [("W3", (2, 5)), ("W1", (3, 7)), ("b2", (4,))]:
        p[key][idx] += eps
        lp, _ = forward_backward(p, x, z)
        p[key][idx] -= 2 * eps
        lm, _ = forward_backward(p, x, z)
        p[key][idx] += eps
        num = (lp - lm) / (2 * eps)
        rel = abs(num - g[key][idx]) / (abs(num) + 1e-8)
        worst = max(worst, rel)
    return worst


def train(p, Xtr, Ztr, Xva, Zva, epochs=40, bs=128, lr=1e-3):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            g_acc = {k: np.zeros_like(vv) for k, vv in p.items()}
            for i in idx:
                loss, g = forward_backward(p, Xtr[i], Ztr[i])
                tot += loss
                for k in g_acc:
                    g_acc[k] += g[k]
            t += 1
            for k in p:                          # Adam
                g_ = g_acc[k] / len(idx)
                m[k] = 0.9 * m[k] + 0.1 * g_
                v2[k] = 0.999 * v2[k] + 0.001 * g_ ** 2
                mh = m[k] / (1 - 0.9 ** t)
                vh = v2[k] / (1 - 0.999 ** t)
                p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if ep % 5 == 0 or ep == epochs - 1:
            va = np.mean([forward_backward(p, Xva[i], Zva[i])[0]
                          for i in range(len(Xva))])
            print(f"  epoch {ep:3d}  train NLL/sample {tot/n:7.3f}"
                  f"   val NLL {va:7.3f}")
    return p


def predict(p, x):
    h1 = np.tanh(p["W1"] @ x + p["b1"])
    h2 = np.tanh(p["W2"] @ h1 + p["b2"])
    out = p["W3"] @ h2 + p["b3"]
    L, _ = build_L(out)
    return out[:4], L @ L.T


# ----------------------------------------------------------------------
# 4. Train + evaluate
# ----------------------------------------------------------------------

if __name__ == "__main__":
    params = init(rng)
    gc = gradient_check(params, Xtr[0], Ztr[0])
    print(f"gradient check (max rel err): {gc:.2e}")
    assert gc < 1e-4, "backprop is wrong — fix before training"

    print("Training NPE network...")
    params = train(params, Xtr, Ztr, Xva, Zva)

    # SBC-lite: 80% central coverage per parameter on validation sims
    zsc = stats.norm.ppf(0.90)
    cover = np.zeros(4)
    for i in range(len(Xva)):
        mu, Sig = predict(params, Xva[i])
        sd = np.sqrt(np.diag(Sig))
        cover += (np.abs(Zva[i] - mu) < zsc * sd)
    cover /= len(Xva)
    print("\n80% interval coverage on held-out sims (target 0.80):")
    for name, c in zip(["alpha", "k", "beta", "sigma"], cover):
        print(f"   {name:<6} {c:.3f}")

    # ------------------------------------------------------------------
    # 5. THE test: the 'real' panel from prototype 1 through the gate
    # ------------------------------------------------------------------
    from refinement_gate import SPEND, REVENUE
    x_real = (summaries(SPEND, REVENUE) - x_mean) / x_std
    mu_hat, Sig_hat = predict(params, x_real)

    print("\nLearned proposal on the 'real' panel (truth in brackets):")
    th_hat = to_constrained(mu_hat[None])[0]
    for name, h, t_ in zip(["alpha", "k", "beta", "sigma"], th_hat,
                           [TRUE["alpha"], TRUE["k"], TRUE["beta"], TRUE["sigma"]]):
        print(f"   {name:<6} {h:7.3f}   [{t_}]")

    # correlation the network learned for this panel
    sd_hat = np.sqrt(np.diag(Sig_hat))
    corr_kb = Sig_hat[1, 2] / (sd_hat[1] * sd_hat[2])
    print(f"   learned corr(log k, log beta) = {corr_kb:+.2f} "
          f"  (oracle posterior: +0.93)")

    # gate it — mild inflation, exactly as a deployed proposal would be
    q = stats.multivariate_normal(mu_hat, 1.5 * Sig_hat)
    n = 4000
    zq = q.rvs(n, random_state=rng)
    log_w = log_posterior(zq) - q.logpdf(zq)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    ess_ratio = (1.0 / np.sum(w ** 2)) / n
    verdict = ("ACCEPT" if ess_ratio > 0.10
               else "REFINE" if ess_ratio > 0.01 else "ESCALATE")
    print(f"\nGATE: ESS/n = {ess_ratio:.4f}  ->  {verdict}")

    if verdict != "ESCALATE":
        if verdict == "ACCEPT":
            idx = rng.choice(n, 2000, p=w)
            post = to_constrained(zq[idx])
        else:
            seed_pt = zq[np.argmax(w)]
            dr = metropolis(seed_pt, 1500)[0][500:]
            post = to_constrained(dr)
        print("final posterior (mean ± sd):")
        for name, t_, m_, s_ in zip(["alpha", "k", "beta", "sigma"],
                                    [TRUE["alpha"], TRUE["k"], TRUE["beta"],
                                     TRUE["sigma"]], post.mean(0), post.std(0)):
            print(f"   {name:<6} {m_:7.3f} ± {s_:.3f}   [truth {t_}]")

    print("\nEnd-to-end: simulator -> trained network -> proposal -> gate.")
    print("This is the RFC's inference path with a real learned component;")
    print("scaling it means replacing summaries with the factorized panel")
    print("encoder and the Gaussian head with a flow (RFC sections 2.1/2.3).")
