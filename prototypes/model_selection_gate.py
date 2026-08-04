"""
Prototype 2: Discrete model selection in the ESS-gated refinement loop.
Papilon RFC 001 — open question: "configuration choices (model family
selection) need a discrete head handled by mixture-over-models
importance sampling."

Setup
-----
True data-generating process uses DELAYED adstock (effect peaks at lag 2,
common for CTV/upper-funnel). The meta-learner must choose between:

    M_GEO     geometric adstock  (peaks at lag 0)   theta = (alpha, k, beta, sigma)
    M_DELAY   delayed adstock    (free peak lag)    theta = (alpha, mu_lag, k, beta, sigma)

The amortized head emits:  P(m) over families  +  a Gaussian proposal q_m
per family. For each family, importance sampling against the true
likelihood yields an unbiased marginal-likelihood estimate:

    Z_hat_m = mean_i [ p(theta_i, D | m) / q_m(theta_i) ]

and the corrected model posterior is  P(m | D) ∝ P(m) * Z_hat_m.

The property under test: even when the amortized head is CONFIDENTLY
WRONG about the family (P(M_GEO)=0.9), the Bayes factor from the data
overrides it — the discrete choice self-corrects exactly like the
continuous parameters did in prototype 1.

Self-contained: numpy/scipy only.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(21)

# ----------------------------------------------------------------------
# Adstock families
# ----------------------------------------------------------------------

L_MAX = 10
S_HILL = 2.0


def adstock_geometric(spend, alpha):
    w = alpha ** np.arange(L_MAX)
    w /= w.sum()
    return np.convolve(spend, w)[: len(spend)]


def adstock_delayed(spend, alpha, mu_lag):
    """Weight peaks at lag mu_lag (Google-style delayed adstock)."""
    lags = np.arange(L_MAX)
    w = alpha ** ((lags - mu_lag) ** 2)
    w /= w.sum()
    return np.convolve(spend, w)[: len(spend)]


def hill(x, k, s=S_HILL):
    return x ** s / (k ** s + x ** s)


# ----------------------------------------------------------------------
# Truth: delayed adstock, peak at lag 2
# ----------------------------------------------------------------------

TRUE = dict(alpha=0.60, mu_lag=2.0, k=3.0, beta=2.2, sigma=0.35)
N_WEEKS = 130


def simulate(seed=5):
    r = np.random.default_rng(seed)
    spend = np.clip(r.gamma(2.0, 2.0, N_WEEKS)
                    + 1.5 * np.sin(np.arange(N_WEEKS) / 8), 0, None)
    mu = TRUE["beta"] * hill(
        adstock_delayed(spend, TRUE["alpha"], TRUE["mu_lag"]), TRUE["k"]) + 1.0
    return spend, mu + r.normal(0, TRUE["sigma"], N_WEEKS)


SPEND, REVENUE = simulate()

# ----------------------------------------------------------------------
# Per-family log joint  log p(theta, D | m)  in unconstrained space
#   M_GEO:   z = [logit(alpha), log k, log beta, log sigma]
#   M_DELAY: z = [logit(alpha), logit(mu_lag / 6), log k, log beta, log sigma]
# ----------------------------------------------------------------------


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _lj_logit(a):          # log|d sigmoid / dz|
    return a - 2 * np.log1p(np.exp(a))


def logjoint_geo(z):
    z = np.atleast_2d(z)
    alpha = _expit(z[:, 0]); k = np.exp(z[:, 1])
    beta = np.exp(z[:, 2]); sigma = np.exp(z[:, 3])
    out = np.empty(len(z))
    for i in range(len(z)):
        mu = beta[i] * hill(adstock_geometric(SPEND, alpha[i]), k[i]) + 1.0
        out[i] = stats.norm.logpdf(REVENUE, mu, sigma[i]).sum()
    out += (stats.beta.logpdf(alpha, 2, 2) + stats.gamma.logpdf(k, 3)
            + stats.halfnorm.logpdf(beta, scale=3)
            + stats.halfnorm.logpdf(sigma, scale=1))
    out += _lj_logit(z[:, 0]) + z[:, 1] + z[:, 2] + z[:, 3]
    return out


def logjoint_delay(z):
    z = np.atleast_2d(z)
    alpha = _expit(z[:, 0]); mu_lag = 6.0 * _expit(z[:, 1])
    k = np.exp(z[:, 2]); beta = np.exp(z[:, 3]); sigma = np.exp(z[:, 4])
    out = np.empty(len(z))
    for i in range(len(z)):
        mu = beta[i] * hill(adstock_delayed(SPEND, alpha[i], mu_lag[i]), k[i]) + 1.0
        out[i] = stats.norm.logpdf(REVENUE, mu, sigma[i]).sum()
    out += (stats.beta.logpdf(alpha, 2, 2)
            + stats.beta.logpdf(mu_lag / 6.0, 2, 2)   # peak lag prior on (0,6)
            + stats.gamma.logpdf(k, 3)
            + stats.halfnorm.logpdf(beta, scale=3)
            + stats.halfnorm.logpdf(sigma, scale=1))
    out += (_lj_logit(z[:, 0]) + _lj_logit(z[:, 1]) + np.log(6.0)
            + z[:, 2] + z[:, 3] + z[:, 4])
    return out


FAMILIES = {
    "M_GEO":   dict(logjoint=logjoint_geo,   dim=4),
    "M_DELAY": dict(logjoint=logjoint_delay, dim=5),
}

# ----------------------------------------------------------------------
# Stand-in "amortized proposals": short MCMC per family, then Gaussian fit
# (simulates a trained per-family flow head; full covariance per proto-1)
# ----------------------------------------------------------------------


def metropolis(logjoint, z0, n_steps, step, seed):
    r = np.random.default_rng(seed)
    z = z0.copy(); lp = logjoint(z[None])[0]
    out = np.empty((n_steps, len(z0)))
    for t in range(n_steps):
        zp = z + r.normal(0, step, len(z0))
        lpp = logjoint(zp[None])[0]
        if np.log(r.uniform()) < lpp - lp:
            z, lp = zp, lpp
        out[t] = z
    return out


def fit_gaussian_proposal(logjoint, dim, seed):
    draws = metropolis(logjoint, np.zeros(dim), 6000, 0.07, seed)[2500:]
    return draws.mean(0), 1.3 * np.cov(draws.T)   # mildly inflated full cov


print("Fitting stand-in amortized proposals (one-time, per family)...")
PROPOSAL = {name: fit_gaussian_proposal(f["logjoint"], f["dim"], i + 100)
            for i, (name, f) in enumerate(FAMILIES.items())}

# ----------------------------------------------------------------------
# Mixture-over-models importance sampling gate
# ----------------------------------------------------------------------


def model_selection_gate(head_probs, n=3000,
                         accept_thresh=0.10, refine_thresh=0.01):
    """head_probs: the amortized discrete head's P(m). Returns corrected
    model posterior, per-family ESS, and per-family verdicts."""
    report = {}
    log_evidence = {}
    for name, fam in FAMILIES.items():
        mean, cov = PROPOSAL[name]
        q = stats.multivariate_normal(mean, cov)
        z = q.rvs(n, random_state=rng)
        log_w = fam["logjoint"](z) - q.logpdf(z)
        # unbiased marginal likelihood estimate (log scale)
        log_evidence[name] = logsumexp(log_w) - np.log(n)
        w = np.exp(log_w - log_w.max()); w /= w.sum()
        ess = (1.0 / np.sum(w ** 2)) / n
        verdict = ("ACCEPT" if ess > accept_thresh
                   else "REFINE" if ess > refine_thresh else "ESCALATE")
        report[name] = dict(ess=ess, verdict=verdict,
                            draws=z, weights=w)

    # corrected model posterior: P(m|D) ∝ P_head(m) * Z_hat_m
    names = list(FAMILIES)
    log_post = np.array([np.log(head_probs[m]) + log_evidence[m] for m in names])
    log_post -= logsumexp(log_post)
    model_post = dict(zip(names, np.exp(log_post)))

    log_bf = log_evidence["M_DELAY"] - log_evidence["M_GEO"]
    return model_post, log_bf, log_evidence, report


def show(title, head_probs):
    model_post, log_bf, log_ev, report = model_selection_gate(head_probs)
    print(f"\n=== {title} ===")
    print(f"   amortized head P(m):   " +
          "  ".join(f"{m}={p:.2f}" for m, p in head_probs.items()))
    print(f"   log evidence:          " +
          "  ".join(f"{m}={log_ev[m]:.1f}" for m in log_ev))
    print(f"   log Bayes factor (DELAY vs GEO): {log_bf:+.1f}")
    print(f"   corrected P(m | D):    " +
          "  ".join(f"{m}={p:.3f}" for m, p in model_post.items()))
    for m, r in report.items():
        print(f"   {m:<8} ESS/n={r['ess']:.3f} -> {r['verdict']}")
    win = max(model_post, key=model_post.get)
    if win == "M_DELAY":
        r = report["M_DELAY"]
        idx = rng.choice(len(r["weights"]), 2000, p=r["weights"])
        zs = r["draws"][idx]
        mu_lag = 6.0 * _expit(zs[:, 1])
        print(f"   winning family M_DELAY: peak lag posterior "
              f"= {mu_lag.mean():.2f} ± {mu_lag.std():.2f}  (truth {TRUE['mu_lag']})")
    return model_post


if __name__ == "__main__":
    print(f"truth: DELAYED adstock, peak lag = {TRUE['mu_lag']}, "
          f"{N_WEEKS} weeks\n")

    # Scenario 1: head is right and confident
    show("Scenario 1 — head correct (P_DELAY=0.9)",
         {"M_GEO": 0.10, "M_DELAY": 0.90})

    # Scenario 2: head is CONFIDENTLY WRONG — data must override
    show("Scenario 2 — head wrong (P_GEO=0.9): Bayes factor must override",
         {"M_GEO": 0.90, "M_DELAY": 0.10})

    # Scenario 3: head maximally uncertain
    show("Scenario 3 — head uniform (0.5 / 0.5)",
         {"M_GEO": 0.50, "M_DELAY": 0.50})

    print("\nProperty demonstrated: the discrete head, like the continuous")
    print("proposals in prototype 1, is advisory. Marginal-likelihood")
    print("estimates from the same importance weights correct the model")
    print("choice, and per-family ESS gates the continuous posteriors.")
