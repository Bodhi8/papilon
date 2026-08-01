# Design Doc: Amortized Structural Inference for Papilon
## Solving sim-to-real, context length, and posterior calibration

**Status:** Draft v0.1 · Vector1 Research
**Scope:** The meta-learner layer ("zero-knob Papilon") — a prior-fitted network trained on PyCausalSim that maps an observed marketing panel directly to a posterior over structural parameters and configuration, with safety machinery that makes amortization trustworthy on real data.

---

## 0. The core design move

The three problems are usually treated separately. The unifying idea here is:

> **The amortized posterior is never the final answer. It is a *proposal* — a very good one — that gets gated, refined, and audited.**

Concretely, the inference path for a new dataset is:

```
panel ──► encoder ──► typicality gate ──► amortized posterior q(θ|D)
                            │                     │
                       (OOD? → fallback      importance refinement
                        to full MCMC,        against true likelihood
                        log episode)               │
                                             ESS diagnostic ──► accept / escalate
                                                   │
                                             conformal wrap on
                                             decision quantities
```

Each of the three "hard problems" maps to one stage of this path: sim-to-real is handled by the prior design + typicality gate + continual simulator updating; context length is handled inside the encoder; calibration is handled by the refinement step plus two audit harnesses (SBC offline, experiment-coverage online). Amortization buys speed; the gate and refinement buy correctness; the audits buy proof.

---

## 1. Sim-to-real gap

**Failure mode:** the meta-learner is a Bayesian machine whose prior *is* PyCausalSim's generative family. If real marketing environments fall outside that family, the network doesn't fail loudly — it confidently projects reality onto the nearest simulation it has seen.

### 1.1 Widen the prior deliberately (domain randomization)

Treat PyCausalSim's environment sampler as the object under engineering. The prior must be a *mixture over structural families*, not one family with random coefficients:

- **Functional-form mixtures.** Geometric AND delayed/Weibull adstock; Hill AND logistic AND piecewise-threshold saturation; additive AND multiplicative channel interaction; linear AND regime-switching baselines.
- **Structural nuisances.** Unobserved confounders of varying strength; missing channels (spend the model never sees); competitor shocks; feedback policies of different aggressiveness (last-touch chasers, ROAS-target bidders, fixed-budget pacers) — the reflexivity itself is randomized.
- **Data pathology.** Outliers, level shifts, COVID-style demand breaks, platform algorithm-change discontinuities, measurement error on both spend and outcome, missingness patterns copied from real panels.

Rule of thumb from robotics domain randomization: it is far cheaper to make the prior embarrassingly wide than to make it exactly right. The network learns to infer *which* family it is in as part of inference.

### 1.2 Detect what the prior can't cover (typicality gate)

Widening never covers everything, so the system must know when it's out of distribution:

- Train a lightweight density model (or classifier two-sample test) on the encoder's dataset embeddings of simulated panels. At inference, score the real panel's embedding.
- Additionally have the meta-learner emit a self-assessed misspecification head, trained by injecting held-out structural families the network never saw and labeling them OOD.
- **Gate behavior:** typicality below threshold → do not trust the amortized posterior. Fall back to full MCMC with weak priors, and *log the episode as a simulator gap* (see 1.3). The failure is loud, cheap, and productive.

### 1.3 Close the loop: the simulator as a living prior (MNE's job)

Every real engagement eventually yields partial ground truth — geo-lift results, holdout incrementality, realized outcomes of budget shifts. Each becomes an MNE episode: `(panel embedding, amortized posterior, refined posterior, experimental truth, gap diagnostics)`.

Two continual-learning loops run over the episode store:

1. **Empirical-Bayes hyperprior update.** Periodically refit PyCausalSim's sampling distributions to the empirical distribution of parameters recovered from real, experiment-validated engagements. The simulator's prior contracts toward reality with every client.
2. **Gap-driven family expansion.** Episodes flagged by the typicality gate are triaged: what structure did reality have that the simulator lacked? New structural families get added to the mixture, and the network is retrained. Misspecification stops being a silent risk and becomes the backlog.

This is the strongest claim in the whole architecture: **the sim-to-real gap is not a constant to be minimized once; it is a monitored, shrinking quantity with a paper trail.**

### 1.4 Robustness at training time

Two cheap additions from the robust-SBI literature: train with noise-augmented simulations (the network sees corrupted versions of its own simulator, so it can't overfit to simulator cleanliness), and mix a small fraction of real panels into training with semi-supervised objectives (masked-span prediction on real data; supervised heads only where experiment labels exist).

---

## 2. Context length

**Problem size:** a serious panel is ~200 geos × 150+ weeks × 10–20 channels plus controls — ~30k geo-weeks of multivariate series. Naive tokenization is hopeless; naive summary statistics throw away exactly the lag structure the network needs.

### 2.1 Exploit exchangeability: factorized encoder

The data has symmetry the architecture should hard-code rather than learn:

- **Geos are exchangeable** (conditional on geo-level covariates). So the geo axis gets a permutation-invariant encoder — Set Transformer / DeepSets pooling over per-geo embeddings. This also means the same network handles 20 geos or 500 without retraining.
- **Time is ordered and long.** Within each geo, encode the multichannel series with a patch-based temporal encoder (PatchTST-style: tokenize into 4–8-week patches) or a state-space backbone (Mamba-class), which handles 150+ weeks in linear time and preserves lag/carryover information that summary stats destroy.
- **Channels get an embedding table** keyed on channel *type* (search, social, CTV, …), not position — so channel count and identity can vary across engagements.

Architecture sketch:

```
per geo:   [channels × weeks] ──patch──► temporal encoder ──► geo vector
across geos:  {geo vectors} ──set-attention pool──► panel embedding
decode:    panel embedding ──cross-attention──► posterior heads:
              • global structural params (adstock α, Hill k/s per channel-type)
              • per-geo deviations (hierarchical offsets)
              • configuration (model family, refit cadence, transform choices)
              • typicality / misspecification score
```

Token math: 200 geos × ~20 patches ≈ 4k geo-patch tokens before pooling — comfortable. The factorization turns an intractable context problem into a mid-sized one.

### 2.2 Belt and suspenders: learned + engineered summaries

Alongside the raw-patch path, feed a bank of engineered sufficient-statistic tokens: spend↔outcome cross-correlograms at lags 0–12, empirical saturation curvature per channel, STL seasonal components, structural-break statistics. Classic ABC lived entirely on summaries; here they are auxiliary inputs that stabilize training and give the network a fast path for the easy part of inference. Ablate them in the paper — reviewers will ask.

### 2.3 Posterior form

Output heads are normalizing flows (or mixture-density heads) per parameter block, conditioned on the panel embedding — a full joint posterior, not marginals, because the budget optimizer downstream consumes correlated uncertainty (α and k trade off; the optimizer must see that).

---

## 3. Calibration: making the amortized posterior trustworthy

Three layers, from offline proof to online guarantee.

### 3.1 Offline: simulation-based calibration as a CI gate

SBC (Talts et al.): draw θ* from the prior, simulate a panel, run the network, compute the rank of θ* within posterior draws. Across thousands of replications, ranks must be uniform; systematic deviation localizes exactly *which* parameter's posterior is over/under-confident. Complement with expected-coverage curves (nominal vs. empirical coverage at every credibility level).

**This runs in CI.** Every retrain of the meta-learner must pass SBC before deployment, the same way unit tests gate code. Calibration regression becomes a build failure, not a client incident.

### 3.2 At inference: amortization as proposal, not verdict

The decisive trick — the amortized posterior q(θ|D) is used as an **importance-sampling proposal against the true PyMC likelihood**:

```python
theta = q.sample(n=4000)                      # amortized draws — free
log_w = model.logp(theta, D) - q.logp(theta)  # exact reweighting
ess   = effective_sample_size(log_w)

if ess / n > 0.10:      # proposal is good → reweighted posterior is
    return reweight(theta, log_w)             # asymptotically exact
elif ess / n > 0.01:    # decent init → short-run MCMC seeded at q
    return pm.sample(init=q, tune=200, draws=500)
else:                   # proposal poor → full inference + log gap episode
    escalate_to_full_mcmc(); mne.log_gap(D, q, diagnostics)
```

Properties worth stating plainly: when the network is right you get exact Bayesian inference at amortized speed; when it's wrong, the ESS *tells you so quantitatively* before any number reaches a human. The ESS ratio is the single scalar trust signal for the whole system, and it composes with the typicality gate (1.2) — one catches distributional strangeness, the other catches likelihood disagreement.

### 3.3 Online: experiments as coverage audits, conformal wrap on decisions

Real ground truth arrives continuously in this business — every geo-lift test is a labeled point. Use it twice:

- **Coverage ledger.** For every experiment, record whether realized lift fell inside the predicted interval at each nominal level. MNE stores the ledger; a dashboard tracks empirical coverage per channel-type and per client. Nominal 80% intervals catching 62% of outcomes is a detected, dated, attributable event.
- **Conformal wrap.** For the small set of *decision-facing* quantities (channel lift, marginal ROAS), conformalize the amortized intervals against the ledger. This yields finite-sample coverage guarantees that hold **even under model misspecification** — the honest promise to a client: "whatever the model's internals believe, these intervals empirically cover at their stated rate on environments like yours."

The layering is the point: SBC proves calibration *within the simulator's world*, importance refinement enforces agreement with the *likelihood on this dataset*, and conformal coverage guarantees performance in the *real world where neither may hold*.

---

## 4. What this yields

- **Product:** Papilon 3.0 inference path — point at a panel, receive a calibrated posterior over structure and config in seconds, with a trust scalar (ESS × typicality) attached to every output, and automatic escalation when trust is low. The remaining human inputs are objectives, constraints, and risk tolerance — the knobs that *should* be human.
- **Research program / paper skeleton:**
  1. Method — amortized structural inference for reflexive marketing systems (encoder factorization + flow posterior + proposal-refinement inference)
  2. Sim-to-real — domain-randomized causal prior, typicality gating, empirical-Bayes simulator updating (with the MNE episode store as the mechanism)
  3. Calibration — SBC-in-CI, ESS-gated refinement, conformal decision intervals; benchmark against Meridian/Robyn-style knob-tuned workflows on synthetic environments with known truth and on real panels with experiment ground truth
- **Evaluation plan:** (a) parameter recovery + SBC on held-out simulated families, including deliberately misspecified ones the gate must catch; (b) time-to-calibrated-posterior vs. full MCMC and vs. hand-tuned baselines; (c) decision regret — incremental revenue of budget plans under each method in simulated environments where true response is known; (d) real-world coverage ledger over ≥1 year of geo experiments.

## 5. Open questions (honest list)

- Minimum viable simulator diversity: how many structural families before the typicality gate stops firing on typical real panels? (Empirical; start logging early.)
- Importance refinement requires the amortized config to define a tractable PyMC likelihood — configuration choices (model family selection) need a discrete head handled by mixture-over-models importance sampling, which is fiddlier than the continuous case.
- Conformal exchangeability: experiments across clients aren't i.i.d.; use Mondrian/grouped conformal (per channel-type strata) and say so in the paper.
- Compute: training the PFN is the big line item; the panel-embedding factorization keeps it single-node-multi-GPU scale, but budget for repeated retrains as the simulator expands.
