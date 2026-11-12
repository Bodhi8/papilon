# RFC 003: The learning loop, conjugate geo pooling and the SMC fallback

Author: Brian Curry, Vector1 Research
Status: implemented in Papilon 1.0.0 and PyCausalSim 0.6.0
Date: 2026-11-12

## 1. The learning loop

A cycle has four steps.

1. **Learn.** Append the new periods. Score the previous cycle's forecast on them
   (coverage of the 90% interval, error, bias). Record experiment results as lift tests.
2. **Refresh.** Refit on all periods, with every recorded lift test in the likelihood.
3. **Decide.** Plan with the downside-protected rule. Rank lift tests by the expected
   improvement of the plan (RFC 002, section 2) and recommend the top one when its expected
   gain exceeds twice its simulation error. Choose an exploration pair: the pair of
   channels whose marginal returns the posterior ranks closest to a coin flip at the
   planned mix, or, if none, the channel with the widest range weighted by spend and the
   largest other channel. Pulse money between them, plus and minus `explore` times the
   smaller planned spend, in alternating blocks, keeping the total fixed.
4. **Forecast.** Forecast the quarter's schedule, to be scored next cycle.

Only new information returns to the model: periods, measurements and variation the loop
created on purpose. The model's estimates never become inputs, which is what keeps the
loop from confirming its own errors.

**Grading.** `pycausalsim.futures.simulate_loop` replays a world from `marketing_worlds`
with the spend the loop chose. Tests are applied to the world (treated regions dark for
the test weeks), measured as the true difference in sales plus noise with the standard
error of a matched-market design, and returned to the loop as a lift test whose factor
restores the untreated spend. Each cycle is graded on the error and coverage of the
estimated returns over the window seen, the true steady-state gain of the plan, and
realized sales.

## 2. Conjugate geo pooling

Per region g, conditional on shared curve parameters theta, sales are linear in the
baseline coefficients b_g and the regional effects beta_g. With

    b_g ~ N(0, sigma_g^2 diag(s_b^2)),  beta_g ~ N(mu, sigma_g^2 diag((tau mu / s_hat_g)^2)),
    sigma_g^2 ~ InvGamma(a0, b0_g),

the region's marginal likelihood is available in closed form (normal and inverse-gamma).
s_hat_g is a plug-in noise scale from a rough per-region fit, so the effect prior has
spread about tau times mu in outcome units. The sampled dimension is the shared curve
parameters plus mu and tau per channel, independent of the number of regions. Regional
effects are drawn afterwards from their conditional multivariate t.

## 3. Tempered SMC fallback

When importance sampling from the mixture proposal q fails its gate, particles from q are
moved through targets q^(1-b) p^b, with b rising from 0 to 1 in steps chosen so the
effective sample size of the incremental weights stays at half the particles. After each
step: systematic resampling and random-walk Metropolis moves under the current target
with the particle covariance, the step size adapted to the acceptance rate. The product of
the mean incremental weights estimates the evidence, so Bayes-factor comparison keeps
working when the fallback runs.
