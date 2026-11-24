/*
 * Copyright 2026 Brian Curry / Vector1 Research
 * Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
 *
 * Papilon scenario planner engine. Pure functions, no DOM, so the same code runs in the
 * page and under Node in Papilon's tests.
 *
 * A channel's steady-state contribution per period at constant spend x, for posterior
 * draw d, is  sum_g  w[d][g] * f(x * s[g]; half_sat[d], shape[d])  with f the channel's
 * saturation curve (hill, s_curve or logistic). A national model has one component; a
 * regional model has one per region. This is the same quantity MMM._steady sums.
 */
(function (root) {
  "use strict";

  function sat(kind, a, hs, sh) {
    if (a <= 0) return 0;
    if (kind === "logistic") { const e = Math.exp(-a / hs); return (1 - e) / (1 + e); }
    if (kind === "hill") return a / (a + hs);
    const as = Math.pow(a, sh);
    return as / (as + Math.pow(hs, sh));
  }

  function Model(payload) {
    this.p = payload;
    this.media = payload.media;
    this.K = payload.media.length;
    this.n = payload.curves.half_sat.length;
    this._mean = new Map();
  }

  /* Contribution of channel j at spend x, one value per draw. */
  Model.prototype.draws = function (j, x, out) {
    const c = this.p.curves, n = this.n, kind = c.saturation[j], s = c.scales[j];
    out = out || new Float64Array(n);
    for (let d = 0; d < n; d++) {
      const w = c.weights[j][d], hs = c.half_sat[d][j], sh = c.shape[d][j];
      let v = 0;
      for (let g = 0; g < s.length; g++) v += w[g] * sat(kind, x * s[g], hs, sh);
      out[d] = v;
    }
    return out;
  };

  Model.prototype.mean = function (j, x) {
    const key = j + ":" + x;
    let v = this._mean.get(key);
    if (v === undefined) {
      const a = this.draws(j, x);
      v = 0; for (let d = 0; d < a.length; d++) v += a[d];
      v /= a.length;
      if (this._mean.size > 200000) this._mean.clear();
      this._mean.set(key, v);
    }
    return v;
  };

  Model.prototype.marginal = function (j, x) {
    const h = Math.max(0.01 * x, 1e-6 * (this.p.scale[j] || 1));
    return (this.mean(j, x + h) - this.mean(j, x)) / h;
  };

  /* Total contribution per draw at spend vector x. */
  Model.prototype.total = function (x) {
    const t = new Float64Array(this.n), buf = new Float64Array(this.n);
    for (let j = 0; j < this.K; j++) { this.draws(j, x[j], buf); for (let d = 0; d < this.n; d++) t[d] += buf[d]; }
    return t;
  };

  function quantile(arr, q) {
    const a = Float64Array.from(arr).sort();
    const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }
  function mean(arr) { let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; }
  function sum(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s; }

  /* Closest point to v with sum T and lo <= x <= hi (bisection on a shift). */
  function project(v, lo, hi, T) {
    const K = v.length;
    const at = (lam) => v.map((x, j) => Math.min(hi[j], Math.max(lo[j], x + lam)));
    const span = Math.max(1, T, ...hi) + Math.max(...v.map(Math.abs));
    let a = -span, b = span;
    for (let it = 0; it < 200; it++) {
      const m = (a + b) / 2;
      if (sum(at(m)) < T) a = m; else b = m;
    }
    const x = at((a + b) / 2);
    const gap = T - sum(x);           // spread any rounding left over onto channels with room
    for (let j = 0, r = gap; j < K && Math.abs(r) > 1e-12; j++) {
      const room = r > 0 ? hi[j] - x[j] : lo[j] - x[j];
      const step = r > 0 ? Math.min(r, room) : Math.max(r, room);
      x[j] += step; r -= step;
    }
    return x;
  }

  /*
   * Exact allocation of a separable objective on a grid (dynamic programming), for
   * the expected outcome. Works for S-curves where greedy allocation stalls at zero.
   * Returns, for every budget level b*u above sum(lo), the best value and the plan.
   */
  function allocate(model, lo, hi, budget, N) {
    const K = model.K, u = budget / N;
    const g = [], m = [];
    for (let j = 0; j < K; j++) {
      const mj = Math.min(N, Math.floor((hi[j] - lo[j]) / u + 1e-9));
      m.push(mj);
      const row = new Float64Array(mj + 1);
      for (let k = 0; k <= mj; k++) row[k] = model.mean(j, lo[j] + k * u);
      g.push(row);
    }
    let F = new Float64Array(N + 1);
    const choice = [];
    const c0 = new Int32Array(N + 1);
    for (let b = 0; b <= N; b++) { const k = Math.min(b, m[0]); F[b] = g[0][k]; c0[b] = k; }
    choice.push(c0);
    for (let j = 1; j < K; j++) {
      const G = new Float64Array(N + 1).fill(-Infinity), cj = new Int32Array(N + 1), row = g[j];
      for (let b = 0; b <= N; b++) {
        let best = -Infinity, arg = 0;
        const top = Math.min(b, m[j]);
        for (let k = 0; k <= top; k++) {
          const v = F[b - k] + row[k];
          if (v > best) { best = v; arg = k; }
        }
        G[b] = best; cj[b] = arg;
      }
      F = G; choice.push(cj);
    }
    const plan = (b) => {
      const x = new Array(K);
      for (let j = K - 1; j >= 0; j--) { const k = choice[j][b]; x[j] = lo[j] + k * u; b -= k; }
      return x;
    };
    return { value: F, plan, u };
  }

  /*
   * Best-improvement search over transfers between pairs of channels, with shrinking
   * steps. objective(totalDraws) is maximized; base is subtracted for the quantile case.
   */
  function transfers(model, x, lo, hi, objective, opts) {
    const K = model.K, n = model.n, T = sum(x);
    x = x.slice();
    const C = []; for (let j = 0; j < K; j++) C.push(model.draws(j, x[j]));
    const tot = new Float64Array(n);
    for (let j = 0; j < K; j++) for (let d = 0; d < n; d++) tot[d] += C[j][d];
    let best = objective(tot);
    let step = (opts && opts.step) || Math.max(T, 1) / 8;
    const stop = Math.max(T, 1) * 1e-5;
    const cand = new Float64Array(n);
    let evals = 0;
    while (step > stop) {
      let improved = true, guard = 0;
      while (improved && guard++ < 400) {
        improved = false;
        const down = [], up = [];
        for (let j = 0; j < K; j++) {
          down.push(x[j] - step >= lo[j] - 1e-12 ? model.draws(j, x[j] - step) : null);
          up.push(x[j] + step <= hi[j] + 1e-12 ? model.draws(j, x[j] + step) : null);
        }
        let bi = -1, bj = -1, bv = best;
        for (let i = 0; i < K; i++) {
          if (!down[i]) continue;
          for (let j = 0; j < K; j++) {
            if (i === j || !up[j]) continue;
            const Di = down[i], Uj = up[j], Ci = C[i], Cj = C[j];
            for (let d = 0; d < n; d++) cand[d] = tot[d] - Ci[d] + Di[d] - Cj[d] + Uj[d];
            const v = objective(cand); evals++;
            if (v > bv + 1e-12 * Math.max(1, Math.abs(bv))) { bv = v; bi = i; bj = j; }
          }
        }
        if (bi >= 0) {
          for (let d = 0; d < n; d++) tot[d] = tot[d] - C[bi][d] + down[bi][d] - C[bj][d] + up[bj][d];
          C[bi] = down[bi]; C[bj] = up[bj];
          x[bi] -= step; x[bj] += step; best = bv; improved = true;
        }
      }
      step /= 2;
    }
    return { x, value: best, evals };
  }

  function limits(model, start, total, constraints) {
    // constraints: {maxChange (fraction or null), cap (multiple of history), lo:[], hi:[], locked:[]}
    const K = model.K, hist = model.p.historical_spend;
    const startMix = start.map((v) => v / Math.max(sum(start), 1e-12) * total);
    const lo = [], hi = [];
    for (let j = 0; j < K; j++) {
      let l = 0, h = (constraints.cap || 3) * hist[j];
      const ref = startMix[j];
      if (constraints.maxChange != null) { l = Math.max(l, ref * (1 - constraints.maxChange)); h = Math.min(h, ref * (1 + constraints.maxChange)); }
      if (constraints.lo && constraints.lo[j] != null) l = Math.max(l, constraints.lo[j]);
      if (constraints.hi && constraints.hi[j] != null) h = Math.min(h, constraints.hi[j]);
      if (constraints.locked && constraints.locked[j]) { l = constraints.lockedAt ? constraints.lockedAt[j] : ref; h = l; }
      if (h < l) h = l;
      lo.push(l); hi.push(h);
    }
    return { lo, hi, startMix };
  }

  /*
   * Plan a budget. opts:
   *   mode: "fixed" | "target_roi" | "target_mroi"
   *   start: starting spend per period (the comparison mix)
   *   total: budget per period (fixed mode)
   *   objective: "quantile" (protect the downside, Papilon's default) | "mean"
   *   quantile: 0.10
   *   target: ROI or marginal ROI target (flexible modes)
   *   constraints: see limits()
   * Returns {spend, notes[], feasible}.
   */
  function plan(model, opts) {
    const notes = [];
    const t0 = Date.now();
    if (opts.mode === "target_mroi") {
      const { lo, hi } = limitsFlexible(model, opts.start, opts.constraints);
      const x = [];
      for (let j = 0; j < model.K; j++) {
        let v = lo[j];
        const steps = 400;
        for (let k = steps; k >= 0; k--) {
          const s = lo[j] + (hi[j] - lo[j]) * k / steps;
          if (model.marginal(j, s) >= opts.target) { v = s; break; }
        }
        x.push(v);
        if (v >= hi[j] - 1e-9 && hi[j] > lo[j]) notes.push(model.media[j] + " reached its upper limit before its marginal return fell to the target.");
        if (v <= lo[j] + 1e-9 && model.marginal(j, lo[j]) < opts.target) notes.push(model.media[j] + " is below the target even at its lower limit, so it stays there.");
      }
      return { spend: x, notes, feasible: true, ms: Date.now() - t0 };
    }
    if (opts.mode === "target_roi") {
      const { lo, hi } = limitsFlexible(model, opts.start, opts.constraints);
      const sLo = sum(lo), sHi = sum(hi), N = 600;
      if (sHi - sLo < 1e-9) return { spend: lo, notes: ["The channel limits fix every channel; there is nothing to plan."], feasible: true, ms: Date.now() - t0 };
      const A = allocate(model, lo, hi, sHi - sLo, N);
      let pick = -1;
      for (let b = N; b >= 0; b--) {
        const spend = sLo + b * A.u;
        if (spend > 0 && A.value[b] / spend >= opts.target) { pick = b; break; }
      }
      if (pick < 0) {
        notes.push("No budget within the channel limits reaches a total ROI of " + opts.target + ". Showing the smallest budget the limits allow.");
        return { spend: lo, notes, feasible: false, ms: Date.now() - t0 };
      }
      let x = A.plan(pick);
      x = transfers(model, x, lo, hi, mean, { step: A.u }).x;
      if (pick === N) notes.push("Every channel reached its upper limit with total ROI still above the target, so the plan stops at the limits.");
      return { spend: x, notes, feasible: true, ms: Date.now() - t0 };
    }
    // fixed budget
    const total = opts.total;
    const L = limits(model, opts.start, total, opts.constraints);
    const lo = L.lo, hi = L.hi;
    if (sum(lo) > total + 1e-9 || sum(hi) < total - 1e-9) {
      notes.push("The channel limits cannot be met with this total budget. Widen the limits or change the budget.");
      return { spend: L.startMix, notes, feasible: false, ms: Date.now() - t0 };
    }
    const N = 400, free = total - sum(lo);
    let xm;
    if (free < 1e-12) xm = lo.slice();
    else {
      const A = allocate(model, lo, hi, free, N);
      xm = project(A.plan(N), lo, hi, total);
      xm = transfers(model, xm, lo, hi, mean, { step: A.u }).x;
    }
    if (opts.objective !== "quantile") return { spend: xm, notes, feasible: true, ms: Date.now() - t0 };
    const q = opts.quantile == null ? 0.10 : opts.quantile;
    const base = model.total(L.startMix);
    const obj = (t) => { const g = new Float64Array(t.length); for (let d = 0; d < t.length; d++) g[d] = t[d] - base[d]; return quantile(g, q); };
    const x0 = project(L.startMix, lo, hi, total);
    // the quantile surface is rough: start from the best point on the path from the
    // starting mix to the expected-outcome optimum, and from the starting mix itself
    let bestPath = x0, bestVal = -Infinity;
    for (let k = 0; k <= 20; k++) {
      const t = k / 20, p = x0.map((v, j) => v + t * (xm[j] - v));
      const v = obj(model.total(p));
      if (v > bestVal) { bestVal = v; bestPath = p; }
    }
    const a = transfers(model, bestPath, lo, hi, obj, { step: Math.max(total, 1) / 16 });
    const b = transfers(model, x0, lo, hi, obj, { step: Math.max(total, 1) / 16 });
    const r = a.value >= b.value ? a : b;
    return { spend: r.x, notes, feasible: true, ms: Date.now() - t0 };
  }

  function limitsFlexible(model, start, constraints) {
    // flexible budgets: limits are relative to each channel's starting spend
    const K = model.K, hist = model.p.historical_spend, lo = [], hi = [];
    for (let j = 0; j < K; j++) {
      let l = 0, h = (constraints.cap || 3) * hist[j];
      if (constraints.maxChange != null) { l = Math.max(l, start[j] * (1 - constraints.maxChange)); h = Math.min(h, start[j] * (1 + constraints.maxChange)); }
      if (constraints.lo && constraints.lo[j] != null) l = Math.max(l, constraints.lo[j]);
      if (constraints.hi && constraints.hi[j] != null) h = Math.min(h, constraints.hi[j]);
      if (constraints.locked && constraints.locked[j]) { l = constraints.lockedAt ? constraints.lockedAt[j] : start[j]; h = l; }
      if (h < l) h = l;
      lo.push(l); hi.push(h);
    }
    return { lo, hi };
  }

  /* Everything the page shows about a spend vector, compared with a reference vector. */
  function evaluate(model, x, ref) {
    const K = model.K, n = model.n;
    const tot = model.total(x), base = model.total(ref);
    const lift = new Float64Array(n);
    for (let d = 0; d < n; d++) lift[d] = tot[d] - base[d];
    const spend = sum(x);
    const roiDraws = new Float64Array(n);
    for (let d = 0; d < n; d++) roiDraws[d] = spend > 0 ? tot[d] / spend : 0;
    const channels = [];
    for (let j = 0; j < K; j++) {
      const c = model.draws(j, x[j]);
      const m = mean(c);
      channels.push({
        channel: model.media[j], spend: x[j], contribution: m,
        contribution_p05: quantile(c, 0.05), contribution_p95: quantile(c, 0.95),
        roi: x[j] > 0 ? m / x[j] : null, mroi: model.marginal(j, x[j]),
        beyond_history: x[j] > model.p.support_high[j] * (1 + 1e-9),
      });
    }
    return {
      spend, contribution: mean(tot), contribution_p05: quantile(tot, 0.05), contribution_p95: quantile(tot, 0.95),
      roi: spend > 0 ? mean(tot) / spend : null, roi_p05: quantile(roiDraws, 0.05), roi_p95: quantile(roiDraws, 0.95),
      lift: mean(lift), lift_p05: quantile(lift, 0.05), lift_p95: quantile(lift, 0.95),
      prob_better: lift.reduce((a, v) => a + (v > 0 ? 1 : 0), 0) / n,
      channels,
    };
  }

  /* Mean curve with a 90% band at nPts spend levels from 0 to xmax. */
  function curve(model, j, xmax, nPts) {
    const out = [];
    for (let k = 0; k <= nPts; k++) {
      const x = xmax * k / nPts, c = model.draws(j, x);
      out.push({ x, mean: mean(c), p05: quantile(c, 0.05), p95: quantile(c, 0.95) });
    }
    return out;
  }

  const api = { Model, plan, evaluate, curve, project, allocate, transfers, limits, quantile, sat };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PapilonEngine = api;
})(typeof window !== "undefined" ? window : this);
