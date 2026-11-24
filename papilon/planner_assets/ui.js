/*
 * Copyright 2026 Brian Curry / Vector1 Research
 * Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
 *
 * Papilon scenario planner page: controls, results, charts. Numbers come from engine.js.
 */
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("papilon-data").textContent);
  const E = window.PapilonEngine;
  const M = new E.Model(D);
  const K = M.K;
  const $ = (s) => document.querySelector(s);
  const per = D.period, pers = D.period + "s";
  const nice = (c) => String(c).replace(/_/g, " ");
  const col = (j) => (j < 8 ? `var(--series-${j + 1})` : "var(--series-other)");
  const sum = (v) => v.reduce((a, b) => a + b, 0);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------------ formatting
  function num(v) {
    if (v == null || !isFinite(v)) return "n/a";
    const a = Math.abs(v), s = v < 0 ? "−" : "";
    if (a >= 1e9) return s + (a / 1e9).toFixed(a < 1e10 ? 2 : 1) + "B";
    if (a >= 1e6) return s + (a / 1e6).toFixed(a < 1e7 ? 2 : 1) + "M";
    if (a >= 1e4) return s + (a / 1e3).toFixed(a < 1e5 ? 1 : 0) + "K";
    if (a >= 100) return s + Math.round(a).toLocaleString("en-US");
    if (a >= 10) return s + a.toFixed(1);
    return s + a.toFixed(2);
  }
  const money = (v) => (v != null && v < 0 ? "−" : "") + D.currency + num(Math.abs(v));
  const signed = (v) => (v > 0 ? "+" : "") + num(v);
  const ratio = (v) => (v == null || !isFinite(v) ? "n/a" : v.toFixed(2));
  const pct = (v) => (!isFinite(v) ? "n/a" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0) + "%");
  const unitNote = D.units ? ` (${D.currency ? D.currency + " " : ""}${D.units})` : "";

  // ------------------------------------------------------------------ state
  const startTotal = (w) => sum(D.windows[w].spend);
  const S = {
    win: 0, mode: "fixed", total: startTotal(0), totalEdited: false, objective: "quantile",
    targetRoi: null, targetMroi: 1, maxChange: 0.5, noLimit: false, cap: 3,
    lo: Array(K).fill(null), hi: Array(K).fill(null), locked: Array(K).fill(false),
    manual: null, horizon: D.horizon, saved: [],
  };
  const start = () => D.windows[S.win].spend;
  {
    const ev = E.evaluate(M, start(), start());
    S.targetRoi = Math.max(0.05, Math.floor(ev.roi * 0.9 * 20) / 20);
  }
  let last = null;   // last computed scenario

  function constraints() {
    return { maxChange: S.noLimit ? null : S.maxChange, cap: S.cap, lo: S.lo, hi: S.hi, locked: S.locked, lockedAt: start() };
  }

  function compute() {
    const st = start();
    let spend, notes = [], ms = 0, feasible = true;
    if (S.mode === "manual") {
      if (!S.manual) S.manual = st.slice();
      spend = S.manual.slice();
    } else {
      const r = E.plan(M, {
        mode: S.mode, start: st, total: S.total, objective: S.objective, quantile: 0.1,
        target: S.mode === "target_roi" ? S.targetRoi : S.targetMroi, constraints: constraints(),
      });
      spend = r.spend; notes = r.notes; ms = r.ms; feasible = r.feasible;
    }
    const ref = S.mode === "fixed" ? st.map((v) => v / Math.max(sum(st), 1e-12) * S.total) : st;
    const ev = E.evaluate(M, spend, ref);
    const evStart = E.evaluate(M, st, st);
    return { spend, notes, ms, feasible, ev, evStart, ref, start: st, mode: S.mode, objective: S.objective,
             title: scenarioTitle(), horizon: S.horizon };
  }

  function scenarioTitle() {
    if (S.mode === "fixed") return `Best mix for ${money(S.total)} a ${per}, ` + (S.objective === "quantile" ? "protecting the downside" : "maximizing the expected outcome");
    if (S.mode === "target_roi") return `The most spend that keeps total ROI at ${ratio(S.targetRoi)} or above`;
    if (S.mode === "target_mroi") return `Each channel up to a marginal return of ${ratio(S.targetMroi)}`;
    return "Your own mix";
  }

  // ------------------------------------------------------------------ svg helpers
  const NS = "http://www.w3.org/2000/svg";
  function S_(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "style") e.setAttribute("style", attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function ticks(max, n) {
    if (!(max > 0)) return [0, 1];
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = 0; v <= max * 1.0001 + 1e-12; v += step) out.push(+v.toPrecision(12));
    if (out[out.length - 1] < max) out.push(+(out[out.length - 1] + step).toPrecision(12));
    return out;
  }
  const tip = $("#tip");
  function showTip(evt, html) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  const row = (k, v) => `<div class="r"><span>${k}</span><span>${v}</span></div>`;
  const width = (el, fallback) => Math.max(280, el.clientWidth || fallback || 800);

  // ------------------------------------------------------------------ header
  function header() {
    const g = D.gate;
    $("#meta").innerHTML = esc(`${D.kind === "regional" ? "Regional" : "National"} model of ${nice(D.target)} · ${D.n_periods} ${pers}, ${D.first_date} to ${D.last_date} · ${K} paid channels${unitNote ? " · money" + unitNote : ""} · Papilon ${D.papilon_version}`)
      + ` <span class="badge ${g.passed ? "ok" : "no"}" title="Papilon's quality gate on the posterior">${g.passed ? "✓ " : "! "}${esc(g.verdict)}</span>`;
    document.querySelectorAll(".per").forEach((e) => { e.textContent = per; });
    document.querySelectorAll(".per-plural").forEach((e) => { e.textContent = pers; });
    document.querySelectorAll(".cur").forEach((e) => { e.textContent = D.currency; });
    $("#target-name").textContent = nice(D.target);
  }

  // ------------------------------------------------------------------ controls
  function buildControls() {
    const w = $("#window");
    D.windows.forEach((win, i) => { const o = document.createElement("option"); o.value = i; o.textContent = win.label; w.appendChild(o); });
    w.addEventListener("change", () => {
      S.win = +w.value;
      if (!S.totalEdited) S.total = startTotal(S.win);
      S.manual = null; syncControls(); schedule();
    });
    document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
      if (S.mode !== "manual" && r.value === "manual" && last) S.manual = last.spend.slice();
      S.mode = r.value; syncControls(); schedule();
    }));
    document.querySelectorAll("input[name=objective]").forEach((r) => r.addEventListener("change", () => { S.objective = r.value; schedule(); }));
    $("#total").addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (v > 0) { S.total = v; S.totalEdited = true; schedule(); } });
    const chips = $("#total-chips");
    [[-0.2, "−20%"], [-0.1, "−10%"], [0, "Starting"], [0.1, "+10%"], [0.2, "+20%"], [0.5, "+50%"]].forEach(([f, lbl]) => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = lbl;
      b.addEventListener("click", () => { S.total = startTotal(S.win) * (1 + f); S.totalEdited = f !== 0; syncControls(); schedule(); });
      chips.appendChild(b);
    });
    $("#target-roi").addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (v >= 0) { S.targetRoi = v; schedule(); } });
    $("#target-mroi").addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (v >= 0) { S.targetMroi = v; schedule(); } });
    $("#max-change").addEventListener("input", (e) => { S.maxChange = +e.target.value / 100; $("#max-change-out").textContent = "±" + e.target.value + "%"; schedule(); });
    $("#no-limit").addEventListener("change", (e) => { S.noLimit = e.target.checked; $("#max-change").disabled = S.noLimit; schedule(); });
    $("#cap").addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (v >= 1) { S.cap = v; buildSliders(); schedule(); } });
    $("#horizon").addEventListener("input", (e) => { const v = parseInt(e.target.value, 10); if (v >= 1) { S.horizon = v; render(); } });
    $("#copy-start").addEventListener("click", () => { S.manual = start().slice(); buildSliders(); schedule(); });

    const lim = $("#channel-limits");
    lim.innerHTML = `<div class="lim head"><span>Channel</span><span>Min / ${per}</span><span>Max / ${per}</span><span title="Lock at starting spend">🔒</span></div>`;
    D.media.forEach((c, j) => {
      const d = document.createElement("div"); d.className = "lim";
      d.innerHTML = `<span><i class="swatch" style="background:${col(j)}"></i>${esc(nice(c))}</span>
        <input type="number" min="0" step="any" aria-label="Minimum spend for ${esc(nice(c))}" data-k="lo" data-j="${j}">
        <input type="number" min="0" step="any" aria-label="Maximum spend for ${esc(nice(c))}" data-k="hi" data-j="${j}">
        <input type="checkbox" aria-label="Lock ${esc(nice(c))} at its starting spend" data-k="locked" data-j="${j}">`;
      lim.appendChild(d);
    });
    lim.addEventListener("input", (e) => {
      const t = e.target, j = +t.dataset.j, k = t.dataset.k;
      if (!k) return;
      if (k === "locked") S.locked[j] = t.checked;
      else { const v = parseFloat(t.value); S[k][j] = isFinite(v) && t.value !== "" ? v : null; }
      schedule();
    });
    buildSliders();
  }

  function buildSliders() {
    const box = $("#sliders"); box.innerHTML = "";
    const m = S.manual || (last ? last.spend : start());
    D.media.forEach((c, j) => {
      const max = Math.max(S.cap * D.historical_spend[j], m[j] * 1.05);
      const d = document.createElement("div"); d.className = "slider";
      d.innerHTML = `<div class="lbl"><b><i class="swatch" style="background:${col(j)}"></i>${esc(nice(c))}</b><span id="sv${j}">${money(m[j])}</span></div>
        <input type="range" min="0" max="${max}" step="${max / 400}" value="${m[j]}" aria-label="Spend per ${per} on ${esc(nice(c))}">`;
      d.querySelector("input").addEventListener("input", (e) => {
        if (!S.manual) S.manual = m.slice();
        S.manual[j] = +e.target.value; $("#sv" + j).textContent = money(S.manual[j]); schedule();
      });
      box.appendChild(d);
    });
  }

  function syncControls() {
    $("#window").value = S.win;
    document.querySelectorAll("input[name=mode]").forEach((r) => { r.checked = r.value === S.mode; });
    document.querySelectorAll("input[name=objective]").forEach((r) => { r.checked = r.value === S.objective; });
    document.querySelectorAll(".mode-opts").forEach((e) => { e.hidden = e.dataset.for !== S.mode; });
    $("#limits-group").style.opacity = S.mode === "manual" ? 0.45 : 1;
    $("#total").value = +S.total.toPrecision(6);
    $("#target-roi").value = S.targetRoi;
    $("#target-mroi").value = S.targetMroi;
    $("#max-change").value = Math.round(S.maxChange * 100);
    $("#max-change-out").textContent = "±" + Math.round(S.maxChange * 100) + "%";
    $("#max-change").disabled = S.noLimit;
    $("#no-limit").checked = S.noLimit;
    $("#cap").value = S.cap;
    $("#horizon").value = S.horizon;
    document.querySelectorAll("#channel-limits input").forEach((t) => {
      const j = +t.dataset.j, k = t.dataset.k;
      if (k === "locked") t.checked = S.locked[j]; else t.value = S[k][j] == null ? "" : S[k][j];
    });
    $("#start-total").textContent = `${money(startTotal(S.win))} a ${per} across ${K} channels`;
    if (S.mode === "manual") buildSliders();
  }

  let timer = null;
  function schedule() { clearTimeout(timer); timer = setTimeout(() => { last = compute(); render(); }, 90); }

  // ------------------------------------------------------------------ results
  function render() {
    if (!last) return;
    const r = last, ev = r.ev, H = S.horizon, st = r.evStart;
    $("#scenario-title").textContent = r.title;
    const cmp = r.mode === "fixed" ? `${D.windows[S.win].label.toLowerCase()}, scaled to the same budget` : D.windows[S.win].label.toLowerCase();
    $("#scenario-meta").textContent = `Compared with the ${cmp}. Totals cover ${H} ${pers}.` + (r.ms ? ` Solved in ${r.ms} ms.` : "");

    // notes
    const notes = r.notes.slice();
    const beyond = ev.channels.filter((c) => c.beyond_history).map((c) => nice(c.channel));
    if (beyond.length) notes.push(`${beyond.join(", ")} ${beyond.length > 1 ? "are" : "is"} planned above any spend the brand has sustained, so ${beyond.length > 1 ? "their curves are" : "its curve is"} extrapolated there. A lift test at that level would confirm it.`);
    const moved = (a) => Math.abs(r.spend[D.media.indexOf(a)] - r.start[D.media.indexOf(a)]) > 0.05 * Math.max(r.start[D.media.indexOf(a)], 1e-9);
    D.indistinguishable.forEach(([a, b]) => {
      if (moved(a) || moved(b)) notes.push(`The model cannot tell ${nice(a)} and ${nice(b)} apart at the margin, so moves between them rest on weak evidence.`);
    });
    $("#notes").innerHTML = notes.length ? `<div class="note warn"><span class="ico">!</span><div>${notes.length === 1 ? esc(notes[0]) : "<ul>" + notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>"}</div></div>` : "";

    // tiles
    const dSpend = ev.spend / st.spend - 1, dRoi = ev.roi - st.roi;
    const extra = ev.spend - sum(r.ref);
    const tiles = [
      { label: `Spend over ${H} ${pers}`, value: money(ev.spend * H), sub: `${money(ev.spend)} a ${per}`,
        delta: Math.abs(dSpend) > 1e-4 ? `${pct(dSpend)} vs start` : "Same as start", cls: "" },
      { label: `${nice(D.target)} from media over ${H} ${pers}`, value: num(ev.contribution * H),
        sub: `90% range ${num(ev.contribution_p05 * H)} to ${num(ev.contribution_p95 * H)}` },
      { label: "Return on spend (ROI)", value: ratio(ev.roi), sub: `90% range ${ratio(ev.roi_p05)} to ${ratio(ev.roi_p95)}`,
        delta: Math.abs(dRoi) > 0.005 ? `${dRoi > 0 ? "▲" : "▼"} ${Math.abs(dRoi).toFixed(2)} vs start` : "", cls: dRoi > 0 ? "up" : "down" },
      { label: r.mode === "fixed" ? "Gain over the starting mix" : "Change from the starting plan", value: signed(ev.lift * H),
        sub: `90% range ${signed(ev.lift_p05 * H)} to ${signed(ev.lift_p95 * H)}` +
             (r.mode !== "fixed" && Math.abs(extra) > 1e-6 * Math.max(1, ev.spend)
               ? (extra > 0 ? ` · ${ratio(ev.lift / extra)} back per extra ${D.currency || "unit"} spent` : ` · ${ratio(ev.lift / extra)} lost per ${D.currency || "unit"} cut`) : ""),
        delta: `${Math.round(ev.prob_better * 100)}% chance it is better`, cls: ev.prob_better >= 0.8 ? "up" : ev.prob_better < 0.5 ? "down" : "" },
    ];
    $("#tiles").innerHTML = tiles.map((t) => `<div class="tile"><div class="label">${esc(t.label)}</div><div class="value">${esc(t.value)}</div>
      <div class="sub">${esc(t.sub)}</div>${t.delta ? `<div class="delta ${t.cls || ""}">${esc(t.delta)}</div>` : ""}</div>`).join("");

    dumbbell(); planTable(); curves(); savedTable();
  }

  function dumbbell() {
    const r = last, el = $("#dumbbell"), W = width(el, 800), rowH = 34, top = 6, left = 130, right = 70;
    const H = top + rowH * K + 26;
    el.innerHTML = "";
    const svg = S_("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Starting and scenario spend by channel" }, el);
    const xmax = Math.max(...r.spend, ...r.start) * 1.08 || 1;
    const x = (v) => left + (W - left - right) * v / xmax;
    const g = S_("g", { class: "grid" }, svg);
    ticks(xmax, 5).filter((t) => t <= xmax).forEach((t) => {
      S_("line", { x1: x(t), x2: x(t), y1: top, y2: top + rowH * K }, g);
      S_("text", { x: x(t), y: top + rowH * K + 16, "text-anchor": "middle", text: money(t) }, svg);
    });
    D.media.forEach((c, j) => {
      const y = top + rowH * j + rowH / 2, a = r.start[j], b = r.spend[j], ch = a > 0 ? b / a - 1 : NaN;
      S_("text", { x: left - 12, y: y + 4, "text-anchor": "end", class: "lab", text: nice(c) }, svg);
      S_("line", { x1: x(a), x2: x(b), y1: y, y2: y, style: `stroke:${col(j)};stroke-width:3;stroke-linecap:round;opacity:.45` }, svg);
      S_("circle", { cx: x(a), cy: y, r: 5.5, style: `fill:var(--surface-1);stroke:${col(j)};stroke-width:2` }, svg);
      S_("circle", { cx: x(b), cy: y, r: 7, style: `fill:${col(j)};stroke:var(--surface-1);stroke-width:2` }, svg);
      const lx = b >= a ? x(b) + 12 : x(b) - 12;
      S_("text", { x: lx, y: y + 4, "text-anchor": b >= a ? "start" : "end", class: "val", text: money(b) }, svg);
      S_("text", { x: W - 4, y: y + 4, "text-anchor": "end", class: "val", style: "font-weight:600", text: isFinite(ch) ? pct(ch) : "new" }, svg);
      const hit = S_("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, style: "fill:transparent" }, svg);
      const cc = r.ev.channels[j];
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${esc(nice(c))}</b>${row("Start", money(a))}${row("Scenario", money(b))}${row("Change", isFinite(ch) ? pct(ch) : "new")}${row("Contribution", num(cc.contribution))}${row("ROI", ratio(cc.roi))}${row("Marginal ROI", ratio(cc.mroi))}`));
      hit.addEventListener("mouseleave", hideTip);
    });
  }

  function planTable() {
    const r = last, H = S.horizon, t = $("#plan-table");
    const lockedNow = S.mode !== "manual" ? S.locked : Array(K).fill(false);
    let h = `<thead><tr><th>Channel</th><th>Start / ${per}</th><th>Scenario / ${per}</th><th>Change</th><th>Over ${H} ${pers}</th><th>Contribution / ${per}</th><th>ROI</th><th>Marginal ROI</th></tr></thead><tbody>`;
    r.ev.channels.forEach((c, j) => {
      const ch = r.start[j] > 0 ? c.spend / r.start[j] - 1 : NaN;
      h += `<tr><td><i class="swatch" style="background:${col(j)}"></i>${esc(nice(c.channel))}${c.beyond_history ? '<span class="flag warn">beyond history</span>' : ""}${lockedNow[j] ? '<span class="flag lock">locked</span>' : ""}</td>
        <td>${money(r.start[j])}</td><td>${money(c.spend)}</td><td>${isFinite(ch) ? pct(ch) : "new"}</td><td>${money(c.spend * H)}</td>
        <td>${num(c.contribution)} <span class="hint">(${num(c.contribution_p05)} to ${num(c.contribution_p95)})</span></td><td>${ratio(c.roi)}</td><td>${ratio(c.mroi)}</td></tr>`;
    });
    const ev = r.ev, ts = sum(r.start);
    h += `<tr class="total"><td>Total</td><td>${money(ts)}</td><td>${money(ev.spend)}</td><td>${pct(ev.spend / ts - 1)}</td><td>${money(ev.spend * H)}</td><td>${num(ev.contribution)}</td><td>${ratio(ev.roi)}</td><td></td></tr></tbody>`;
    t.innerHTML = h;
  }

  const curveCache = new Map();
  function curveFor(j, xmax) {
    const key = j + ":" + xmax.toPrecision(6);
    if (!curveCache.has(key)) curveCache.set(key, E.curve(M, j, xmax, 48));
    return curveCache.get(key);
  }

  function curves() {
    const r = last, box = $("#curves");
    box.innerHTML = "";
    D.media.forEach((c, j) => {
      const xmax = Math.max(S.cap * D.historical_spend[j], D.support_high[j] * 1.15, r.spend[j] * 1.1, r.start[j] * 1.1);
      const pts = curveFor(j, xmax);
      const ymax = Math.max(...pts.map((p) => p.p95)) * 1.05 || 1;
      const cc = r.ev.channels[j];
      const card = document.createElement("div"); card.className = "curve";
      card.innerHTML = `<h4><i class="swatch" style="background:${col(j)}"></i>${esc(nice(c))}</h4>
        <div class="cmeta">Scenario ${money(r.spend[j])} → ${num(cc.contribution)} a ${per} · marginal ROI ${ratio(cc.mroi)}</div>`;
      const W = 320, Hh = 180, L = 44, R = 10, T = 8, B = 24;
      const svg = S_("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": `Response curve for ${nice(c)}` }, card);
      const x = (v) => L + (W - L - R) * v / xmax, y = (v) => T + (Hh - T - B) * (1 - v / ymax);
      const pid = "hatch" + j;
      const defs = S_("defs", {}, svg);
      const pat = S_("pattern", { id: pid, patternUnits: "userSpaceOnUse", width: 5, height: 5, patternTransform: "rotate(45)" }, defs);
      S_("line", { x1: 0, y1: 0, x2: 0, y2: 5, style: "stroke:var(--hatch);stroke-width:2" }, pat);
      const g = S_("g", { class: "grid" }, svg);
      ticks(ymax, 3).filter((t) => t <= ymax).forEach((t) => {
        S_("line", { x1: L, x2: W - R, y1: y(t), y2: y(t) }, g);
        S_("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end", text: num(t) }, svg);
      });
      ticks(xmax, 3).filter((t) => t <= xmax).forEach((t) => S_("text", { x: x(t), y: Hh - 6, "text-anchor": "middle", text: money(t) }, svg));
      const sh = D.support_high[j];
      if (sh < xmax) {
        S_("rect", { x: x(sh), y: T, width: x(xmax) - x(sh), height: Hh - T - B, style: `fill:url(#${pid});opacity:.55` }, svg);
        S_("line", { x1: x(sh), x2: x(sh), y1: T, y2: Hh - B, style: "stroke:var(--hatch);stroke-width:1" }, svg);
      }
      const band = pts.map((p) => `${x(p.x)},${y(p.p95)}`).concat(pts.slice().reverse().map((p) => `${x(p.x)},${y(p.p05)}`));
      S_("polygon", { points: band.join(" "), style: `fill:${col(j)};opacity:.18` }, svg);
      S_("polyline", { points: pts.map((p) => `${x(p.x)},${y(p.mean)}`).join(" "), style: `fill:none;stroke:${col(j)};stroke-width:2` }, svg);
      S_("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), style: "stroke:var(--line)" }, svg);
      const ys = M.mean(j, r.start[j]), yp = M.mean(j, r.spend[j]);
      S_("circle", { cx: x(r.start[j]), cy: y(ys), r: 5, style: `fill:var(--surface-1);stroke:${col(j)};stroke-width:2` }, svg);
      S_("circle", { cx: x(r.spend[j]), cy: y(yp), r: 6, style: `fill:${col(j)};stroke:var(--surface-1);stroke-width:2` }, svg);
      const cross = S_("line", { y1: T, y2: Hh - B, style: "stroke:var(--text-muted);stroke-dasharray:3 3;opacity:0" }, svg);
      const hit = S_("rect", { x: L, y: T, width: W - L - R, height: Hh - T - B, style: "fill:transparent" }, svg);
      hit.addEventListener("mousemove", (e) => {
        const b = svg.getBoundingClientRect(), sx = (e.clientX - b.left) * W / b.width;
        const v = Math.max(0, Math.min(xmax, (sx - L) / (W - L - R) * xmax));
        const i = Math.round(v / xmax * (pts.length - 1)), p = pts[i];
        const q = pts[Math.min(i + 1, pts.length - 1)], o = pts[Math.max(i - 1, 0)];
        const mr = (q.mean - o.mean) / Math.max(q.x - o.x, 1e-12);
        cross.setAttribute("x1", x(p.x)); cross.setAttribute("x2", x(p.x)); cross.style.opacity = 1;
        showTip(e, `<b>${esc(nice(c))} at ${money(p.x)} a ${per}</b>${row("Contribution", num(p.mean))}${row("90% range", num(p.p05) + " to " + num(p.p95))}${row("ROI", ratio(p.x > 0 ? p.mean / p.x : NaN))}${row("Marginal ROI", ratio(mr))}${p.x > sh ? row("Note", "beyond sustained history") : ""}`);
      });
      hit.addEventListener("mouseleave", () => { cross.style.opacity = 0; hideTip(); });
      box.appendChild(card);
    });
  }

  // ------------------------------------------------------------------ saved scenarios
  function snapshot() {
    const { saved, ...rest } = S;
    return JSON.parse(JSON.stringify(rest));
  }
  function savedTable() {
    const card = $("#saved-card");
    card.hidden = !S.saved.length;
    if (!S.saved.length) return;
    const cols = S.saved;
    const H = S.horizon;
    const line = (label, f) => `<tr><td>${label}</td>${cols.map((s) => `<td>${f(s)}</td>`).join("")}</tr>`;
    let h = `<thead><tr><th></th>${cols.map((s, i) => `<th>${esc(s.name)} <button class="x" data-load="${i}" title="Open this scenario">↺</button><button class="x" data-del="${i}" title="Remove">×</button></th>`).join("")}</tr></thead><tbody>`;
    h += line("Goal", (s) => esc(s.title));
    h += line(`Spend / ${per}`, (s) => money(s.ev.spend));
    h += line(`Spend over ${H} ${pers}`, (s) => money(s.ev.spend * H));
    h += line(`${esc(nice(D.target))} from media / ${per}`, (s) => `${num(s.ev.contribution)} <span class="hint">(${num(s.ev.contribution_p05)} to ${num(s.ev.contribution_p95)})</span>`);
    h += line("ROI", (s) => ratio(s.ev.roi));
    h += line(`Gain vs its comparison / ${per}`, (s) => signed(s.ev.lift));
    h += line("Chance it is better", (s) => Math.round(s.ev.prob_better * 100) + "%");
    D.media.forEach((c, j) => { h += line(`<i class="swatch" style="background:${col(j)}"></i>${esc(nice(c))} / ${per}`, (s) => money(s.spend[j])); });
    $("#saved-table").innerHTML = h + "</tbody>";
  }
  $("#saved-table").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.del != null) { S.saved.splice(+b.dataset.del, 1); savedTable(); }
    if (b.dataset.load != null) { Object.assign(S, JSON.parse(JSON.stringify(S.saved[+b.dataset.load].state))); syncControls(); schedule(); }
  });

  function save() {
    if (!last) return;
    const name = `Scenario ${S.saved.length + 1}`;
    S.saved.push({ name, state: snapshot(), title: last.title, spend: last.spend.slice(), ev: last.ev });
    savedTable(); msg(`Saved as “${name}”.`);
  }

  // ------------------------------------------------------------------ share and export
  function encodeState() {
    const o = { s: snapshot(), v: S.saved.map((x) => ({ n: x.name, s: x.state })) };
    return btoa(unescape(encodeURIComponent(JSON.stringify(o))));
  }
  function restore() {
    const m = /[#&]s=([^&]+)/.exec(location.hash);
    if (!m) return;
    try {
      const o = JSON.parse(decodeURIComponent(escape(atob(decodeURIComponent(m[1])))));
      Object.assign(S, o.s);
      if (S.win >= D.windows.length) S.win = 0;
      S.saved = (o.v || []).map((x) => {
        const keep = JSON.parse(JSON.stringify(S));
        Object.assign(S, x.s); const r = compute(); Object.assign(S, keep);
        return { name: x.n, state: x.s, title: r.title, spend: r.spend, ev: r.ev };
      });
    } catch (err) { /* a damaged link opens the default scenario */ }
  }
  function share() {
    const url = location.href.split("#")[0] + "#s=" + encodeURIComponent(encodeState());
    history.replaceState(null, "", url);
    const done = () => msg("Link copied. It reopens this scenario and your saved ones in this file.");
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done, () => msg("The address bar now holds a link to this scenario."));
    else msg("The address bar now holds a link to this scenario.");
  }
  function csv() {
    const rows = [["scenario", "goal", "channel", `start_per_${per}`, `spend_per_${per}`, `spend_over_${S.horizon}_${pers}`,
      `contribution_per_${per}`, "contribution_p05", "contribution_p95", "roi", "marginal_roi", "beyond_history"]];
    const add = (name, r) => r.ev.channels.forEach((c, j) => rows.push([name, r.title, c.channel, r.start ? r.start[j] : "", c.spend, c.spend * S.horizon,
      c.contribution, c.contribution_p05, c.contribution_p95, c.roi, c.mroi, c.beyond_history]));
    add("current", last);
    S.saved.forEach((s) => add(s.name, { ev: s.ev, title: s.title, start: null }));
    const text = rows.map((r) => r.map((v) => { const s = typeof v === "number" ? (+v.toPrecision(8)).toString() : String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = "papilon-scenarios.csv"; document.body.appendChild(a); a.click(); a.remove();
    msg("Downloaded papilon-scenarios.csv.");
  }
  function msg(t) { $("#action-msg").textContent = t; }

  // ------------------------------------------------------------------ model results
  function modelTab() {
    const g = D.gate, tiles = [];
    if (D.fit) {
      tiles.push({ label: "Variation explained (R²)", value: (D.fit.r2 * 100).toFixed(1) + "%", sub: "In-sample fit" });
      tiles.push({ label: "Mean absolute % error", value: (D.fit.mape * 100).toFixed(1) + "%", sub: `wMAPE ${(D.fit.wmape * 100).toFixed(1)}%` });
    }
    tiles.push({ label: "Quality gate", value: g.verdict, sub: `${g.method} sampling · effective sample ${(g.ess * 100).toFixed(0)}%` });
    tiles.push({ label: "Model family", value: g.family.length > 22 ? g.family.split(":")[0] : g.family, sub: "Chosen by Bayes factor" });
    $("#model-tiles").innerHTML = tiles.map((t) => `<div class="tile"><div class="label">${esc(t.label)}</div><div class="value" style="font-size:22px">${esc(t.value)}</div><div class="sub">${esc(t.sub)}</div></div>`).join("");
    $("#cautions").innerHTML = D.cautions.length ? `<div class="note warn"><span class="ico">!</span><div><b>Cautions Papilon raises about this model</b><ul>${D.cautions.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div></div>` : "";
    $("#fit-card").hidden = !D.fit; $("#decomp-card").hidden = !D.decomposition;
    roiTable();
  }

  function fitChart() {
    if (!D.fit) return;
    const el = $("#fit"), W = width(el, 900), Hh = 260, L = 52, R = 12, T = 10, B = 26;
    el.innerHTML = "";
    const svg = S_("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Actual and fitted outcome over time" }, el);
    const a = D.fit.actual, f = D.fit.fitted, n = a.length;
    const lo = Math.min(...a, ...f), hi = Math.max(...a, ...f), pad = (hi - lo) * 0.08;
    const y0 = Math.max(0, lo - pad), y1 = hi + pad;
    const x = (i) => L + (W - L - R) * i / (n - 1), y = (v) => T + (Hh - T - B) * (1 - (v - y0) / (y1 - y0));
    const tk = ticks(y1 - y0, 4).map((t) => y0 + t).filter((t) => t <= y1);
    const g = S_("g", { class: "grid" }, svg);
    tk.forEach((t) => { S_("line", { x1: L, x2: W - R, y1: y(t), y2: y(t) }, g); S_("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end", text: num(t) }, svg); });
    const years = [];
    D.fit.dates.forEach((d, i) => { const yr = d.slice(0, 4); if (!years.length || years[years.length - 1][0] !== yr) years.push([yr, i]); });
    years.forEach(([yr, i]) => S_("text", { x: x(i), y: Hh - 6, "text-anchor": "start", text: yr }, svg));
    S_("polyline", { points: a.map((v, i) => `${x(i)},${y(v)}`).join(" "), style: "fill:none;stroke:var(--text-muted);stroke-width:1.5" }, svg);
    S_("polyline", { points: f.map((v, i) => `${x(i)},${y(v)}`).join(" "), style: "fill:none;stroke:var(--accent);stroke-width:2" }, svg);
    const cross = S_("line", { y1: T, y2: Hh - B, style: "stroke:var(--text-muted);stroke-dasharray:3 3;opacity:0" }, svg);
    const dot = S_("circle", { r: 4, style: "fill:var(--accent);stroke:var(--surface-1);stroke-width:2;opacity:0" }, svg);
    const hit = S_("rect", { x: L, y: T, width: W - L - R, height: Hh - T - B, style: "fill:transparent" }, svg);
    hit.addEventListener("mousemove", (e) => {
      const b = svg.getBoundingClientRect(), sx = (e.clientX - b.left) * W / b.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((sx - L) / (W - L - R) * (n - 1))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.style.opacity = 1;
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(f[i])); dot.style.opacity = 1;
      showTip(e, `<b>${esc(D.fit.dates[i])}</b>${row("Actual", num(a[i]))}${row("Fitted", num(f[i]))}${row("Error", pct(a[i] ? (f[i] - a[i]) / a[i] : NaN))}`);
    });
    hit.addEventListener("mouseleave", () => { cross.style.opacity = 0; dot.style.opacity = 0; hideTip(); });
  }

  function decompChart() {
    if (!D.decomposition) return;
    const el = $("#decomp"), W = width(el, 900), parts = D.decomposition.slice().sort((p, q) => Math.abs(q.total) - Math.abs(p.total));
    const tot = sum(D.decomposition.map((p) => p.total)), rowH = 28, L = 150, R = 70, T = 4;
    const Hh = T + rowH * parts.length + 6;
    el.innerHTML = "";
    const svg = S_("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Share of the outcome by driver" }, el);
    const maxShare = Math.max(...parts.map((p) => Math.abs(p.total) / tot));
    const x = (v) => (W - L - R) * v / maxShare;
    parts.forEach((p, i) => {
      const y = T + rowH * i, share = p.total / tot, j = D.media.indexOf(p.name);
      const fill = j >= 0 ? col(j) : p.kind === "baseline" ? "var(--text-muted)" : "var(--series-other)";
      S_("text", { x: L - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "lab", text: p.kind === "baseline" ? "baseline (trend, season)" : nice(p.name) + (p.kind === "control" ? " (control)" : p.kind === "organic" ? " (organic)" : "") }, svg);
      const w = Math.max(1, x(Math.abs(share)));
      S_("rect", { x: L, y: y + 6, width: w, height: rowH - 12, rx: 3, style: `fill:${fill};opacity:${p.kind === "control" ? 0.6 : 1}` }, svg);
      S_("text", { x: L + w + 8, y: y + rowH / 2 + 4, class: "val", text: (share < -0.0005 ? "−" : "") + (Math.abs(share) * 100).toFixed(1) + "%" }, svg);
      const hit = S_("rect", { x: 0, y, width: W, height: rowH, style: "fill:transparent" }, svg);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${esc(nice(p.name))}</b>${row("Total", num(p.total))}${row("Share", (share * 100).toFixed(1) + "%")}`));
      hit.addEventListener("mouseleave", hideTip);
    });
  }

  function forestChart() {
    const el = $("#forest"), W = width(el, 900), rowH = 34, L = 130, R = 20, T = 4, B = 26;
    const Hh = T + rowH * K + B;
    el.innerHTML = "";
    const svg = S_("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "ROI with 90% ranges and marginal ROI by channel" }, el);
    const xmax = Math.max(...D.roi.map((r) => Math.max(r.roas_p95, r.mroas_mean))) * 1.08 || 1;
    const x = (v) => L + (W - L - R) * Math.max(0, v) / xmax;
    const g = S_("g", { class: "grid" }, svg);
    ticks(xmax, 5).filter((t) => t <= xmax).forEach((t) => { S_("line", { x1: x(t), x2: x(t), y1: T, y2: T + rowH * K }, g); S_("text", { x: x(t), y: Hh - 6, "text-anchor": "middle", text: t.toFixed(t < 1 && t > 0 ? 2 : 1) }, svg); });
    if (xmax > 1) {
      S_("line", { x1: x(1), x2: x(1), y1: T, y2: T + rowH * K, style: "stroke:var(--text-muted);stroke-dasharray:4 3" }, svg);
      S_("text", { x: x(1) + 4, y: T + 10, text: "1.0" }, svg);
    }
    D.roi.forEach((r, j) => {
      const y = T + rowH * j + rowH / 2;
      S_("text", { x: L - 12, y: y + 4, "text-anchor": "end", class: "lab", text: nice(r.channel) }, svg);
      S_("line", { x1: x(r.roas_p05), x2: x(r.roas_p95), y1: y, y2: y, style: `stroke:${col(j)};stroke-width:3;stroke-linecap:round;opacity:.55` }, svg);
      S_("circle", { cx: x(r.roas_mean), cy: y, r: 6.5, style: `fill:${col(j)};stroke:var(--surface-1);stroke-width:2` }, svg);
      const m = x(r.mroas_mean);
      S_("rect", { x: m - 4.5, y: y - 4.5, width: 9, height: 9, transform: `rotate(45 ${m} ${y})`, style: `fill:var(--surface-1);stroke:var(--text-secondary);stroke-width:2` }, svg);
      const hit = S_("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, style: "fill:transparent" }, svg);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${esc(nice(r.channel))}</b>${row("ROI", ratio(r.roas_mean))}${row("90% range", ratio(r.roas_p05) + " to " + ratio(r.roas_p95))}${row("Marginal ROI", ratio(r.mroas_mean))}${row("Share of " + esc(nice(D.target)), (r.share_of_sales * 100).toFixed(1) + "%")}`));
      hit.addEventListener("mouseleave", hideTip);
    });
  }

  function roiTable() {
    let h = `<thead><tr><th>Channel</th><th>Spend, whole history</th><th>Spend / ${per}</th><th>ROI</th><th>90% range</th><th>Marginal ROI</th><th>Share of ${esc(nice(D.target))}</th></tr></thead><tbody>`;
    D.roi.forEach((r, j) => {
      h += `<tr><td><i class="swatch" style="background:${col(j)}"></i>${esc(nice(r.channel))}</td><td>${money(r.spend)}</td><td>${money(D.historical_spend[j])}</td><td>${ratio(r.roas_mean)}</td><td>${ratio(r.roas_p05)} to ${ratio(r.roas_p95)}</td><td>${ratio(r.mroas_mean)}</td><td>${(r.share_of_sales * 100).toFixed(1)}%</td></tr>`;
    });
    $("#roi-table").innerHTML = h + "</tbody>";
  }

  function about() {
    const rec = D.recommended;
    $("#about").innerHTML = `
      <h3>What this page is</h3>
      <p>A planner for the marketing mix model Papilon fitted to ${esc(nice(D.target))} over ${D.n_periods} ${pers} (${esc(D.first_date)} to ${esc(D.last_date)}). It carries ${D.n_draws} draws from the model's posterior, so every number comes with a range, and it plans budgets inside the page. No data leaves your browser.</p>
      <h3>What the numbers mean</h3>
      <ul>
        <li><b>${esc(nice(D.target))} from media</b> is the part of ${esc(nice(D.target))} the paid channels drive when spend holds steady at the planned level each ${per}. It is incremental: it is not a forecast of total ${esc(nice(D.target))}, which also depends on baseline demand, seasonality, prices and promotions.</li>
        <li><b>ROI</b> is that contribution divided by spend. <b>Marginal ROI</b> is what the next unit of spend would return, the slope of the response curve.</li>
        <li><b>Gain over the starting mix</b> compares the scenario with the starting period's mix, scaled to the same budget for a fixed budget. The chance it is better is the share of posterior draws in which the scenario wins.</li>
        <li><b>Beyond history</b> marks spend above the 95th percentile of what the brand has sustained over its carryover window. The curve there is an extrapolation.</li>
      </ul>
      <h3>How plans are found</h3>
      <ul>
        <li><b>Fixed budget, protect the downside</b> (Papilon's default) maximizes the 10th percentile of the gain over the starting mix. Across 120 simulated brands with known returns, this lost money far less often than maximizing the expected outcome, which leans on the channels a model overrates.</li>
        <li><b>Fixed budget, maximize the expected outcome</b> splits the budget exactly on a grid (dynamic programming, so S-shaped curves are handled), then refines it.</li>
        <li><b>Grow while ROI holds</b> finds the largest total whose best split keeps total ROI at the target. Total ROI falls as spend grows, so a higher target means a smaller budget.</li>
        <li><b>Spend to a marginal return</b> raises each channel until the next unit of spend returns the target. A target of 1 means spending until the last unit just pays for itself, when ${esc(nice(D.target))} is measured in money.</li>
        <li>Channel limits default to ±50% of each channel's starting spend and at most 3× its historical average. Large moves rest on parts of the curve the data has seen least.</li>
      </ul>
      <p>For reference, Papilon's own optimizer, run in Python on all posterior draws, recommends a plan at the historical budget with an expected gain of ${signed(rec.expected_lift)} a ${per} (90% range ${signed(rec.lift_p05)} to ${signed(rec.lift_p95)}, ${Math.round(rec.prob_better * 100)}% chance it is better). The planner's fixed-budget mode with the default settings reproduces it.</p>
      <h3>Limits</h3>
      <ul>
        <li>Results are steady state. They ignore the weeks it takes carryover to build up after a change, and they assume the next periods look like the fitted ones.</li>
        <li>The model is only as good as its identification. Where Papilon flags a channel as uncertain or two channels as indistinguishable, a lift test is the way to settle it.</li>
      </ul>
      <h3>Make one for your own model</h3>
      <p><code>python -m papilon.planner --data sales.csv --target sales --date week --media search,social,video -o planner.html</code>, or from Python, <code>papilon.scenario_planner(model, "planner.html")</code>.</p>`;
  }

  // ------------------------------------------------------------------ tabs and wiring
  function tabs() {
    const list = [...document.querySelectorAll("[role=tab]")];
    const show = (t) => {
      list.forEach((b) => { const on = b === t; b.setAttribute("aria-selected", on); document.getElementById(b.getAttribute("aria-controls")).hidden = !on; });
      if (t.id === "tab-model") { fitChart(); decompChart(); forestChart(); }
      if (t.id === "tab-plan") render();
    };
    list.forEach((b, i) => {
      b.addEventListener("click", () => show(b));
      b.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = list[(i + (e.key === "ArrowRight" ? 1 : list.length - 1)) % list.length]; n.focus(); show(n); }
      });
    });
  }

  let rz = null;
  window.addEventListener("resize", () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (!$("#panel-plan").hidden) dumbbell(); if (!$("#panel-model").hidden) { fitChart(); decompChart(); forestChart(); } }, 150);
  });

  $("#save").addEventListener("click", save);
  $("#share").addEventListener("click", share);
  $("#csv").addEventListener("click", csv);

  header(); buildControls(); restore(); syncControls(); modelTab(); about(); tabs();
  last = compute(); render();
  window.PapilonPlanner = { state: S, compute: () => (last = compute(), render(), last) };
})();
