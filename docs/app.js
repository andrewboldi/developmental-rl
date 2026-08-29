/* Growing Up to Learn — scene + chart engine.
   One fixed canvas, per-stage scissor rendering, on-demand frames only.
   three r147 UMD + fat lines; GSAP 3.15 ScrollTrigger. Data: window.DATA. */
(function () {
  'use strict';
  gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);
  const D = window.DATA;
  const PAL = { blue: 0x4c86d8, orange: 0xc77b2f, green: 0x4f9e6e, violet: 0xaa64c8, gold: 0xad8f2e, red: 0xc75f6b, accent: 0xe8c468, ink: 0xf2ede4 };
  const CSS = { blue: '#4C86D8', orange: '#C77B2F', green: '#4F9E6E', violet: '#AA64C8', gold: '#AD8F2E', red: '#C75F6B', accent: '#E8C468', muted: '#9FA8B8', hair: '#263049', ink: '#F2EDE4' };
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- renderer core ---------------- */
  const canvas = document.getElementById('gl');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const lineMats = [];   // LineMaterial registry (resolution updates)
  const stages = [];     // {el, scene, camera, update(t), extraRender?}
  let renderQueued = false;
  function requestRender() {
    if (renderQueued) return; renderQueued = true;
    requestAnimationFrame(() => { renderQueued = false; drawFrame(); });
  }
  function drawFrame() {
    const W = window.innerWidth, H = window.innerHeight;
    renderer.setScissorTest(false); renderer.setClearColor(0x000000, 0); renderer.clear();
    renderer.setScissorTest(true);
    for (const st of stages) {
      const r = st.el.getBoundingClientRect();
      if (r.bottom < 0 || r.top > H || r.width === 0) continue;
      const y = H - r.bottom;
      if (st.extraRender) { st.extraRender(r, y); continue; }
      renderer.setViewport(r.left, y, r.width, r.height);
      renderer.setScissor(r.left, y, r.width, r.height);
      st.camera.aspect = r.width / r.height;
      if (st.camera.isOrthographicCamera) fitOrtho(st.camera, r.width / r.height);
      st.camera.updateProjectionMatrix();
      renderer.render(st.scene, st.camera);
    }
  }
  function fitOrtho(cam, aspect) {
    let h = cam.userData.viewH || 12;
    const w = cam.userData.viewW || 0;
    if (w && w / aspect > h) h = w / aspect;
    cam.top = h / 2; cam.bottom = -h / 2; cam.left = -h * aspect / 2; cam.right = h * aspect / 2;
  }
  function onResize() {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    for (const m of lineMats) m.resolution.set(window.innerWidth, window.innerHeight);
    requestRender();
  }
  window.addEventListener('resize', onResize);

  /* ---------------- three helpers ---------------- */
  function dimetricCam(viewH) {
    const c = new THREE.OrthographicCamera(-1, 1, 1, -1, -100, 200);
    c.userData.viewH = viewH;
    c.position.set(14, 14 * Math.tan(35.264 * Math.PI / 180) * 1.41, 14);
    c.lookAt(0, 0, 0); return c;
  }
  function lights(scene, amb = 0.55, key = 0.75) {
    scene.add(new THREE.AmbientLight(0xbfd0ff, amb));
    const d = new THREE.DirectionalLight(0xfff2dc, key); d.position.set(6, 10, 4); scene.add(d);
    return d;
  }
  function mat(color, opt = {}) { return new THREE.MeshLambertMaterial(Object.assign({ color }, opt)); }
  function fatLine(pts, color, width, dashed = false) {
    const g = new THREE.LineGeometry();
    g.setPositions(pts.flat());
    const m = new THREE.LineMaterial({ color, linewidth: width, transparent: true, dashed });
    m.resolution.set(window.innerWidth, window.innerHeight);
    if (dashed) { m.defines.USE_DASH = ''; }
    lineMats.push(m);
    const line = new THREE.Line2(g, m); line.computeLineDistances();
    return line;
  }
  function gridToWorld(r, c, H, W, y = 0) { return [c - (W - 1) / 2, y, r - (H - 1) / 2]; }

  /* ---------------- SVG chart engine (dataviz spec) ---------------- */
  const NS = 'http://www.w3.org/2000/svg';
  const tooltip = document.getElementById('tooltip');
  function el(n, attrs, parent) {
    const e = document.createElementNS(NS, n);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e); return e;
  }
  function niceTicks(min, max, n = 4) {
    const span = max - min || 1, step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step0; const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const t = []; for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) t.push(v);
    return t;
  }
  function fmt(v) { return Math.abs(v) >= 1000 ? (v / 1000) + 'k' : String(Math.round(v * 100) / 100); }
  function lineChart(svgId, cfg) {
    const svg = document.getElementById(svgId); if (!svg) return;
    const W = 800, Hh = cfg.height || 320, m = { l: 52, r: 110, t: 14, b: 34 };
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
    const iw = W - m.l - m.r, ih = Hh - m.t - m.b;
    const xs = cfg.series[0].x;
    const xmin = Math.min(...cfg.series.map(s => s.x[0])), xmax = Math.max(...cfg.series.map(s => s.x[s.x.length - 1]));
    let ymin = cfg.ymin != null ? cfg.ymin : Infinity, ymax = cfg.ymax != null ? cfg.ymax : -Infinity;
    if (cfg.ymin == null || cfg.ymax == null) for (const s of cfg.series) {
      for (const v of (s.lo || s.y)) ymin = Math.min(ymin, v);
      for (const v of (s.hi || s.y)) ymax = Math.max(ymax, v);
    }
    if (cfg.ymin == null) ymin = Math.min(ymin, 0);
    const X = v => m.l + (v - xmin) / (xmax - xmin) * iw;
    const Y = v => m.t + ih - (v - ymin) / (ymax - ymin) * ih;
    // grid + axes
    for (const t of niceTicks(ymin, ymax)) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: CSS.hair, 'stroke-width': 1 }, svg);
      el('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, svg).textContent = fmt(t);
    }
    for (const t of niceTicks(xmin, xmax, 5)) {
      el('text', { x: X(t), y: Hh - 10, 'text-anchor': 'middle' }, svg).textContent = fmt(t);
    }
    el('text', { x: m.l + iw / 2, y: Hh - (cfg.xlabelPad || -4) + 4, 'text-anchor': 'middle', fill: CSS.muted }, svg);
    if (cfg.threshold != null) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(cfg.threshold), y2: Y(cfg.threshold), stroke: CSS.accent, 'stroke-width': 1, 'stroke-dasharray': '5 5', opacity: .65 }, svg);
      el('text', { x: m.l + 8, y: Y(cfg.threshold) - 6, fill: CSS.accent }, svg).textContent = cfg.thresholdLabel || '';
    }
    if (cfg.bands) for (const b of cfg.bands) {
      el('rect', { x: X(b.x0), y: m.t, width: X(b.x1) - X(b.x0), height: ih, fill: b.color || '#4C86D8', opacity: .07 }, svg);
      el('text', { x: X(b.x0) + 5, y: m.t + 13, fill: CSS.muted, 'font-size': 10 }, svg).textContent = b.label || '';
    }
    const drawFns = [];
    const usedLabelY = [];
    cfg.series.forEach(s => {
      if (s.lo) {
        let dPath = '';
        s.x.forEach((x, i) => dPath += (i ? 'L' : 'M') + X(x) + ' ' + Y(s.lo[i]));
        for (let i = s.x.length - 1; i >= 0; i--) dPath += 'L' + X(s.x[i]) + ' ' + Y(s.hi[i]);
        el('path', { d: dPath + 'Z', fill: s.color, opacity: .13 }, svg);
      }
      let d = '';
      s.x.forEach((x, i) => d += (i ? 'L' : 'M') + X(x) + ' ' + Y(s.y[i]));
      const p = el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
      const len = p.getTotalLength();
      p.style.strokeDasharray = len; p.style.strokeDashoffset = reduceMotion ? 0 : len;
      drawFns.push(t => { p.style.strokeDashoffset = len * (1 - t); });
      // direct label at line end, collision-nudged
      let ly = Y(s.y[s.y.length - 1]);
      while (usedLabelY.some(u => Math.abs(u - ly) < 14)) ly += 14;
      usedLabelY.push(ly);
      el('text', { x: W - m.r + 8, y: ly + 4, fill: s.color, class: 'dl' }, svg).textContent = s.name;
    });
    // hover crosshair + tooltip
    const hitRect = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent', 'pointer-events': 'all' }, svg);
    const cross = el('line', { y1: m.t, y2: m.t + ih, stroke: CSS.muted, 'stroke-width': 1, opacity: 0 }, svg);
    hitRect.addEventListener('mousemove', ev => {
      const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      const xv = xmin + (loc.x - m.l) / iw * (xmax - xmin);
      let idx = 0, best = Infinity;
      xs.forEach((x, i) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; idx = i; } });
      cross.setAttribute('x1', X(xs[idx])); cross.setAttribute('x2', X(xs[idx])); cross.setAttribute('opacity', .5);
      tooltip.innerHTML = `<div class="trow"><span>${cfg.xname || 'x'}</span><b>${fmt(xs[idx])}</b></div>` +
        cfg.series.map(s => `<div class="trow"><span style="color:${s.color}">${s.name}</span><b>${s.y[idx] != null ? (+s.y[idx]).toFixed(cfg.dp != null ? cfg.dp : 2) : '—'}</b></div>`).join('');
      tooltip.style.opacity = 1;
      tooltip.style.left = Math.min(window.innerWidth - 190, ev.clientX + 14) + 'px';
      tooltip.style.top = (ev.clientY + 14) + 'px';
    });
    hitRect.addEventListener('mouseleave', () => { tooltip.style.opacity = 0; cross.setAttribute('opacity', 0); });
    // legend
    if (cfg.legend) {
      const lg = document.getElementById(cfg.legend);
      if (lg) lg.innerHTML = cfg.series.map(s => `<span><span class="sw" style="background:${s.color}"></span>${s.name}</span>`).join('');
    }
    return { draw: t => drawFns.forEach(f => f(t)) };
  }
  function groupedBars(svg, cfg, x0, width) {
    // draws grouped bars into an existing svg at x offset; returns nothing fancy
    const Hh = cfg.height, m = cfg.m, ih = Hh - m.t - m.b;
    const groups = cfg.groups, series = cfg.series;
    const gw = width / groups.length;
    const bw = Math.min(26, (gw - 18) / series.length);
    const Y = v => m.t + ih - v / (cfg.ymax || 1) * ih;
    for (const t of [0, .25, .5, .75, 1]) {
      el('line', { x1: x0, x2: x0 + width, y1: Y(t), y2: Y(t), stroke: CSS.hair }, cfg.svg);
      el('text', { x: x0 - 7, y: Y(t) + 4, 'text-anchor': 'end' }, cfg.svg).textContent = t;
    }
    groups.forEach((g, gi) => {
      series.forEach((s, si) => {
        const v = s.values[gi];
        const x = x0 + gi * gw + gw / 2 + (si - series.length / 2) * (bw + 2) + 1;
        el('rect', { x, y: Y(v), width: bw, height: m.t + ih - Y(v), fill: s.color, rx: 3 }, cfg.svg);
        el('text', { x: x + bw / 2, y: Y(v) - 5, 'text-anchor': 'middle', fill: s.color, 'font-size': 10.5 }, cfg.svg).textContent = v.toFixed(2).replace('0.', '.');
      });
      el('text', { x: x0 + gi * gw + gw / 2, y: Hh - 10, 'text-anchor': 'middle' }, cfg.svg).textContent = g;
    });
  }

  /* ---------------- charts from DATA ---------------- */
  const chartDraws = {};
  chartDraws.exp1 = lineChart('chart-exp1', {
    xname: 'steps', legend: 'lg-exp1', threshold: 0.9, thresholdLabel: '90%', ymin: 0, ymax: 1, dp: 2,
    series: [
      { name: 'Dyna-Q (model)', color: CSS.blue, x: D.exp1.curves.checkpoints, y: D.exp1.curves.dyna.iqm, lo: D.exp1.curves.dyna.ci_lo, hi: D.exp1.curves.dyna.ci_hi },
      { name: 'Q + replay (no model)', color: CSS.green, x: D.exp1.curves.checkpoints, y: D.exp1.curves.replay.iqm, lo: D.exp1.curves.replay.ci_lo, hi: D.exp1.curves.replay.ci_hi },
      { name: 'Q-learning', color: CSS.orange, x: D.exp1.curves.checkpoints, y: D.exp1.curves.q.iqm, lo: D.exp1.curves.q.ci_lo, hi: D.exp1.curves.q.ci_hi },
    ],
  });
  chartDraws.exp2 = lineChart('chart-exp2', {
    xname: 'steps', legend: 'lg-exp2', threshold: D.exp2.curves.threshold, thresholdLabel: '90%', ymin: 0, ymax: 1, dp: 2,
    bands: [{ x0: 0, x1: 12000, label: 'shoot drill' }, { x0: 12000, x1: 24000, label: 'dribble drill' }],
    series: [
      { name: 'drills, varied', color: CSS.blue, x: D.exp2.curves.checkpoints, y: D.exp2.curves.conditions['drills-varied'].iqm, lo: D.exp2.curves.conditions['drills-varied'].ci_lo, hi: D.exp2.curves.conditions['drills-varied'].ci_hi },
      { name: 'drills, fixed', color: CSS.violet, x: D.exp2.curves.checkpoints, y: D.exp2.curves.conditions['drills-fixed'].iqm, lo: D.exp2.curves.conditions['drills-fixed'].ci_lo, hi: D.exp2.curves.conditions['drills-fixed'].ci_hi },
      { name: 'whole games', color: CSS.orange, x: D.exp2.curves.checkpoints, y: D.exp2.curves.conditions.whole.iqm, lo: D.exp2.curves.conditions.whole.ci_lo, hi: D.exp2.curves.conditions.whole.ci_hi },
      { name: 'whole, optimistic', color: CSS.green, x: D.exp2.curves.checkpoints, y: D.exp2.curves.conditions['whole-optimistic'].iqm },
      { name: 'exploring starts', color: CSS.gold, x: D.exp2.curves.checkpoints, y: D.exp2.curves.conditions['explore-starts'].iqm },
    ],
  });
  // exp3: acquisition lines (left) + retention/transfer bars (right) in one svg
  (function exp3chart() {
    const svg = document.getElementById('chart-exp3'); if (!svg) return;
    const W = 800, Hh = 320; svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
    const m = { t: 16, b: 34 };
    // left: acquisition
    const ac = D.exp3.eval_curves && D.exp3.eval_curves.acquisition;
    const ep = D.exp3.eval_curves ? D.exp3.eval_curves.episodes : D.exp3.practice.episodes;
    const lw = 400, lx = 50, ih = Hh - m.t - m.b;
    const bl = ac ? ac.blocked.iqm : D.exp3.practice.blocked, il = ac ? ac.interleaved.iqm : D.exp3.practice.interleaved;
    const X = i => lx + (ep[i] - ep[0]) / (ep[ep.length - 1] - ep[0]) * lw;
    const Y = v => m.t + ih - v * ih;
    [0, .25, .5, .75, 1].forEach(t => {
      el('line', { x1: lx, x2: lx + lw, y1: Y(t), y2: Y(t), stroke: CSS.hair }, svg);
      el('text', { x: lx - 8, y: Y(t) + 4, 'text-anchor': 'end' }, svg).textContent = t;
    });
    (D.exp3.phase_boundaries || []).forEach(b => {
      const i = ep.findIndex(e => e >= b); if (i > 0) el('line', { x1: X(i), x2: X(i), y1: m.t, y2: m.t + ih, stroke: CSS.hair, 'stroke-dasharray': '3 4' }, svg);
    });
    [['blocked', bl, CSS.orange], ['interleaved', il, CSS.blue]].forEach(([nm, ys, col]) => {
      let d = ''; ys.forEach((v, i) => d += (i ? 'L' : 'M') + X(i) + ' ' + Y(v));
      el('path', { d, fill: 'none', stroke: col, 'stroke-width': 2 }, svg);
      el('text', { x: X(ys.length - 1) - 4, y: Y(ys[ys.length - 1]) - 8, fill: col, 'text-anchor': 'end', class: 'dl' }, svg).textContent = nm;
    });
    el('text', { x: lx + lw / 2, y: Hh - 8, 'text-anchor': 'middle' }, svg).textContent = 'practice episodes (score on currently-practiced passage)';
    // right: bars — retention mean + transfer
    const gx = 520, gwid = 240;
    groupedBars(null, {
      svg, height: Hh, m: { t: m.t, b: m.b }, ymax: 1,
      groups: ['retention', 'transfer'],
      series: [
        { name: 'blocked', color: CSS.orange, values: [D.exp3.retention.blocked.iqm[3], D.exp3.transfer.blocked.iqm != null ? D.exp3.transfer.blocked.iqm : D.exp3.transfer.blocked] },
        { name: 'interleaved', color: CSS.blue, values: [D.exp3.retention.interleaved.iqm[3], D.exp3.transfer.interleaved.iqm != null ? D.exp3.transfer.interleaved.iqm : D.exp3.transfer.interleaved] },
      ],
    }, gx, gwid);
    el('text', { x: gx + gwid / 2, y: Hh - 8, 'text-anchor': 'middle' }, svg).textContent = 'test, learning off';
    const lg = document.getElementById('lg-exp3');
    if (lg) lg.innerHTML = `<span><span class="sw" style="background:${CSS.orange}"></span>blocked</span><span><span class="sw" style="background:${CSS.blue}"></span>interleaved</span>`;
  })();
  chartDraws.exp4 = lineChart('chart-exp4', {
    xname: 'gen', legend: 'lg-exp4', ymin: 0, ymax: 10, dp: 2, xlabelPad: 0,
    series: [
      { name: 'distill', color: CSS.blue, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['generational-distill'].iqm, lo: D.exp4.greedy['generational-distill'].ci_lo, hi: D.exp4.greedy['generational-distill'].ci_hi },
      { name: 'optimistic init', color: CSS.red, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['optimistic-init'].iqm },
      { name: 'weight-copy', color: CSS.orange, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['weight-copy'].iqm },
      { name: 'random advice', color: CSS.violet, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['random-advice'].iqm },
      { name: 'one long life', color: CSS.green, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['one-long-life'].iqm },
      { name: 'no inherit', color: CSS.gold, x: [1, 2, 3, 4, 5], y: D.exp4.greedy['no-inheritance'].iqm },
    ],
  });
  chartDraws.exp5 = lineChart('chart-exp5', {
    xname: 'steps', legend: 'lg-exp5', ymin: 0, dp: 0,
    series: [
      { name: 'adult, walk', color: CSS.red, x: D.exp5.damage['adult-walk'].steps, y: D.exp5.damage['adult-walk'].iqm, lo: D.exp5.damage['adult-walk'].lo, hi: D.exp5.damage['adult-walk'].hi },
      { name: 'adult, balance-first', color: CSS.orange, x: D.exp5.damage['adult-balance-first'].steps, y: D.exp5.damage['adult-balance-first'].iqm },
      { name: 'grow linear, walk', color: CSS.violet, x: D.exp5.damage['grow-linear-walk'].steps, y: D.exp5.damage['grow-linear-walk'].iqm, lo: D.exp5.damage['grow-linear-walk'].lo, hi: D.exp5.damage['grow-linear-walk'].hi },
      { name: 'grow jump', color: CSS.green, x: D.exp5.damage['grow-jump'].steps, y: D.exp5.damage['grow-jump'].iqm },
      { name: 'grow adaptive, walk', color: CSS.blue, x: D.exp5.damage['grow-adaptive-walk'].steps, y: D.exp5.damage['grow-adaptive-walk'].iqm, lo: D.exp5.damage['grow-adaptive-walk'].lo, hi: D.exp5.damage['grow-adaptive-walk'].hi },
    ],
  });
  // scorecard
  (function scorecard() {
    const sc = document.getElementById('scorecard'); if (!sc) return;
    const rows = [
      ['H1', 'world models', '4 of 6', 'blind-at-home 97.5% ≈ sighted; stranger collapses to 29%. Refuted honestly: the Dyna speedup (update-count artifact) and pure dead reckoning.'],
      ['H2', 'microtasks', '2 of 3', 'drills 15.1k vs 40.4k steps (p≤4e-9); structure beats mere diversity (p=1e-14). Boundary: optimism skips the need.'],
      ['H3', 'variation', '3 of 4', 'retention p=5e-10 and transfer p=1e-5 replicate counterbalanced; mechanism PROVEN by ablation. Acquisition edge: boundary.'],
      ['H4', 'teachers', '2 of 4', 'distill 10.0 vs every dose-matched control (p≤2e-20); random advice poisons. Boundary: global optimism solves small worlds.'],
      ['H5', 'growing bodies', '2 of 4', '42× less damage AND a third faster, robust to physics arms. Reversed: gradualism and balance-first.'],
    ];
    sc.innerHTML = rows.map(([h, nm, score, txt]) =>
      `<div class="tile win"><div class="v">${h} <span class="chip ok">${score}</span></div><div class="l"><b>${nm}</b> — ${txt}</div></div>`).join('');
  })();

  /* ---------------- scenes ---------------- */
  function addStage(name, build) {
    const elx = document.querySelector(`[data-stage="${name}"]`) || document.querySelector(`[data-scene="${name}"]`);
    if (!elx) return null;
    const scene = new THREE.Scene();
    const st = build(scene, elx);
    st.el = elx; st.scene = scene; stages.push(st); return st;
  }

  // ---- hero: five floating islands ----
  const hero = addStage('hero', (scene, elx) => {
    const cam = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    lights(scene, .5, .8);
    const islands = new THREE.Group(); scene.add(islands);
    const defs = [
      { c: PAL.blue, build: g => { // home: walls
          for (let i = 0; i < 4; i++) { const w = new THREE.Mesh(new THREE.BoxGeometry(.28, .5, 1.6), mat(0x3a4c6e)); w.position.set(i < 2 ? (i ? .8 : -.8) : 0, .35, i >= 2 ? (i === 2 ? .8 : -.8) : 0); if (i >= 2) w.rotation.y = Math.PI / 2; g.add(w); }
          const fr = new THREE.Mesh(new THREE.BoxGeometry(.3, .42, .3), mat(PAL.accent, { emissive: 0x6b5620 })); fr.position.set(.45, .3, -.4); g.add(fr); } },
      { c: PAL.green, build: g => { const p = new THREE.Mesh(new THREE.BoxGeometry(1.9, .1, 1.2), mat(0x2c5c40)); p.position.y = .12; g.add(p);
          const goal = new THREE.Mesh(new THREE.BoxGeometry(.08, .3, .7), mat(0xf2ede4)); goal.position.set(.92, .28, 0); g.add(goal); } },
      { c: PAL.violet, build: g => { for (let k = 0; k < 7; k++) { const key = new THREE.Mesh(new THREE.BoxGeometry(.22, .1, .9), mat(k % 2 ? 0x222b40 : 0xd8d3c6)); key.position.set(-.7 + k * .24, .18, 0); g.add(key); } } },
      { c: PAL.gold, build: g => { for (let k = 0; k < 3; k++) { const gem = new THREE.Mesh(new THREE.OctahedronGeometry(.22), mat(PAL.accent, { emissive: 0x4a3c14 })); gem.position.set(0, .3 + k * .5, 0); gem.scale.setScalar(1 - k * .22); g.add(gem); } } },
      { c: PAL.red, build: g => { const pole = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, 1), mat(0xd8d3c6)); pole.position.y = .62; pole.rotation.z = .3; g.add(pole);
          const bob = new THREE.Mesh(new THREE.SphereGeometry(.2), mat(PAL.red)); bob.position.set(-.3, 1.1, 0); g.add(bob); } },
    ];
    defs.forEach((d, i) => {
      const g = new THREE.Group();
      const base = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.35, .35, 6), mat(0x1a2336));
      base.position.y = -.05; g.add(base);
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(1.16, 1.16, .06, 6), mat(d.c));
      rim.position.y = .14; g.add(rim);
      d.build(g);
      const ang = (i / 5) * Math.PI * 2 + .5;
      g.position.set(Math.cos(ang) * 6.6, Math.sin(i * 2.1) * 1.1 - .4, Math.sin(ang) * 5.2 - i * 2.6);
      g.rotation.y = -ang + .6;
      islands.add(g);
    });
    cam.position.set(0, 2.6, 15);
    const st = { camera: cam, update(t) {
      cam.position.z = 13 - t * 10; cam.position.y = 2.4 - t * 1.2;
      islands.rotation.y = t * .55;
      cam.lookAt(0, .4, -t * 6);
    } };
    st.update(0); return st;
  });

  // ---- primer: 3x3 grid with growing Q arrows ----
  addStage('primer', scene => {
    const cam = dimetricCam(7.2); cam.userData.viewW = 9; lights(scene);
    const g = new THREE.Group(); scene.add(g);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const tile = new THREE.Mesh(new THREE.BoxGeometry(.94, .18, .94), mat(r === 0 && c === 2 ? 0x4a3c14 : 0x1d2740));
      tile.position.set(c - 1, 0, r - 1); g.add(tile);
    }
    const goal = new THREE.Mesh(new THREE.BoxGeometry(.5, .3, .5), mat(PAL.accent, { emissive: 0x6b5620 }));
    goal.position.set(1, .28, -1); g.add(goal);
    const agent = new THREE.Mesh(new THREE.SphereGeometry(.26, 24, 18), mat(PAL.blue, { emissive: 0x14273f }));
    agent.position.set(-1, .35, 1); g.add(agent);
    // arrows: cone per (cell, dir) pointing toward better value
    const coneGeo = new THREE.ConeGeometry(.09, .3, 10);
    const arrows = new THREE.InstancedMesh(coneGeo, mat(0x7fa8dd), 36);
    arrows.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(arrows);
    const dum = new THREE.Object3D();
    const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // toward -z(up-grid), +x, +z, -x
    function setArrows(t) {
      let i = 0;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        const dGoal = Math.abs(r - 0) + Math.abs(c - 2);
        for (let d = 0; d < 4; d++) {
          const nr = r + dirs[d][1], nc = c + dirs[d][0];
          let v = 0;
          if (nr >= 0 && nr < 3 && nc >= 0 && nc < 3) {
            const nd = Math.abs(nr - 0) + Math.abs(nc - 2);
            v = Math.pow(.75, nd) * (nd < dGoal ? 1 : .25);
          }
          const reveal = Math.max(0, Math.min(1, (t * 6 - dGoal)));
          const s = v * reveal;
          dum.position.set(c - 1 + dirs[d][0] * .28, .35, r - 1 + dirs[d][1] * .28);
          dum.rotation.set(0, 0, 0);
          if (d === 0) dum.rotation.x = -Math.PI / 2 * 0, dum.rotation.x = 0;
          // orient cone along dir
          dum.rotation.y = Math.atan2(dirs[d][0], dirs[d][1]);
          dum.rotation.x = Math.PI / 2;
          dum.scale.setScalar(Math.max(.001, s));
          dum.updateMatrix(); arrows.setMatrixAt(i++, dum.matrix);
        }
      }
      arrows.instanceMatrix.needsUpdate = true;
      // agent strolls a canned loop late in the scrub
      const path = [[1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1]];
      const wt = Math.max(0, t - .55) / .45;
      const fi = wt * (path.length - 1);
      const i0 = Math.floor(fi), i1 = Math.min(path.length - 1, i0 + 1), fr = fi - i0;
      agent.position.x = path[i0][1] * (1 - fr) + path[i1][1] * fr;
      agent.position.z = path[i0][0] * (1 - fr) + path[i1][0] * fr;
    }
    const st = { camera: cam, update: setArrows };
    st.update(reduceMotion ? 1 : 0); return st;
  });

  // ---- H1: blindfold home ----
  const H1 = addStage('h1', scene => {
    const cam = dimetricCam(13); cam.userData.viewW = 19.5; lights(scene, .32, .5);
    function home(hdef, tint) {
      const rows = hdef.ascii || hdef;
      const Hn = rows.length, Wn = rows[0].length;
      const grp = new THREE.Group();
      const floor = new THREE.Mesh(new THREE.BoxGeometry(Wn, .3, Hn), mat(0x151d2f));
      floor.position.y = -.16; grp.add(floor);
      let n = 0; rows.forEach(row => { for (const ch of row) if (ch === '#') n++; });
      const walls = new THREE.InstancedMesh(new THREE.BoxGeometry(.96, .8, .96), mat(tint), n);
      const dum = new THREE.Object3D(); let i = 0;
      let S = hdef.start || [1, 1], G = hdef.goal || [2, 7];
      rows.forEach((row, r) => { [...row].forEach((ch, c) => {
        if (ch === '#') { const [x, , z] = gridToWorld(r, c, Hn, Wn); dum.position.set(x, .4, z); dum.updateMatrix(); walls.setMatrixAt(i++, dum.matrix); }
        if (ch === 'S') S = [r, c]; if (ch === 'G') G = [r, c];
      }); });
      grp.add(walls);
      const fridge = new THREE.Mesh(new THREE.BoxGeometry(.7, .9, .7), mat(PAL.accent, { emissive: 0x5c4a1c }));
      const [gx, , gz] = gridToWorld(G[0], G[1], Hn, Wn); fridge.position.set(gx, .45, gz);
      grp.add(fridge);
      return { grp, Hn, Wn, S, G };
    }
    const A = home(D.exp1.homes.A, 0x2d3c5c), B = home(D.exp1.homes.B, 0x51344a);
    scene.add(A.grp); scene.add(B.grp); B.grp.visible = false;
    const agent = new THREE.Mesh(new THREE.SphereGeometry(.34, 24, 18), mat(0xe8d9b8, { emissive: 0x33291a }));
    agent.position.y = .45; scene.add(agent);
    const ghost = new THREE.Mesh(new THREE.SphereGeometry(.34, 24, 18),
      new THREE.MeshLambertMaterial({ color: PAL.blue, transparent: true, opacity: .55, emissive: 0x1c3050 }));
    ghost.position.y = .45; scene.add(ghost); ghost.visible = false;
    const spot = new THREE.SpotLight(0xfff0c8, 2.2, 9, .55, .5); spot.position.set(0, 4, 0);
    spot.target = agent; scene.add(spot);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(1.5, 3.6, 24, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xffe9b0, transparent: true, opacity: .10, depthWrite: false }));
    cone.position.y = 2.2; scene.add(cone);
    const tether = fatLine([[0, .45, 0], [0, .45, .001]], PAL.red, 2.2); tether.visible = false; scene.add(tether);
    // BFS sighted path in A (illustrating the greedy policy's walk)
    function bfsPath(hm) {
      const raw = hm === A ? D.exp1.homes.A : D.exp1.homes.B;
      const rows = raw.ascii || raw;
      const key = (r, c) => r * 100 + c; const prev = new Map(); const q = [hm.S]; const seen = new Set([key(...hm.S)]);
      while (q.length) { const [r, c] = q.shift();
        if (r === hm.G[0] && c === hm.G[1]) break;
        for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) { const nr = r + dr, nc = c + dc;
          if (rows[nr][nc] !== '#' && !seen.has(key(nr, nc))) { seen.add(key(nr, nc)); prev.set(key(nr, nc), [r, c]); q.push([nr, nc]); } } }
      const path = [[...hm.G]]; let cur = hm.G;
      while (key(...cur) !== key(...hm.S)) { cur = prev.get(key(...cur)); path.unshift(cur); }
      return path;
    }
    const sightedPath = bfsPath(A);
    function place(mesh, hm, rc, frac, rc2) {
      const [x0, , z0] = gridToWorld(rc[0], rc[1], hm.Hn, hm.Wn);
      let x = x0, z = z0;
      if (rc2) { const [x1, , z1] = gridToWorld(rc2[0], rc2[1], hm.Hn, hm.Wn); x = x0 + (x1 - x0) * frac; z = z0 + (z1 - z0) * frac; }
      mesh.position.x = x; mesh.position.z = z;
    }
    function playTraj(hm, traj, t, withGhost) {
      const path = traj.path; const fi = t * (path.length - 1);
      const i0 = Math.floor(fi), i1 = Math.min(path.length - 1, i0 + 1), fr = fi - i0;
      place(agent, hm, path[i0].true, fr, path[i1].true);
      if (withGhost) { ghost.visible = true; place(ghost, hm, path[i0].bel, fr, path[i1].bel);
        const H = path[i0].H || 0; const s = 1 + Math.min(1.6, H * .45); ghost.scale.setScalar(s);
        ghost.material.opacity = Math.max(.2, .6 - H * .1);
        // divergence tether
        const p1 = agent.position, p2 = ghost.position;
        const d = p1.distanceTo(p2);
        tether.visible = d > .8;
        if (tether.visible) { tether.geometry.dispose(); const g2 = new THREE.LineGeometry(); g2.setPositions([p1.x, .45, p1.z, p2.x, .45, p2.z]); tether.geometry = g2; tether.computeLineDistances(); }
      } else { ghost.visible = false; tether.visible = false; }
      spot.position.set(agent.position.x, 4, agent.position.z);
      cone.position.set(agent.position.x, 2.2, agent.position.z);
    }
    const st = { camera: cam, update(t) {
      // beats: 0-.25 sighted A · .25-.5 blind-A-touch · .5-.62 swap to B · .62-.92 blind-B-touch · .92-1 hold
      if (t < .25) { A.grp.visible = true; B.grp.visible = false; spot.intensity = 2.2; cone.visible = true;
        const u = t / .25; const fi = u * (sightedPath.length - 1); const i0 = Math.floor(fi), i1 = Math.min(sightedPath.length - 1, i0 + 1);
        place(agent, A, sightedPath[i0], fi - i0, sightedPath[i1]); ghost.visible = false; tether.visible = false;
      } else if (t < .5) { A.grp.visible = true; B.grp.visible = false; spot.intensity = .25; cone.visible = false;
        playTraj(A, D.exp1.trajs['blind-A-touch'], (t - .25) / .25, true);
      } else if (t < .62) { const u = (t - .5) / .12; A.grp.visible = u < .5; B.grp.visible = u >= .5; spot.intensity = .25; cone.visible = false; ghost.visible = false; tether.visible = false;
        agent.position.set(-6 + u * 12 * 0, .45, 0);
      } else { A.grp.visible = false; B.grp.visible = true; spot.intensity = .25; cone.visible = false;
        playTraj(B, D.exp1.trajs['blind-B-touch'], Math.min(1, (t - .62) / .3), true);
      }
    } };
    st.update(reduceMotion ? .4 : 0); return st;
  });

  // ---- H2: pitch with ghost trails ----
  addStage('h2', scene => {
    const cam = dimetricCam(10); cam.userData.viewW = 15.5; lights(scene, .5, .75);
    const P = D.exp2.pitch, Wn = P.W, Hn = P.H;
    const slab = new THREE.Mesh(new THREE.BoxGeometry(Wn + .6, .3, Hn + .6), mat(0x17332a));
    slab.position.y = -.16; scene.add(slab);
    // pitch lines
    const lineM = new THREE.MeshBasicMaterial({ color: 0xd8d3c6, transparent: true, opacity: .22 });
    const mid = new THREE.Mesh(new THREE.PlaneGeometry(.08, Hn), lineM);
    mid.rotation.x = -Math.PI / 2; mid.position.y = .02; scene.add(mid);
    (P.goal_cells || [[2, 10], [3, 10], [4, 10]]).forEach(rc => {
      const fr = new THREE.Mesh(new THREE.BoxGeometry(.9, .34, .9), new THREE.MeshLambertMaterial({ color: PAL.accent, transparent: true, opacity: .5, emissive: 0x51431a }));
      const [x, , z] = gridToWorld(rc[0], rc[1], Hn, Wn); fr.position.set(x, .17, z); scene.add(fr);
    });
    const ball = new THREE.Mesh(new THREE.SphereGeometry(.2, 18, 14), mat(0xf2ede4));
    ball.position.y = .25; scene.add(ball);
    const agent = new THREE.Mesh(new THREE.SphereGeometry(.3, 24, 18), mat(PAL.blue, { emissive: 0x12233c })); agent.position.y = .4; scene.add(agent);
    const conds = [['whole', PAL.orange], ['drills-fixed', PAL.violet], ['drills-varied', PAL.blue]];
    const trails = new THREE.Group(); scene.add(trails);
    const trailObjs = [];
    conds.forEach(([cond, col]) => {
      ['25', '50', '100'].forEach((stg, si) => {
        const t = D.exp2.trajs[cond] && D.exp2.trajs[cond][stg]; if (!t || !t.steps.length) return;
        const pts = t.steps.map(s => { const [x, , z] = gridToWorld(s.agent[0], s.agent[1], Hn, Wn); return [x, .12 + si * .04, z]; });
        if (pts.length < 2) return;
        const ln = fatLine(pts, col, 2.4); ln.material.transparent = true; ln.material.opacity = 0;
        trails.add(ln);
        trailObjs.push({ cond, stg: si, line: ln, scored: t.scored, steps: t.steps });
      });
    });
    const st = { camera: cam, update(t) {
      // three beats, one per condition; within each, stages 25→50→100 reveal
      const ci = Math.min(2, Math.floor(t * 3)); const u = (t * 3) - ci;
      trailObjs.forEach(o => {
        const active = conds[ci][0] === o.cond;
        const sReveal = Math.max(0, Math.min(1, u * 3 - o.stg));
        o.line.material.opacity = active ? .75 * sReveal : .10;
      });
      const [cond] = conds[ci];
      const stg = u < .34 ? '25' : u < .67 ? '50' : '100';
      const tr = D.exp2.trajs[cond] && D.exp2.trajs[cond][stg];
      if (tr && tr.steps.length) {
        const wt = (u % .34) / .34; const fi = wt * (tr.steps.length - 1);
        const i0 = Math.floor(fi), i1 = Math.min(tr.steps.length - 1, i0 + 1), fr = fi - i0;
        const s0 = tr.steps[i0], s1 = tr.steps[i1];
        const [ax0, , az0] = gridToWorld(s0.agent[0], s0.agent[1], Hn, Wn);
        const [ax1, , az1] = gridToWorld(s1.agent[0], s1.agent[1], Hn, Wn);
        agent.position.x = ax0 + (ax1 - ax0) * fr; agent.position.z = az0 + (az1 - az0) * fr;
        const b0 = s0.ball, b1 = s1.ball;
        if (b0 && b1) { const [bx0, , bz0] = gridToWorld(b0[0], b0[1], Hn, Wn); const [bx1, , bz1] = gridToWorld(b1[0], b1[1], Hn, Wn);
          ball.visible = true; ball.position.x = bx0 + (bx1 - bx0) * fr; ball.position.z = bz0 + (bz1 - bz0) * fr;
        } else { ball.visible = false; }
      }
    } };
    st.update(reduceMotion ? .95 : 0); return st;
  });

  // ---- H3: piano roll wall ----
  addStage('h3', scene => {
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, -50, 100);
    cam.userData.viewH = 13; cam.userData.viewW = 15; cam.position.set(0, 0, 20); cam.lookAt(0, 0, 0);
    lights(scene, .8, .35);
    const S = D.exp3.structure; const nK = D.exp3.n_keys || 8, len = 12;
    function lane(xoff, label) {
      const grp = new THREE.Group(); grp.position.x = xoff; scene.add(grp);
      const cells = new THREE.InstancedMesh(new THREE.BoxGeometry(.42, .42, .18), new THREE.MeshLambertMaterial(), len * nK);
      const dum = new THREE.Object3D(); const colors = [];
      let i = 0;
      for (let p = 0; p < len; p++) for (let k = 0; k < nK; k++) {
        dum.position.set((p - len / 2 + .5) * .5, (k - nK / 2 + .5) * .5, 0);
        dum.updateMatrix(); cells.setMatrixAt(i, dum.matrix);
        cells.setColorAt(i, new THREE.Color(0x1c2438)); i++;
      }
      cells.instanceColor.needsUpdate = true;
      grp.add(cells);
      return { grp, cells };
    }
    const L1 = lane(-3.6), L2 = lane(3.6);
    const blackRow = new Set([1, 3, 6]);
    function paintLane(L, passageIdx, playT, col, errAt) {
      const keys = S.correct_keys[passageIdx];
      let i = 0; const c = new THREE.Color(); const laneCol = new THREE.Color(col);
      const played = playT * len;
      for (let p = 0; p < len; p++) for (let k = 0; k < nK; k++) {
        const correct = keys && keys[p] === k;
        c.set(blackRow.has(k) ? 0x141b2c : 0x1c2438);           // key-bed rows
        if (correct) c.set(0x3a4a6c);                            // the fingering, faintly visible
        if (Math.abs(played - p) < .5) c.lerp(new THREE.Color(0x8892a8), .35); // playhead column
        if (correct && p < played) {
          const glow = Math.max(0, 1 - (played - p) * .28);
          c.copy(laneCol).lerp(new THREE.Color(0xffffff), .25 * glow);
          if (glow < .3) c.lerp(new THREE.Color(0x1c2438), .55 - glow); // trail fades
        }
        if (errAt != null && p === errAt && Math.abs(played - p) < 1 && k === ((keys[p] + 3) % nK)) c.set(PAL.red);
        L.cells.setColorAt(i++, c);
      }
      L.cells.instanceColor.needsUpdate = true;
    }
    const st = { camera: cam, update(t) {
      // beats: 0-.5 practice (blocked plays A repeatedly; interleaved cycles A/B/C) · .5-1 test on novel-ish (passage index 3 if present else 0)
      if (t < .5) {
        const u = (t / .5) * 3; const rep = Math.floor(u), pt = u - rep;
        paintLane(L1, 0, pt, PAL.orange, null);                    // blocked: drill A
        paintLane(L2, rep % 3, pt, PAL.blue, null);                // interleaved: rotate
      } else {
        const u = (t - .5) / .5;
        const novel = S.passages.length > 3 ? 3 : 0;
        const exc = (S.exceptions && S.exceptions[0] != null) ? (Array.isArray(S.exceptions[0]) ? S.exceptions[0][0] : S.exceptions[0]) : 6;
        paintLane(L1, novel, u, PAL.orange, typeof exc === 'number' ? exc : 6); // blocked stumbles
        paintLane(L2, novel, u, PAL.blue, null);                                // interleaved flows
      }
    } };
    st.update(reduceMotion ? .8 : 0); return st;
  });

  // ---- H4: trapgrid generations ----
  addStage('h4', scene => {
    const cam = dimetricCam(13.5); cam.userData.viewW = 21.5; lights(scene, .4, .6);
    const Gd = D.exp4.grid; const rows = (Gd.map.trim ? Gd.map.trim().split('\n') : Gd.map);
    const Hn = Gd.height || rows.length, Wn = Gd.width || rows[0].length;
    const floor = new THREE.Mesh(new THREE.BoxGeometry(Wn, .3, Hn), mat(0x141c2e)); floor.position.y = -.16; scene.add(floor);
    let nW = 0; rows.forEach(r => { for (const ch of r) if (ch === '#') nW++; });
    if (nW) {
      const walls = new THREE.InstancedMesh(new THREE.BoxGeometry(.96, .7, .96), mat(0x27324e), nW);
      const dum = new THREE.Object3D(); let i = 0;
      rows.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === '#') { const [x, , z] = gridToWorld(r, c, Hn, Wn); dum.position.set(x, .35, z); dum.updateMatrix(); walls.setMatrixAt(i++, dum.matrix); } }));
      scene.add(walls);
    }
    (Gd.candies || []).forEach(rc => {
      const cd = new THREE.Mesh(new THREE.SphereGeometry(.26, 16, 12), mat(0xd678a8, { emissive: 0x54263e }));
      const [x, , z] = gridToWorld(rc[0], rc[1], Hn, Wn); cd.position.set(x, .3, z); scene.add(cd);
    });
    const goalRC = Gd.goal || [5, 13];
    const gem = new THREE.Mesh(new THREE.OctahedronGeometry(.7), mat(PAL.accent, { emissive: 0x6b5620 }));
    { const [x, , z] = gridToWorld(goalRC[0], goalRC[1], Hn, Wn); gem.position.set(x, .6, z); }
    scene.add(gem);
    // advice skeleton per generation
    const genLines = [];
    (D.exp4.advice || []).forEach((adv, gi) => {
      const pts = adv.pairs.map(p => { const [x, , z] = gridToWorld(p.rc[0], p.rc[1], Hn, Wn); return [x, .16 + gi * .05, z]; });
      if (pts.length < 2) return;
      const col = new THREE.Color(PAL.blue).lerp(new THREE.Color(PAL.accent), gi / 4);
      const ln = fatLine(pts, col.getHex(), 1.6 + gi * .9);
      ln.material.transparent = true; ln.material.opacity = 0; scene.add(ln);
      genLines.push(ln);
    });
    const finalPts = (D.exp4.final_paths['generational-distill'] || []).map(rc => { const [x, , z] = gridToWorld(rc[0], rc[1], Hn, Wn); return [x, .5, z]; });
    const golden = finalPts.length > 1 ? fatLine(finalPts.map(pt => [pt[0], .55, pt[2]]), PAL.accent, 6) : null;
    if (golden) { golden.material.transparent = true; golden.material.opacity = 0; scene.add(golden); }
    const st = { camera: cam, update(t) {
      const g = t * 6; // 5 gens + golden finale
      genLines.forEach((ln, i) => {
        const u = Math.max(0, Math.min(1, g - i));
        ln.material.opacity = u * (.10 + .22 * (i + 1) / 5);
      });
      if (golden) golden.material.opacity = Math.max(0, Math.min(1, g - 5));
      gem.rotation.y = t * 3;
      gem.scale.setScalar(1 + Math.max(0, Math.min(1, g - 5)) * .25);
    } };
    st.update(reduceMotion ? 1 : 0); return st;
  });

  // ---- H5: scissor-split growing pendulums ----
  addStage('h5', (scene, elx) => {
    // two scenes rendered side by side inside one stage
    function pendulumScene(cond, col) {
      const sc = new THREE.Scene(); lights(sc, .5, .8);
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, -50, 100);
      cam.userData.viewH = 4.6; cam.position.set(0, .95, 10); cam.lookAt(0, .85, 0);
      const ground = new THREE.Mesh(new THREE.BoxGeometry(6, .16, 2), mat(0x1a2336)); sc.add(ground);
      const cart = new THREE.Mesh(new THREE.BoxGeometry(.7, .3, .5), mat(0x33415f)); cart.position.y = .25; sc.add(cart);
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(.05, .07, 1), mat(0xd8d3c6));
      const bob = new THREE.Mesh(new THREE.SphereGeometry(.22, 20, 16), mat(col));
      const pivot = new THREE.Group(); pivot.position.y = .4;
      pole.position.y = .5; pivot.add(pole); bob.position.y = 1; pivot.add(bob);
      sc.add(pivot);
      const flash = new THREE.Mesh(new THREE.PlaneGeometry(6, 3), new THREE.MeshBasicMaterial({ color: PAL.red, transparent: true, opacity: 0, depthWrite: false }));
      flash.position.set(0, 1.4, -1); sc.add(flash);
      return { sc, cam, pivot, pole, bob, flash, cond,
        theta: D.exp5.theta[cond] || [], sizes: D.exp5.sizes[cond], falls: D.exp5.falls[cond] || [] };
    }
    const left = pendulumScene('adult-walk', PAL.red);
    const right = pendulumScene('grow-adaptive-walk', PAL.blue);
    // HTML damage counters
    const hud = document.createElement('div');
    hud.className = 'hud';
    hud.innerHTML = `
      <div style="position:absolute;top:64px;left:0;right:50%;text-align:center;font-family:var(--mono);font-size:12px;color:${CSS.muted}">born adult<br><span id="dmgL" style="font-size:26px;color:${CSS.red}">0</span><br>damage</div>
      <div style="position:absolute;top:64px;left:50%;right:0;text-align:center;font-family:var(--mono);font-size:12px;color:${CSS.muted}">grows when ready<br><span id="dmgR" style="font-size:26px;color:${CSS.blue}">0</span><br>damage</div>
      <div style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:${CSS.hair}"></div>`;
    elx.appendChild(hud);
    const dmgL = hud.querySelector('#dmgL'), dmgR = hud.querySelector('#dmgR');
    const budget = 120000;
    function cumDamage(falls, step) { let s = 0; for (const f of falls) { if (f.step <= step) s += f.dmg; } return s; }
    function updateSide(side, t) {
      const trainStep = t * budget;
      // size from example schedule
      const ss = side.sizes; let size = 1;
      if (ss && ss.example && ss.example.length) {
        const idx = Math.min(ss.example.length - 1, Math.floor(t * (ss.example.length - 1)));
        size = ss.example[idx];
      } else if (ss && ss.iqm) {
        const idx = Math.min(ss.iqm.length - 1, Math.floor(t * (ss.iqm.length - 1)));
        size = ss.iqm[idx];
      }
      side.pivot.scale.setScalar(size);
      side.bob.scale.setScalar(.6 + Math.pow(size, 3) * .6);
      // theta animation: loop the 200-step adult trace, speed by scrub
      const th = side.theta;
      if (th.length) {
        const i = Math.floor(t * 4 * (th.length - 1)) % th.length;
        side.pivot.rotation.z = -th[i] * 2.2;
      }
      // fall flash near a fall event
      const nearFall = side.falls.some(f => Math.abs(f.step - trainStep) < budget * .004);
      side.flash.material.opacity = nearFall ? .12 * Math.pow(size, 2) : 0;
      return cumDamage(side.falls, trainStep);
    }
    const st = {
      camera: left.cam,
      update(t) {
        st._t = t;
        const dl = updateSide(left, t), dr = updateSide(right, t);
        dmgL.textContent = dl.toFixed(0); dmgR.textContent = dr.toFixed(1);
      },
      extraRender(r, y) {
        const w2 = r.width / 2;
        [[left, r.left], [right, r.left + w2]].forEach(([side, x]) => {
          renderer.setViewport(x, y, w2, r.height); renderer.setScissor(x, y, w2, r.height);
          side.cam.aspect = w2 / r.height; fitOrtho(side.cam, w2 / r.height); side.cam.updateProjectionMatrix();
          renderer.render(side.sc, side.cam);
        });
      },
    };
    st.update(reduceMotion ? .8 : 0); return st;
  });

  /* ---------------- scroll wiring ---------------- */
  function wireStage(name) {
    const stObj = stages.find(s => s.el && (s.el.dataset.stage === name || s.el.dataset.scene === name));
    if (!stObj) return;
    if (name === 'hero') {
      ScrollTrigger.create({
        trigger: stObj.el.closest('.chapter') || stObj.el, start: 'top top', end: 'bottom top', scrub: 1,
        onUpdate(self) { stObj.update(self.progress * .9); requestRender(); },
      });
      return;
    }
    const scrolly = stObj.el.closest('.scrolly') || stObj.el.parentElement;
    ScrollTrigger.create({
      trigger: scrolly, start: 'top top', end: 'bottom bottom', scrub: 1, invalidateOnRefresh: true,
      onUpdate(self) { stObj.update(self.progress); requestRender(); },
      onToggle() { requestRender(); },
    });
    if (reduceMotion) { stObj.update(name === 'h1' ? .4 : .85); }
  }
  ['hero', 'primer', 'h1', 'h2', 'h3', 'h4', 'h5'].forEach(wireStage);

  // step cards fade in/out
  gsap.utils.toArray('.step .card').forEach(card => {
    gsap.fromTo(card, { autoAlpha: 0, y: 26 }, {
      autoAlpha: 1, y: 0, duration: reduceMotion ? 0 : .6, ease: 'power2.out',
      scrollTrigger: { trigger: card.parentElement, start: 'top 70%', end: 'bottom 20%', toggleActions: 'play none none reverse', fastScrollEnd: true },
    });
  });
  // charts draw on entry
  Object.entries(chartDraws).forEach(([k, obj]) => {
    if (!obj) return;
    const svg = document.getElementById('chart-' + k); if (!svg) return;
    if (reduceMotion) { obj.draw(1); return; }
    const state = { p: 0 };
    gsap.to(state, {
      p: 1, ease: 'none',
      scrollTrigger: { trigger: svg, start: 'top 85%', end: 'top 30%', scrub: 1 },
      onUpdate: () => obj.draw(state.p),
    });
  });

  /* ---------------- growth ruler ---------------- */
  (function ruler() {
    const wrap = document.querySelector('#ruler .tickwrap'); if (!wrap) return;
    const chapters = [
      ['hero', 'birth'], ['ch-primer', 'one table'], ['ch1', 'blindfold'], ['ch2', 'drills'],
      ['ch3', 'variation'], ['ch4', 'teachers'], ['ch5', 'growing'], ['ch6', 'lesson'], ['ch7', 'methods'],
    ];
    const pencil = wrap.querySelector('.pencil');
    const btns = [];
    chapters.forEach(([id, nm], i) => {
      const b = document.createElement('button');
      b.innerHTML = `<span class="mark"></span><span class="nm">${nm}</span>`;
      b.style.bottom = (i / (chapters.length - 1)) * 100 + '%';
      b.addEventListener('click', () => {
        const target = document.getElementById(id);
        if (target) gsap.to(window, { scrollTo: { y: target, autoKill: true }, duration: reduceMotion ? 0 : .9, ease: 'power2.inOut' });
      });
      wrap.appendChild(b); btns.push(b);
    });
    const fill = document.querySelector('#topbar .fill');
    ScrollTrigger.create({
      start: 0, end: () => document.documentElement.scrollHeight - window.innerHeight, scrub: 0,
      onUpdate(self) {
        pencil.style.height = (self.progress * 100) + '%';
        if (fill) fill.style.width = (self.progress * 100) + '%';
        const mid = window.scrollY + window.innerHeight * .5;
        let idx = 0;
        chapters.forEach(([id], i) => {
          const elc = document.getElementById(id);
          if (elc && elc.offsetTop <= mid) idx = i;
        });
        btns.forEach((b, i) => b.classList.toggle('active', i === idx));
      },
    });
  })();

  onResize();
  document.fonts && document.fonts.ready.then(() => { ScrollTrigger.refresh(); requestRender(); });
  window.addEventListener('load', () => { ScrollTrigger.refresh(); requestRender(); });
})();
