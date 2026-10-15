'use strict';
(function () {
  const STATUS_URL = 'status.cgi';
  const ANALYZE_URL = 'analyze.cgi';
  const LABEL_URL = 'label.cgi';
  const TIMELINE_URL = 'timeline.cgi';
  const TIMELINE_MAX = 625;  // ~5 min of 0.48 s windows (matches the device ring)
  const LIVE_WINDOWS = 250;  // 2 min shown on Live
  // Lanes of the key-sounds chart. g = index into the device's group array.
  const LANES = [
    { label: 'Train event', kind: 'state', color: '#F2B705' },
    { label: 'Train score', kind: 'score', color: '#1F4FA8' },
    { label: 'Rail sounds', g: 0, color: '#1F4FA8' },
    { label: 'Train horn', g: 1, color: '#C25E00', horn: true },
    { label: 'Other horns', g: 2, color: '#B3261E' },
    { label: 'Road vehicles', g: 3, color: '#3B4652' },
    { label: 'Speech', g: 4, color: '#6A3FB0' },
    { label: 'Music', g: 5, color: '#1E7A48' }
  ];
  const PARAM_URL = '/axis-cgi/param.cgi';
  const HOP_S = 0.48;
  const SERIES_MAX = 600; // ~10 min of 1 Hz status samples
  const PARAMS = ['Enabled', 'Sensitivity', 'TriggerSeconds', 'ReleaseSeconds', 'SmoothingSeconds',
    'MinLevelDbfs', 'CooldownSeconds', 'HornSensitivity', 'HornTriggers', 'ScoreSource', 'MaxEventSeconds'];
  const DEFAULTS = { Enabled: 'yes', Sensitivity: '50', TriggerSeconds: '3', ReleaseSeconds: '8', SmoothingSeconds: '2',
    MinLevelDbfs: '-65', CooldownSeconds: '10', HornSensitivity: '50', HornTriggers: 'yes', ScoreSource: 'head',
    MaxEventSeconds: '600' };
  const PRESETS = {
    rural: { Sensitivity: '62', TriggerSeconds: '2', ReleaseSeconds: '6', SmoothingSeconds: '2', MinLevelDbfs: '-70', CooldownSeconds: '10' },
    suburban: { Sensitivity: '50', TriggerSeconds: '3', ReleaseSeconds: '8', SmoothingSeconds: '2', MinLevelDbfs: '-65', CooldownSeconds: '10' },
    urban: { Sensitivity: '35', TriggerSeconds: '6', ReleaseSeconds: '10', SmoothingSeconds: '3', MinLevelDbfs: '-55', CooldownSeconds: '20' }
  };
  const UNITS = { TriggerSeconds: ' s', ReleaseSeconds: ' s', SmoothingSeconds: ' s', CooldownSeconds: ' s', MinLevelDbfs: ' dBFS', MaxEventSeconds: ' s' };
  const FETCH_TIMEOUT_MS = 5000;
  // fetch() with a timeout so a busy device shows "lost connection" instead of freezing the UI.
  function fetchT(url, opts, ms) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ms || FETCH_TIMEOUT_MS);
    return fetch(url, Object.assign({ credentials: 'same-origin', signal: ctl.signal }, opts || {}))
      .finally(() => clearTimeout(timer));
  }
  const RAIL_RE = /train|rail|subway/i;
  const C = { score: '#1F4FA8', scoreSoft: '#9DB3DB', red: '#B3261E', grid: '#E3E6E9', event: '#FBE7A6', axis: '#C5CCD3' };

  const $ = (id) => document.getElementById(id);
  const text = (el, s) => { el.textContent = s; };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const thrOf = (s) => 0.95 - 0.009 * (clamp(+s, 1, 100) - 1);
  const hornThrOf = (s) => 0.6 - 0.0055 * (clamp(+s, 1, 100) - 1);
  const levelPct = (db) => clamp((db + 90) / 80 * 100, 0, 100);
  const pad2 = (n) => String(n).padStart(2, '0');
  const clock = (t) => { const d = new Date(t * 1000); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`; };
  const hhmm = (t) => { const d = new Date(t * 1000); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`; };
  const mmss = (s) => `${Math.floor(s / 60)}:${pad2(Math.floor(s % 60))}`;
  const fmtDb = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
  const midnight = () => { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime() / 1000; };

  let status = null;
  const series = [];
  let timeline = [];          // {seq, t, head, score, level, state, g:[6]}
  let timelineSeq = 0;
  let paramGroup = null;
  const params = {};          // last loaded/saved value per param (strings)
  const dirty = new Set();
  let settingsLoaded = false;
  let histFilter = 'all';
  let testWav = null;         // ArrayBuffer of 16 kHz mono PCM16 WAV
  let lastResult = null;

  // ------------------------------------------------------------------ routing
  const VIEWS = ['live', 'tuning', 'history', 'test'];
  function currentView() { const v = location.hash.slice(1); return VIEWS.includes(v) ? v : 'live'; }
  function route() {
    const v = currentView();
    VIEWS.forEach((name) => { $('view-' + name).hidden = name !== v; });
    document.querySelectorAll('.nav a').forEach((a) => {
      if (a.dataset.view === v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (status) render();
  }
  window.addEventListener('hashchange', route);

  // ------------------------------------------------------------------ charts
  function drawBars(canvas, vals, opt) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    c.strokeStyle = C.grid; c.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((g) => { c.beginPath(); c.moveTo(0, Math.round(h - g * h) + 0.5); c.lineTo(w, Math.round(h - g * h) + 0.5); c.stroke(); });
    const n = Math.max(opt.slots || vals.length, 1);
    const slot = w / n, gap = slot > 4 ? 1.5 : (slot > 2 ? 1 : 0);
    const off = n - vals.length; // right-align so "now" is at the right edge
    if (opt.shade) {
      c.fillStyle = C.event;
      vals.forEach((v, i) => { if (opt.shade[i]) c.fillRect((off + i) * slot, 0, slot + 0.5, h); });
    }
    vals.forEach((v, i) => {
      const bh = clamp(v, 0, 1) * (h - 2);
      c.fillStyle = opt.threshold !== undefined && v >= opt.threshold ? C.score : C.scoreSoft;
      c.fillRect((off + i) * slot + gap / 2, h - bh, Math.max(1, slot - gap), bh);
    });
    c.strokeStyle = C.axis; c.beginPath(); c.moveTo(0, h - 0.5); c.lineTo(w, h - 0.5); c.stroke();
    if (opt.threshold !== undefined) {
      const y = Math.round(h - opt.threshold * h) + 0.5;
      c.strokeStyle = C.red; c.lineWidth = 2; c.setLineDash([5, 4]);
      c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); c.setLineDash([]);
      if (opt.thresholdLabel) {
        c.font = '600 12px ' + getComputedStyle(document.body).getPropertyValue('--mono');
        const label = opt.threshold.toFixed(2), tw = c.measureText(label).width;
        c.fillStyle = '#fff'; c.fillRect(w - tw - 8, y - 16, tw + 8, 14);
        c.fillStyle = C.red; c.fillText(label, w - tw - 4, y - 5);
      }
    }
  }

  function hexAlpha(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // rows: [{score, state, g:[...]}], right-aligned in `slots` columns.
  function drawLanes(canvas, rows, opt) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const gutter = 116, laneH = (h - 4) / LANES.length, cellH = laneH - 4;
    const n = Math.max(opt.slots || rows.length, 1), cw = (w - gutter) / n, off = n - rows.length;
    c.font = '12px ' + getComputedStyle(document.body).getPropertyValue('--sans');
    c.textBaseline = 'middle';
    LANES.forEach((lane, li) => {
      const y = li * laneH + 2;
      c.fillStyle = '#3B4652';
      c.fillText(lane.label, 0, y + cellH / 2);
      c.fillStyle = '#F1F3F5';
      c.fillRect(gutter, y, w - gutter, cellH);
      rows.forEach((r, i) => {
        let v;
        if (lane.kind === 'state') v = r.state === 2 ? 1 : (r.state === 1 ? 0.35 : 0);
        else if (lane.kind === 'score') v = r.score;
        else v = r.g[lane.g];
        if (!(v > 0.02)) return;
        const x = gutter + (off + i) * cw;
        c.fillStyle = hexAlpha(lane.color, Math.min(1, 0.12 + 0.88 * v));
        c.fillRect(x, y, Math.max(1, cw - (cw > 3 ? 0.5 : 0)), cellH);
        const hit = (lane.kind === 'score' && opt.thr !== undefined && v >= opt.thr) || (lane.horn && opt.hornThr !== undefined && v >= opt.hornThr);
        if (hit) { c.strokeStyle = '#12161B'; c.lineWidth = 1.5; c.strokeRect(x + 0.75, y + 0.75, Math.max(1, cw - 1.5), cellH - 1.5); }
      });
    });
  }

  function lanesSummary(rows) {
    if (!rows.length) return 'Collecting…';
    const peak = (k) => Math.max(...rows.map((r) => r.g[k]));
    const share = (k) => Math.round(rows.filter((r) => r.g[k] >= 0.3).length / rows.length * 100);
    return `Peaks in view — rail ${peak(0).toFixed(2)}, train horn ${peak(1).toFixed(2)}, other horns ${peak(2).toFixed(2)}, ` +
      `road vehicles ${peak(3).toFixed(2)} · speech in ${share(4)}% of windows, music in ${share(5)}%`;
  }

  // ------------------------------------------------------------------ detector simulation (mirrors detector.c)
  // Mirrors detector.c: count-based moving average over round(smooth/0.48) windows,
  // loudness gate, trigger/release timers, horn-triggered starts (>= 2 consecutive
  // windows), cooldown and the max-duration cap. points are 0.48 s windows:
  // [{t, raw, horn, level}]; sets p.smooth. Returns events [{start, end}].
  function simulate(points, cfg) {
    const thr = thrOf(cfg.sens), hthr = hornThrOf(cfg.hornSens), events = [];
    const n = clamp(Math.round(cfg.smooth / HOP_S), 1, 64);
    let above = 0, below = 0, active = null, coolUntil = -Infinity, hornRun = 0;
    const buf = [];
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const dt = i ? clamp(p.t - points[i - 1].t, 0.01, 1.0) : HOP_S;
      const gated = p.level < cfg.minLevel;
      buf.push(gated ? 0 : p.raw);
      if (buf.length > n) buf.shift();
      const s = buf.reduce((a, b) => a + b, 0) / buf.length;
      hornRun = !gated && (p.horn || 0) >= hthr ? hornRun + 1 : 0;
      const hornNow = hornRun === 2;
      const isAbove = s >= thr;
      if (!active) {
        above = isAbove ? above + dt : 0;
        if (p.t >= coolUntil && (above >= cfg.trigger || (cfg.hornTriggers && hornNow))) {
          active = { start: p.t - Math.min(above, cfg.trigger) }; below = 0;
        }
      } else {
        below = isAbove ? 0 : below + dt;
        const capped = cfg.maxEvent > 0 && p.t - active.start >= cfg.maxEvent;
        if (below >= cfg.release || capped) {
          events.push({ start: active.start, end: p.t - (capped ? 0 : below) });
          coolUntil = p.t + cfg.cool; active = null; above = 0; below = 0;
        }
      }
      p.smooth = s;
    }
    if (active && points.length) events.push({ start: active.start, end: points[points.length - 1].t, open: true });
    return events;
  }

  function currentCfg() {
    const v = (id) => +$(id).value;
    return { sens: v('Sensitivity'), trigger: v('TriggerSeconds'), release: v('ReleaseSeconds'), smooth: v('SmoothingSeconds'),
      minLevel: v('MinLevelDbfs'), cool: v('CooldownSeconds'), maxEvent: v('MaxEventSeconds'), hornSens: v('HornSensitivity'),
      hornTriggers: $('HornTriggers').checked, source: params.ScoreSource || 'head' };
  }
  const rawOf = (p, source) => (source === 'yamnet' ? p.rail : (source === 'max' ? Math.max(p.head, p.rail) : p.head));

  // ------------------------------------------------------------------ parameters
  async function loadParams() {
    for (const g of ['root.Train_detect', 'root.train_detect']) {
      const r = await fetch(`${PARAM_URL}?action=list&group=${encodeURIComponent(g)}`, { credentials: 'same-origin' });
      const t = await r.text();
      if (r.ok && !/^#\s*Error/m.test(t) && t.includes('=')) {
        t.split(/\r?\n/).forEach((line) => {
          const m = line.match(/^(root\.[^.]+)\.([A-Za-z]+)=(.*)$/);
          if (m && PARAMS.includes(m[2])) { paramGroup = m[1]; params[m[2]] = m[3]; }
        });
        settingsLoaded = PARAMS.every((p) => p in params);
        applyParamsToControls();
        setSave(settingsLoaded ? 'ok' : 'bad', settingsLoaded ? 'All changes saved to the device' : 'Some settings could not be read; saving disabled');
        return;
      }
    }
    throw new Error('settings not readable (administrator rights needed)');
  }

  function applyParamsToControls() {
    PARAMS.forEach((p) => {
      if (!(p in params)) return;
      const el = $(p);
      if (p === 'ScoreSource') {
        el.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === params[p])));
      } else if (p === 'Enabled') {
        setEnabledUi(params[p] === 'yes');
      } else if (el.type === 'checkbox') {
        el.checked = params[p] === 'yes';
      } else {
        el.value = params[p];
      }
      showValue(p);
    });
    updatePresetUi();
    renderTuning();
  }

  function showValue(p) {
    const out = $(p + 'Out');
    if (out) out.value = (p === 'MinLevelDbfs' ? fmtDb(+$(p).value) : $(p).value) + (UNITS[p] || '');
    if (p === 'Sensitivity') text($('thrText'), thrOf($(p).value).toFixed(2));
    if (p === 'HornSensitivity') text($('hornThrText'), hornThrOf($(p).value).toFixed(2));
  }

  function controlValue(p) {
    if (p === 'ScoreSource' || p === 'Enabled') return params[p];
    const el = $(p);
    return el.type === 'checkbox' ? (el.checked ? 'yes' : 'no') : el.value;
  }

  let saveTimer = null;
  function queueSave(p) {
    if (!settingsLoaded || !paramGroup) { setSave('bad', 'Settings not loaded; change not saved'); return; }
    params[p] = controlValue(p);
    dirty.add(p);
    setSave('warn', 'Saving…');
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDirty, 350);
  }

  let saveChain = Promise.resolve();
  function saveDirty() { saveChain = saveChain.then(saveDirtyNow); return saveChain; }
  async function saveDirtyNow() {
    if (!dirty.size) return;
    const q = new URLSearchParams({ action: 'update' });
    [...dirty].forEach((p) => q.set(`${paramGroup}.${p}`, params[p]));
    dirty.clear();
    try {
      const r = await fetchT(`${PARAM_URL}?${q.toString()}`);
      const t = (await r.text()).trim();
      if (r.ok && t === 'OK') setSave('ok', 'All changes saved to the device');
      else setSave('bad', `Save failed (${r.status}) ${t.slice(0, 80)}`);
    } catch (e) {
      setSave('bad', 'Save failed: ' + e.message);
    }
  }

  function setSave(kind, msg) {
    $('saveDot').className = 'dot dot-' + kind;
    text($('saveState'), msg);
  }

  function updatePresetUi() {
    let match = 'custom';
    for (const [name, def] of Object.entries(PRESETS)) {
      if (Object.entries(def).every(([k, v]) => String(+params[k]) === String(+v))) { match = name; break; }
    }
    document.querySelectorAll('#presetGroup button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.preset === match)));
  }

  ['Sensitivity', 'TriggerSeconds', 'ReleaseSeconds', 'SmoothingSeconds', 'MinLevelDbfs', 'CooldownSeconds', 'HornSensitivity', 'MaxEventSeconds'].forEach((p) => {
    $(p).addEventListener('input', () => { showValue(p); params[p] = $(p).value; updatePresetUi(); renderTuning(); });
    $(p).addEventListener('change', () => queueSave(p));
  });
  $('HornTriggers').addEventListener('change', () => { queueSave('HornTriggers'); renderTuning(); });
  $('ScoreSource').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    params.ScoreSource = b.dataset.value;
    $('ScoreSource').querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    queueSave('ScoreSource');
    renderTuning();
  }));
  document.querySelectorAll('#presetGroup button').forEach((b) => b.addEventListener('click', () => {
    const def = PRESETS[b.dataset.preset];
    if (!def) return;
    Object.entries(def).forEach(([k, v]) => { $(k).value = v; showValue(k); queueSave(k); });
    updatePresetUi();
    renderTuning();
  }));
  $('resetBtn').addEventListener('click', () => {
    Object.entries(DEFAULTS).forEach(([k, v]) => { params[k] = v; });
    applyParamsToControls();
    PARAMS.forEach((p) => queueSave(p));
  });

  function setEnabledUi(on) {
    $('enabledToggle').setAttribute('aria-pressed', String(on));
    text($('enabledText'), on ? 'On' : 'Off');
  }
  $('enabledToggle').addEventListener('click', () => {
    const on = $('enabledToggle').getAttribute('aria-pressed') !== 'true';
    setEnabledUi(on);
    params.Enabled = on ? 'yes' : 'no';
    queueSave('Enabled');
  });

  // ------------------------------------------------------------------ status polling
  async function readJson(r) {
    const ct = r.headers.get('content-type') || '';
    if (!ct.includes('json')) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    return j;
  }

  async function poll() {
    try {
      const r = await fetchT(STATUS_URL, { cache: 'no-store' });
      status = await readJson(r);
      const t = status.inference.last_wall || Date.now() / 1000;
      if (!series.length || series[series.length - 1].t !== t) {
        series.push({ t, score: status.scores.score, head: status.scores.head, rail: status.scores.yamnet_rail,
          level: status.scores.level_db, active: status.state === 'active' });
        while (series.length > SERIES_MAX) series.shift();
      }
      await pollTimeline();
      $('connBanner').hidden = true;
      render();
    } catch (e) {
      $('connBanner').hidden = false;
      text($('connBanner'), 'Lost connection to the app: ' + e.message + ' — retrying');
      $('devDot').className = 'dot dot-bad';
      text($('devAudioText'), 'App not responding');
    } finally {
      setTimeout(poll, 1000);
    }
  }

  async function pollTimeline() {
    const r = await fetchT(`${TIMELINE_URL}?since=${timelineSeq}`, { cache: 'no-store' });
    const j = await readJson(r);
    if (j.seq < timelineSeq) timeline = []; // app restarted
    j.rows.forEach((x) => timeline.push({ seq: x[0], t: x[1], head: x[2], score: x[3], level: x[4], state: x[5], g: x.slice(6) }));
    if (timeline.length > TIMELINE_MAX) timeline = timeline.slice(-TIMELINE_MAX);
    timelineSeq = j.seq;
  }

  function render() {
    renderNav();
    const v = currentView();
    if (v === 'live') renderLive();
    else if (v === 'tuning') renderTuning();
    else if (v === 'history') renderHistory();
    else renderTest();
  }

  function renderNav() {
    const a = status.audio;
    $('devDot').className = 'dot ' + (a.streaming ? 'dot-ok' : 'dot-bad');
    text($('devAudioText'), a.streaming ? `Mic streaming · ${a.native_rate / 1000} kHz` : 'No audio from microphone');
  }

  // ------------------------------------------------------------------ live
  function trainsToday(hist) {
    const m = midnight();
    return hist.filter((e) => e.start >= m && e.label !== 'false_alarm').length;
  }

  function renderLive() {
    const s = status, sc = s.scores, cfg = s.config;
    if (!dirty.has('Enabled')) setEnabledUi(cfg.enabled);
    const active = s.state === 'active' && s.current;
    $('heroActive').hidden = !active;
    $('heroIdle').hidden = !!active;
    if (active) {
      const hornAgo = s.last_horn && s.last_horn >= s.current.start ? ` · horn heard ${Math.max(0, Math.round(s.inference.last_wall - s.last_horn))} s ago` : '';
      text($('heroActiveSub'), `Started ${clock(s.current.start)}${hornAgo}`);
      text($('heroElapsed'), mmss(Math.max(0, s.current.elapsed_s || 0)));
      text($('heroConf'), sc.score.toFixed(2));
    } else {
      const last = s.history[0];
      const eb = $('heroIdleEyebrow');
      $('heroIdle').classList.toggle('disabled', !cfg.enabled);
      if (!s.audio.streaming) { eb.className = 'eyebrow bad'; text(eb, 'No audio'); text($('heroIdleTitle'), 'Microphone silent'); text($('heroIdleSub'), 'No audio is reaching the detector. Check the device audio settings.'); }
      else if (!cfg.enabled) { eb.className = 'eyebrow off'; text(eb, 'Paused'); text($('heroIdleTitle'), 'Detection off'); text($('heroIdleSub'), 'Events are not sent. The meters keep running.'); }
      else {
        eb.className = 'eyebrow ok'; text(eb, 'Listening'); text($('heroIdleTitle'), 'No train');
        text($('heroIdleSub'), last ? `Last train ${hhmm(last.start)} · lasted ${Math.round(last.duration_s)} s · peak ${last.peak_score.toFixed(2)}` : 'No trains detected since the app started');
      }
      const n = trainsToday(s.history);
      text($('heroToday'), `${n} train${n === 1 ? '' : 's'}`);
      text($('heroScore'), sc.score.toFixed(2));
    }

    text($('thrLegend'), sc.threshold.toFixed(2));
    const pts = series.slice(-120);
    drawBars($('liveChart'), pts.map((p) => p.score), { threshold: sc.threshold, shade: pts.map((p) => p.active), slots: 120 });

    const tl = timeline.slice(-LIVE_WINDOWS);
    drawLanes($('keysChart'), tl, { slots: LIVE_WINDOWS, thr: sc.threshold, hornThr: sc.horn_threshold });
    text($('keysSummary'), lanesSummary(tl));

    text($('levelVal'), fmtDb(sc.level_db));
    $('levelBar').style.width = levelPct(sc.level_db) + '%';
    $('gateMark').style.left = levelPct(cfg.min_level_db) + '%';
    text($('gateText'), `Gate at ${fmtDb(cfg.min_level_db)} dBFS — quieter audio is ignored${sc.gated ? ' (currently below the gate)' : ''}`);
    text($('hornVal'), sc.horn.toFixed(2));
    $('hornBar').style.width = clamp(sc.horn * 100, 0, 100) + '%';
    $('hornMark').style.left = clamp(sc.horn_threshold * 100, 0, 100) + '%';
    text($('hornLast'), s.last_horn ? `Last horn ${clock(s.last_horn)}` : '');

    $('topList').replaceChildren(...sc.top.map((t) => {
      const li = document.createElement('li');
      const left = document.createElement('div');
      const name = document.createElement('div'); name.className = 'name'; name.textContent = t.name;
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('i'); fill.style.width = clamp(t.score * 100, 0, 100) + '%';
      if (RAIL_RE.test(t.name)) fill.className = 'rail';
      bar.appendChild(fill); left.append(name, bar);
      const val = document.createElement('span'); val.className = 'val'; val.textContent = t.score.toFixed(2);
      li.append(left, val);
      return li;
    }));

    const rows = s.history.slice(0, 5);
    $('recentBody').replaceChildren(...(rows.length ? rows.map((e) => row([
      cell(clock(e.start), 'num'), cell(`${e.duration_s.toFixed(1)} s`, 'num'), pbarCell(e.peak_score),
      cell(`${fmtDb(e.peak_level_db)} dBFS`, 'num'), cell(e.horn ? 'Yes' : '—')])) : [emptyRow(5, 'No detections yet')]));

    const inf = s.inference;
    text($('healthText'), `Inference ${inf.latency_ms_avg.toFixed(0)} ms avg on ${inf.device} · ${inf.errors} errors · ${inf.skipped_windows} skipped · uptime ${(s.uptime_s / 3600).toFixed(1)} h` +
      (s.audio.inject_remaining_s > 0 ? ` · TEST AUDIO playing (${Math.ceil(s.audio.inject_remaining_s)} s left)` : '') +
      (s.last_error ? ` · last error: ${s.last_error}` : ''));
  }

  function cell(v, cls) { const td = document.createElement('td'); td.textContent = v; if (cls) td.className = cls; return td; }
  function row(cells) { const tr = document.createElement('tr'); tr.append(...cells); return tr; }
  function emptyRow(n, msg) { const td = cell(msg, 'muted'); td.colSpan = n; return row([td]); }
  function pbarCell(v) {
    const td = document.createElement('td');
    const wrap = document.createElement('span'); wrap.className = 'pbar';
    const track = document.createElement('span'); const fill = document.createElement('i');
    fill.style.width = clamp(v * 100, 0, 100) + '%'; track.appendChild(fill);
    const num = document.createElement('span'); num.className = 'num'; num.textContent = v.toFixed(2);
    wrap.append(track, num); td.appendChild(wrap);
    return td;
  }

  // ------------------------------------------------------------------ tuning
  function percentile(arr, q) {
    if (!arr.length) return NaN;
    const a = [...arr].sort((x, y) => x - y);
    return a[clamp(Math.floor(q * (a.length - 1)), 0, a.length - 1)];
  }

  function renderTuning() {
    if (!settingsLoaded) return;
    const cfg = currentCfg();
    const thr = thrOf(cfg.sens);
    // Preview: replay the buffered scores through the detector with the slider values.
    // rail for the detector = max over rail classes incl. horn/whistle (as in the C code)
    const pts = timeline.map((x) => ({ t: x.t, raw: rawOf({ head: x.head, rail: Math.max(x.g[0], x.g[1]) }, cfg.source),
      horn: x.g[1], level: x.level }));
    if (pts.length >= 10) {
      const events = simulate(pts, cfg);
      const span = pts[pts.length - 1].t - pts[0].t;
      text($('previewSpan'), span >= 90 ? `${Math.round(span / 60)} minutes` : `${Math.round(span)} seconds`);
      const n = events.length;
      const pv = $('previewText');
      pv.className = 'strong ' + (n ? 'preview-ok' : '');
      text(pv, n ? `${n} event${n === 1 ? '' : 's'} would fire with these settings` : 'No events would fire with these settings');
      drawBars($('previewChart'), pts.map((p) => p.smooth), { threshold: thr, shade: pts.map((p) => events.some((e) => p.t >= e.start && p.t <= e.end)) });
    } else {
      text($('previewSpan'), 'few seconds'); text($('previewText'), 'Collecting data… keep this page open');
    }
    // Timing bar: proportional but readable (sqrt scale)
    const g = (s) => Math.max(1.2, Math.sqrt(Math.max(s, 0.1)) * 2);
    text($('tbTrig'), (+cfg.trigger).toFixed(1)); text($('tbRel'), cfg.release); text($('tbCool'), cfg.cool);
    document.querySelector('.tb-trig').style.flexGrow = g(cfg.trigger);
    document.querySelector('.tb-rel').style.flexGrow = g(cfg.release);
    document.querySelector('.tb-cool').style.flexGrow = cfg.cool > 0 ? g(cfg.cool) : 0.001;
    // Loudness gate visual
    const idleLevels = series.filter((p) => !p.active).map((p) => p.level);
    const lo = percentile(idleLevels, 0.1), hi = percentile(idleLevels, 0.9);
    if (!Number.isNaN(lo)) {
      $('gateBg').style.left = levelPct(lo) + '%'; $('gateBg').style.width = Math.max(1, levelPct(hi) - levelPct(lo)) + '%';
      text($('gateBgText'), `Background ${fmtDb(lo)}…${fmtDb(hi)} dBFS`);
    }
    const peaks = status ? status.history.filter((e) => e.label !== 'false_alarm').map((e) => e.peak_level_db) : [];
    $('gateTrain').hidden = !peaks.length;
    if (peaks.length) {
      const a = Math.min(...peaks) - 6, b = Math.max(...peaks);
      $('gateTrain').style.left = levelPct(a) + '%'; $('gateTrain').style.width = Math.max(1, levelPct(b) - levelPct(a)) + '%';
      text($('gateTrainText'), `Trains ${fmtDb(a)}…${fmtDb(b)} dBFS`);
    }
    $('gateLine').style.left = levelPct(cfg.minLevel) + '%';
    if (status) $('gateLive').style.left = levelPct(status.scores.level_db) + '%';
  }

  // ------------------------------------------------------------------ history
  async function setLabel(e, label) {
    const next = e.label === label ? 'none' : label;
    try {
      const r = await fetch(`${LABEL_URL}?start=${encodeURIComponent(e.start.toFixed(3))}&label=${next}`, {
        method: 'POST', credentials: 'same-origin', headers: { 'X-Requested-With': 'traindet' } });
      await readJson(r);
      e.label = next === 'none' ? null : next;
      renderHistory();
    } catch (err) {
      alert('Could not save the answer: ' + err.message);
    }
  }

  function renderHistory() {
    const hist = status.history;
    text($('histDate'), new Date().toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' }) +
      ` · ${hist.length} event${hist.length === 1 ? '' : 's'} in memory`);
    const m = midnight();
    const today = hist.filter((e) => e.start >= m);
    const fa = today.filter((e) => e.label === 'false_alarm').length;
    const durs = hist.map((e) => e.duration_s).sort((a, b) => a - b);
    const med = durs.length ? durs[Math.floor(durs.length / 2)] : null;
    const longest = hist.reduce((a, e) => (!a || e.duration_s > a.duration_s ? e : a), null);
    const hours = {};
    today.forEach((e) => { const h = new Date(e.start * 1000).getHours(); hours[h] = (hours[h] || 0) + 1; });
    const busiest = Object.entries(hours).sort((a, b) => b[1] - a[1])[0];
    const tiles = [
      ['Trains today', String(today.length - fa), fa ? `${fa} marked as false alarm` : 'none marked as false alarm'],
      ['Median duration', med === null ? '—' : `${Math.round(med)} s`, longest ? `longest ${Math.round(longest.duration_s)} s at ${hhmm(longest.start)}` : 'no events yet'],
      ['Horns heard', String(status.horns_total), `${hist.filter((e) => e.horn).length} during an event`],
      ['Busiest hour', busiest ? `${pad2(+busiest[0])}–${pad2((+busiest[0] + 1) % 24)}` : '—', busiest ? `${busiest[1]} train${busiest[1] > 1 ? 's' : ''}` : 'no trains today']
    ];
    $('statTiles').replaceChildren(...tiles.map(([l, v, sub]) => {
      const d = document.createElement('div'); d.className = 'stat-tile';
      const a = document.createElement('span'); a.className = 'small muted'; a.textContent = l;
      const b = document.createElement('span'); b.className = 'v'; b.textContent = v;
      const c = document.createElement('span'); c.className = 'small muted'; c.textContent = sub;
      d.append(a, b, c); return d;
    }));

    const now = Date.now() / 1000, from = now - 86400;
    $('dayStrip').replaceChildren(...hist.filter((e) => e.end >= from).map((e) => {
      const i = document.createElement('i');
      i.style.left = clamp((e.start - from) / 86400 * 100, 0, 100) + '%';
      i.style.width = Math.max(0.3, e.duration_s / 86400 * 100) + '%';
      i.title = `${clock(e.start)} · ${e.duration_s.toFixed(0)} s`;
      if (e.label === 'false_alarm') i.className = 'fa';
      return i;
    }));
    $('dayAxis').replaceChildren(...[24, 18, 12, 6, 0].map((h) => {
      const s = document.createElement('span'); s.textContent = h ? hhmm(now - h * 3600) : 'now'; return s;
    }));

    document.querySelectorAll('#histFilters button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.filter === histFilter)));
    const rows = hist.filter((e) => histFilter === 'all' || (histFilter === 'horn' ? e.horn : !e.label));
    $('histBody').replaceChildren(...(rows.length ? rows.map((e) => {
      const fb = document.createElement('td');
      const box = document.createElement('div'); box.className = 'fb';
      [['train', 'Train'], ['false_alarm', 'False alarm']].forEach(([lab, txt]) => {
        const b = document.createElement('button'); b.type = 'button'; b.dataset.label = lab; b.textContent = txt;
        b.setAttribute('aria-pressed', String(e.label === lab));
        b.addEventListener('click', () => setLabel(e, lab));
        box.appendChild(b);
      });
      fb.appendChild(box);
      const scoreTd = document.createElement('td');
      const pk = document.createElement('span'); pk.className = 'num'; pk.textContent = e.peak_score.toFixed(2);
      const mean = document.createElement('span'); mean.className = 'small muted'; mean.textContent = ` peak · ${e.mean_score.toFixed(2)} mean`;
      scoreTd.append(pk, mean);
      return row([cell(clock(e.start), 'num'), cell(`${e.duration_s.toFixed(1)} s`, 'num'), scoreTd,
        cell(`${fmtDb(e.peak_level_db)} dBFS`, 'num'),
        cell((e.horn ? 'Yes' : '—') + (e.test ? ' · TEST' : '') + (e.capped ? ' · capped' : '')), fb]);
    }) : [emptyRow(6, hist.length ? 'Nothing matches this filter' : 'No detections yet')]));
  }

  document.querySelectorAll('#histFilters button').forEach((b) => b.addEventListener('click', () => { histFilter = b.dataset.filter; renderHistory(); }));

  $('exportBtn').addEventListener('click', () => {
    if (!status) return;
    const lines = [['start_utc', 'end_utc', 'duration_s', 'peak_score', 'mean_score', 'peak_level_dbfs', 'horn', 'test', 'capped', 'label'].join(',')];
    status.history.forEach((e) => lines.push([new Date(e.start * 1000).toISOString(), new Date(e.end * 1000).toISOString(),
      e.duration_s.toFixed(1), e.peak_score.toFixed(3), e.mean_score.toFixed(3), e.peak_level_db.toFixed(1), e.horn ? 1 : 0,
      e.test ? 1 : 0, e.capped ? 1 : 0, e.label || ''].join(',')));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
    a.download = `train-detections-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ------------------------------------------------------------------ test audio
  function encodeWav16(samples) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
    str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, clamp(Math.round(samples[i] * 32767), -32768, 32767), true);
    return buf;
  }

  async function loadFile(file) {
    $('testError').hidden = true;
    $('resultCard').hidden = true;
    lastResult = null;
    testWav = null; busy(true);
    $('fileInfo').hidden = false;
    text($('fileName'), file.name); text($('fileMeta'), 'decoding…');
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    try {
      const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
      const secs = Math.min(decoded.duration, 60);
      const off = new OfflineAudioContext(1, Math.ceil(secs * 16000), 16000);
      const src = off.createBufferSource();
      src.buffer = decoded; src.connect(off.destination); src.start();
      const rendered = await off.startRendering();
      testWav = encodeWav16(rendered.getChannelData(0));
      text($('fileMeta'), `${decoded.duration.toFixed(1)} s${decoded.duration > 60 ? ' (first 60 s used)' : ''} · ${decoded.numberOfChannels > 1 ? 'stereo, mixed to mono' : 'mono'} · sent as 16 kHz`);
    } catch (e) {
      text($('fileMeta'), 'unreadable');
      showTestError('Could not decode this file: ' + (e.message || e));
    } finally {
      ctx.close();
      busy(false);
    }
  }

  function busy(on) { $('analyzeBtn').disabled = $('injectBtn').disabled = on || !testWav; }
  function showTestError(msg) { $('testError').hidden = false; text($('testError'), msg); }

  $('wavFile').addEventListener('change', () => { if ($('wavFile').files[0]) loadFile($('wavFile').files[0]); });
  const dz = $('dropZone');
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

  $('analyzeBtn').addEventListener('click', async () => {
    if (!testWav) return;
    $('testError').hidden = true; busy(true);
    text($('analyzeBtn'), 'Analyzing…');
    try {
      const r = await fetch(ANALYZE_URL, { method: 'POST', body: testWav, credentials: 'same-origin', headers: { 'Content-Type': 'audio/wav' } });
      lastResult = await readJson(r);
      renderTest();
    } catch (e) {
      showTestError('Analysis failed: ' + e.message);
    } finally {
      text($('analyzeBtn'), 'Analyze on device');
      busy(false);
    }
  });

  $('injectBtn').addEventListener('click', async () => {
    if (!testWav) return;
    if (!window.confirm('Play this file into the live detector?\n\nThe microphone is ignored while it plays, and real ' +
      'events fire (marked as test events) — action rules such as recordings or notifications may run.')) return;
    $('testError').hidden = true; busy(true);
    try {
      const r = await fetch(ANALYZE_URL + '?mode=inject', { method: 'POST', body: testWav, credentials: 'same-origin', headers: { 'Content-Type': 'audio/wav' } });
      const j = await readJson(r);
      $('injectBanner').hidden = false;
      text($('injectText'), `Test audio is playing into the live detector — ${Math.ceil(j.duration_s)} s left`);
    } catch (e) {
      showTestError('Could not start playback: ' + e.message);
    } finally {
      busy(false);
    }
  });

  function renderTest() {
    if (status) {
      const left = status.audio.inject_remaining_s;
      if (left > 0) { $('injectBanner').hidden = false; text($('injectText'), `Test audio is playing into the live detector — ${Math.ceil(left)} s left`); }
      else $('injectBanner').hidden = true;
    }
    if (!lastResult || !settingsLoaded) return;
    const res = lastResult, w = res.windows;
    const cfg = currentCfg();
    const thr = thrOf(cfg.sens);
    const pts = w.map((x) => ({ t: x.t, raw: rawOf({ head: x.prob, rail: x.rail }, cfg.source), horn: x.horn, level: x.level_db }));
    const events = simulate(pts, cfg);
    $('resultCard').hidden = false;
    const v = $('verdict'), icon = $('verdictIcon');
    if (events.length) {
      v.className = 'verdict yes';
      icon.innerHTML = '<circle cx="28" cy="28" r="25" fill="#1E7A48"/><path d="M17 29l7 7 15-16" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>';
      text($('verdictTitle'), events.length > 1 ? `Would trigger ${events.length} train events` : 'Would trigger a train event');
      const e0 = events[0];
      text($('verdictSub'), `from ${mmss(e0.start)} to ${mmss(Math.min(res.duration_s, e0.end + HOP_S * 2))} with your current settings (sensitivity ${cfg.sens}, trigger ${cfg.trigger} s)`);
    } else {
      v.className = 'verdict no';
      icon.innerHTML = '<circle cx="28" cy="28" r="25" fill="#56606B"/><path d="M19 28h18" stroke="#fff" stroke-width="5" stroke-linecap="round"/>';
      text($('verdictTitle'), 'Would not trigger an event');
      const peakSmooth = pts.length ? Math.max(...pts.map((p) => p.smooth)) : 0;
      text($('verdictSub'), peakSmooth < thr
        ? `The smoothed score peaks at ${peakSmooth.toFixed(2)}, below the threshold ${thr.toFixed(2)}`
        : `The score crosses ${thr.toFixed(2)} but not for the ${cfg.trigger} s trigger duration`);
    }
    text($('resPeak'), res.max_prob.toFixed(2));
    text($('resAbove'), `${pts.filter((p) => p.smooth >= thr).length} / ${pts.length}`);
    text($('resLatency'), `${Math.round(w.reduce((a, x) => a + x.latency_ms, 0) / Math.max(1, w.length))} ms`);
    drawBars($('resultChart'), pts.map((p) => p.smooth), { threshold: thr, thresholdLabel: true,
      shade: pts.map((p) => events.some((e) => p.t >= e.start && p.t <= e.end)) });
    if (w.length && w[0].groups) {
      const inEvent = (t) => events.some((e) => t >= e.start && t <= e.end);
      drawLanes($('resultLanes'), w.map((x, i) => ({ score: pts[i].smooth, state: inEvent(x.t) ? 2 : 0, g: x.groups })),
        { slots: w.length, thr, hornThr: hornThrOf(+$('HornSensitivity').value) });
    }
    const dur = res.duration_s;
    $('resultAxis').replaceChildren(...[0, 0.25, 0.5, 0.75, 1].map((f) => { const s = document.createElement('span'); s.textContent = mmss(dur * f); return s; }));
    const best = {};
    w.forEach((x) => x.top.forEach((t) => { best[t.name] = Math.max(best[t.name] || 0, t.score); }));
    const chips = Object.entries(best).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const label = document.createElement('span'); label.className = 'small muted'; label.textContent = 'Heard most:';
    $('resultChips').replaceChildren(label, ...chips.map(([n, s]) => {
      const c = document.createElement('span'); c.className = 'chip' + (RAIL_RE.test(n) ? ' rail' : ''); c.textContent = `${n} ${s.toFixed(2)}`; return c;
    }));
  }

  // ------------------------------------------------------------------ device name (unrestricted VAPIX properties)
  fetch('/axis-cgi/basicdeviceinfo.cgi', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ apiVersion: '1.0', method: 'getAllUnrestrictedProperties' }) })
    .then((r) => r.json()).then((j) => {
      const p = (j.data && j.data.propertyList) || {};
      if (p.ProdShortName) text($('devName'), p.ProdShortName);
    }).catch(() => {});

  window.addEventListener('resize', () => { if (status) render(); });

  route();
  loadParams().catch((e) => setSave('bad', 'Settings unavailable: ' + e.message));
  poll();
})();
