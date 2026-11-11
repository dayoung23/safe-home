/* ─────────────────────────────────────────────────────────────
   Safe Home 공통 스크립트
   - 상태바 시계, 폰 크기 맞춤, API 호출, 토스트
   - 알림 감시: /api/alerts 를 1.5초마다 조회 → 새 알림/갱신 시 푸시 배너 · 위험 경보 화면 · 소리 · 진동
   - 상태 감시: /api/status (카메라·센서·판단 로그) 를 구독한 화면에만 조회
   ───────────────────────────────────────────────────────────── */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const LEVEL_RANK = { "안전": 0, "주의": 1, "위험": 2 };
  const SRC = {
    sound: { ko: "소리", icon: "i-mic" },
    vision: { ko: "영상", icon: "i-video" },
    keypad: { ko: "도어락", icon: "i-lock" },
  };
  const SRC_BY_KO = { "소리": "sound", "영상": "vision", "도어락": "keypad" };
  const isAuth = document.body.dataset.auth === "1";

  /* ── 폰 크기를 화면 높이에 맞춤 ── */
  function fitPhone() {
    const z = Math.min(1, (window.innerHeight - 32) / 872, (window.innerWidth - 24) / 408);
    document.documentElement.style.setProperty("--phone-zoom", Math.max(z, 0.55).toFixed(3));
  }
  fitPhone();
  window.addEventListener("resize", fitPhone);

  /* ── 상태바 시계 ── */
  function tick() {
    const d = new Date();
    const el = $("#sbTime");
    if (el) el.textContent = `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
  }
  tick();
  setInterval(tick, 10000);

  /* ── 설정값 (설정 화면 토글과 공유) ── */
  const pref = {
    get(key, def) { try { const v = localStorage.getItem("sh_" + key); return v === null ? def : v === "1"; } catch { return def; } },
    set(key, val) { try { localStorage.setItem("sh_" + key, val ? "1" : "0"); } catch {} },
  };

  /* ── API ── */
  async function api(url, opts = {}) {
    const init = { ...opts };
    if (opts.json !== undefined) {
      init.method = init.method || "POST";
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(opts.json);
      delete init.json;
    }
    const res = await fetch(url, init);
    if (res.status === 401 && isAuth) { location.href = "/login"; throw new Error("401"); }
    let data = null;
    try { data = await res.json(); } catch {}
    return { ok: res.ok, status: res.status, data };
  }

  /* ── 토스트 ── */
  let toastTimer;
  function toast(msg, ms = 2400) {
    const el = $("#toast");
    if (!el) return;
    el.textContent = msg;
    el.hidden = false;
    el.style.animation = "none"; void el.offsetWidth; el.style.animation = "";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }

  /* ── 알림 문장 해석: judge 가 만든 reason / message 를 화면용으로 나눔 ── */
  function parseAlert(a) {
    let reason = a.reason || "";
    let stage = "";
    if (reason.includes("(확인 중)")) stage = "checking";
    if (/종합 판정/.test(reason)) stage = "final";
    const title = reason.replace(/\s*\(확인 중\)\s*$/, "").replace(/\s*·\s*\d+초 종합 판정\s*$/, "").trim()
      || (a.message || "").split(" — ")[0];
    const detail = (a.message || "").includes(" — ") ? a.message.split(" — ").slice(1).join(" — ") : "";
    const lines = detail ? detail.split(" / ").map((t) => {
      const m = t.match(/^(소리|영상|도어락):\s*(.*)$/);
      return m ? { src: SRC_BY_KO[m[1]], ko: m[1], text: m[2] } : { src: null, ko: "", text: t };
    }) : [];
    return { ...a, title, stage, lines };
  }

  function timeAgo(iso) {
    if (!iso) return "";
    const t = new Date(iso).getTime();
    const s = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (s < 10) return "방금";
    if (s < 60) return `${s}초 전`;
    if (s < 3600) return `${Math.floor(s / 60)}분 전`;
    if (s < 86400) return `${Math.floor(s / 3600)}시간 전`;
    return new Date(iso).toLocaleDateString("ko-KR");
  }
  function hhmm(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  }

  /* ── 확인(읽음) 처리 ── */
  const acked = new Set(JSON.parse(localStorage.getItem("sh_acked") || "[]"));
  function ack(id) { acked.add(id); localStorage.setItem("sh_acked", JSON.stringify([...acked].slice(-200))); }
  function isAcked(id) { return acked.has(id); }

  /* ── 소리 (Web Audio) ── */
  let ctx = null;
  function audio() {
    if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
    return ctx;
  }
  function updateUnlock() {
    const b = $("#soundUnlock");
    if (!b) return;
    const c = audio();
    b.hidden = !isAuth || !pref.get("sound", true) || !c || c.state === "running";
  }
  document.addEventListener("pointerdown", () => { const c = audio(); if (c && c.state !== "running") c.resume().then(updateUnlock); }, { capture: true });
  const unlockBtn = $("#soundUnlock");
  if (unlockBtn) unlockBtn.addEventListener("click", () => { const c = audio(); c && c.resume().then(() => { updateUnlock(); toast("알림 소리가 켜졌어요"); }); });
  setTimeout(updateUnlock, 300);

  function tone(freq, start, dur, type = "sine", gain = 0.18) {
    const c = audio(); if (!c || c.state !== "running") return;
    const o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, c.currentTime + start);
    g.gain.setValueAtTime(0.0001, c.currentTime + start);
    g.gain.exponentialRampToValueAtTime(gain, c.currentTime + start + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + start + dur);
    o.connect(g).connect(c.destination);
    o.start(c.currentTime + start); o.stop(c.currentTime + start + dur + 0.05);
  }
  function playSound(level) {
    if (!pref.get("sound", true)) return;
    if (level === "위험") {                     // 사이렌: 높낮이 반복
      for (let i = 0; i < 6; i++) { tone(880, i * 0.42, 0.2, "square", 0.08); tone(660, i * 0.42 + 0.21, 0.2, "square", 0.08); }
    } else if (level === "주의") {
      tone(784, 0, 0.18); tone(988, 0.2, 0.28);
    } else {
      tone(660, 0, 0.15, "sine", 0.12); tone(880, 0.16, 0.22, "sine", 0.12);
    }
  }
  function vibrate(level) {
    if (!pref.get("vibe", true)) return;
    if (navigator.vibrate) navigator.vibrate(level === "위험" ? [400, 150, 400, 150, 600] : [180]);
    const phone = $("#phone");
    if (phone && level === "위험") { phone.classList.remove("shake"); void phone.offsetWidth; phone.classList.add("shake"); }
  }

  /* ── 푸시 배너 ── */
  function banner({ level, title, text, id, kind }) {
    const layer = $("#notifLayer");
    if (!layer) return;
    const el = document.createElement("div");
    el.className = `notif lv-${level || "안전"}`;
    el.innerHTML = `
      <div class="notif-app"><svg class="i"><use href="#i-shield"/></svg></div>
      <div class="notif-body">
        <div class="notif-meta"><span>Safe Home · ${kind || "알림"}</span><span>지금</span></div>
        <div class="notif-title"></div><div class="notif-text"></div>
      </div>`;
    el.querySelector(".notif-title").textContent = title;
    el.querySelector(".notif-text").textContent = text || "";
    el.addEventListener("click", () => { close(); openDetail(id); });
    layer.prepend(el);
    while (layer.children.length > 3) layer.lastChild.remove();
    let closed = false;
    function close() { if (closed) return; closed = true; el.classList.add("out"); setTimeout(() => el.remove(), 300); }
    setTimeout(close, level === "위험" ? 9000 : 6000);
  }

  /* ── 위험 경보 전체 화면 ── */
  let alarmEl = null;
  function alarm(a) {
    const screen = $(".screen");
    if (!screen) return;
    if (!alarmEl) {
      alarmEl = document.createElement("div");
      alarmEl.className = "alarm";
      alarmEl.innerHTML = `
        <div class="siren"><span></span><span></span><span></span><div class="siren-core"><svg class="i"><use href="#i-alert"/></svg></div></div>
        <h2>위험 감지</h2>
        <div class="d-time"></div>
        <div class="d-msg"></div>
        <div class="d-actions">
          <button class="btn white block" data-act="call"><svg class="i"><use href="#i-phone"/></svg>112 신고하기</button>
          <button class="btn outline block" data-act="view">자세히 보기</button>
          <button class="btn outline block" data-act="close">괜찮아요, 닫기</button>
        </div>`;
      screen.appendChild(alarmEl);
      alarmEl.addEventListener("click", (e) => {
        const act = e.target.closest("[data-act]")?.dataset.act;
        if (!act) return;
        const id = alarmEl.dataset.id;
        alarmEl.hidden = true;
        if (act === "call") toast("📞 112에 연결합니다 (시연 화면)");
        if (act === "view") openDetail(id);
      });
    }
    alarmEl.dataset.id = a.id;
    alarmEl.querySelector(".d-time").textContent = `${hhmm(a.updated_at)} · 우리 집 홈캠`;
    alarmEl.querySelector(".d-msg").textContent = a.title;
    alarmEl.hidden = false;
  }

  function openDetail(id) {
    if (!id) return;
    if (typeof window.openAlertDetail === "function") window.openAlertDetail(id);
    else location.href = `/?alert=${encodeURIComponent(id)}`;
  }

  /* ── 탭 제목 깜빡임 (다른 창 보고 있을 때) ── */
  let titleTimer;
  function flashTitle(text) {
    const base = document.title.replace(/^\(.+?\)\s*/, "");
    clearInterval(titleTimer);
    let on = false, n = 0;
    titleTimer = setInterval(() => {
      document.title = (on = !on) ? `(${text}) ${base}` : base;
      if (++n > 12) { clearInterval(titleTimer); document.title = base; }
    }, 700);
  }

  /* ── 알림 감시 ── */
  const known = new Map();      // id → {level, updated_at, stage}
  let baseline = false;
  const alertSubs = [];
  let lastAlerts = [];

  function notifyFor(a, prev) {
    const first = a.lines[0] ? `${a.lines[0].ko} · ${a.lines[0].text}` : "";
    if (!prev) {
      if (a.level === "위험") { alarm(a); }
      banner({ level: a.level, title: `[${a.level}] ${a.title}`, text: first, id: a.id, kind: a.level === "안전" ? "안심 알림" : "보안 알림" });
      playSound(a.level); vibrate(a.level); flashTitle(a.level);
      return;
    }
    if (LEVEL_RANK[a.level] > LEVEL_RANK[prev.level]) {
      if (a.level === "위험") alarm(a);
      banner({ level: a.level, title: `[${a.level}로 상향] ${a.title}`, text: a.lines.map((l) => l.ko).filter(Boolean).join(" + ") + " 교차 확인", id: a.id, kind: "알림 갱신" });
      playSound(a.level); vibrate(a.level); flashTitle(a.level);
    } else if (a.stage === "final" && prev.stage !== "final") {
      banner({ level: a.level, title: `종합 판정 완료 · ${a.level}`, text: a.title, id: a.id, kind: "AI 판정" });
    }
  }

  async function pollAlerts() {
    try {
      const { data } = await api("/api/alerts?limit=30");
      if (!Array.isArray(data)) return;
      const list = data.map(parseAlert);
      for (const a of list) {
        const prev = known.get(a.id);
        if (baseline && (!prev || prev.updated_at !== a.updated_at)) notifyFor(a, prev);
        known.set(a.id, { level: a.level, updated_at: a.updated_at, stage: a.stage });
      }
      baseline = true;
      lastAlerts = list;
      alertSubs.forEach((cb) => cb(list));
    } catch (e) { /* 서버 재시작 중 등 → 다음 주기에 다시 */ }
  }

  /* ── 상태 감시 (구독한 화면만) ── */
  const statusSubs = [];
  let lastStatus = null;
  async function pollStatus() {
    if (!statusSubs.length) return;
    try {
      const { data } = await api("/api/status");
      if (!data || data.ok === false) return;
      lastStatus = data;
      statusSubs.forEach((cb) => cb(data));
    } catch {}
  }

  if (isAuth) {
    pollAlerts();
    setInterval(pollAlerts, 1500);
    setInterval(pollStatus, 1200);
  }

  /* ── 폰 밖 패널 접기 ── */
  if (document.querySelector(".side")) {
    if (localStorage.getItem("sh_hide_side") === "1") document.body.classList.add("hide-side");
    const t = document.createElement("button");
    t.className = "side-toggle";
    const label = () => (t.textContent = document.body.classList.contains("hide-side") ? "시연 패널 보기" : "시연 패널 숨기기");
    label();
    t.onclick = () => {
      document.body.classList.toggle("hide-side");
      localStorage.setItem("sh_hide_side", document.body.classList.contains("hide-side") ? "1" : "0");
      label();
    };
    document.body.appendChild(t);
  }

  window.SH = {
    api, toast, pref, parseAlert, timeAgo, hhmm, SRC, LEVEL_RANK, ack, isAcked, playSound,
    onAlerts(cb) { alertSubs.push(cb); if (baseline) cb(lastAlerts); },
    onStatus(cb) { statusSubs.push(cb); if (lastStatus) cb(lastStatus); else pollStatus(); },
    refreshAlerts: pollAlerts,
    resetKnown() { known.clear(); baseline = false; pollAlerts(); },
    updateUnlock,
  };
})();
