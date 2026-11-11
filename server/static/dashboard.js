/* 홈 화면: 인사 · 집 상태 · 카메라 · 센서 · 알림 목록 · 알림 상세 */
(() => {
  const $ = (id) => document.getElementById(id);
  const TYPE_KO = {
    person: "사람 감지", entry: "현관 진입", threat: "위협 자세", fall: "쓰러짐", none: "이상 없음", motion: "움직임",
    suspicious: "수상한 상황", scream: "비명", glass_break: "유리 깨짐", door_slam: "문 쾅", bang: "쾅 소리",
    thump: "충격음", shout: "고함", crying: "울음", keypad_input: "정상 인증", keypad_anomaly: "리듬 불일치",
    keypad_failed: "비밀번호 오류", keypad_slow_input: "입력 지연", keypad_register: "패턴 등록", door_locked: "잠금",
  };
  const LV_ICON = { "위험": "i-alert", "주의": "i-alert", "안전": "i-shield-check" };
  let alerts = [];
  let filter = "all";

  /* ── 내 정보 ── */
  SH.api("/api/me").then(({ data }) => {
    if (!data || !data.name) return;
    $("greet").textContent = `${data.name}님, 안녕하세요`;
  });

  /* ── 알림 목록 ── */
  function srcChips(sources) {
    return (sources || []).map((s) => {
      const m = SH.SRC[s]; if (!m) return "";
      return `<span class="chip"><svg class="i"><use href="#${m.icon}"/></svg>${m.ko}</span>`;
    }).join("");
  }
  function stageTag(a) {
    if (a.stage === "checking") return `<span class="stage-tag checking">확인 중</span>`;
    if (a.stage === "final") return `<span class="stage-tag">종합 판정</span>`;
    return "";
  }
  function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

  const rendered = new Map();   // id → updated_at (갱신된 카드 강조용)
  let lastSig = "";
  function renderList(force) {
    const box = $("alertList");
    const sig = filter + "|" + alerts.map((a) => `${a.id}:${a.updated_at}:${SH.isAcked(a.id) ? 1 : 0}`).join(",");
    if (!force && sig === lastSig) return;          // 바뀐 게 없으면 다시 그리지 않음 (깜빡임 방지)
    lastSig = sig;
    const list = alerts.filter((a) => filter === "all" || a.level === filter);
    if (!list.length) {
      box.innerHTML = `<div class="empty"><svg class="i"><use href="#i-shield-check"/></svg><br>${filter === "all" ? "아직 알림이 없어요" : `${filter} 알림이 없어요`}<br><small>위험이 감지되면 바로 알려드릴게요</small></div>`;
      return;
    }
    box.innerHTML = list.map((a) => {
      const isNew = rendered.size > 0 && !rendered.has(a.id);
      const changed = rendered.has(a.id) && rendered.get(a.id) !== a.updated_at;
      return `<button class="alert lv-${a.level} ${SH.isAcked(a.id) ? "acked" : ""} ${changed ? "updated" : ""} ${isNew ? "new" : ""}" data-id="${a.id}">
        <div class="lv-ic"><svg class="i"><use href="#${LV_ICON[a.level] || "i-bell"}"/></svg></div>
        <div class="alert-body">
          <div class="alert-top"><span class="lv-tag">${a.level}</span>${stageTag(a)}<span class="alert-time">${SH.timeAgo(a.updated_at)}</span></div>
          <div class="alert-title">${esc(a.title)}</div>
          <div class="alert-src">${srcChips(a.sources)}${SH.isAcked(a.id) ? '<span class="chip">확인함</span>' : ""}</div>
        </div></button>`;
    }).join("");
    alerts.forEach((a) => rendered.set(a.id, a.updated_at));
  }
  $("alertList").addEventListener("click", (e) => {
    const card = e.target.closest(".alert"); if (card) openAlertDetail(card.dataset.id);
  });
  $("filters").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    filter = b.dataset.f;
    document.querySelectorAll("#filters button").forEach((x) => x.classList.toggle("on", x === b));
    renderList(true);
  });
  $("bellBtn").addEventListener("click", () => $("alertsTop").scrollIntoView({ behavior: "smooth" }));

  /* ── 집 상태 히어로: 최근 10분 안의 확인 안 한 알림 중 가장 높은 등급 ── */
  function unreadRecent() {
    return alerts.filter((a) => !SH.isAcked(a.id) && Date.now() - new Date(a.updated_at).getTime() < 3 * 60 * 1000 && a.level !== "안전");
  }
  function heroTop() {
    return unreadRecent().sort((x, y) => SH.LEVEL_RANK[y.level] - SH.LEVEL_RANK[x.level])[0];
  }
  function renderHero() {
    const recent = unreadRecent();
    const top = heroTop();
    const hero = $("hero");
    if (top && top.level === "위험") {
      hero.className = "hero danger"; $("heroIcon").innerHTML = '<use href="#i-alert"/>';
      $("heroTitle").textContent = "위험이 감지됐어요"; $("heroText").textContent = `${top.title} · 눌러서 확인`;
    } else if (top) {
      hero.className = "hero warn"; $("heroIcon").innerHTML = '<use href="#i-alert"/>';
      $("heroTitle").textContent = "확인이 필요해요"; $("heroText").textContent = `${top.title} · 눌러서 확인`;
    } else {
      hero.className = "hero safe"; $("heroIcon").innerHTML = '<use href="#i-shield-check"/>';
      $("heroTitle").textContent = "우리 집은 안전해요";
      const last = alerts[0];
      $("heroText").textContent = last ? `소리 · 영상 · 도어락이 지켜보고 있어요 · 마지막 알림 ${SH.timeAgo(last.updated_at)}` : "소리 · 영상 · 도어락이 지켜보고 있어요";
    }
    const unread = recent.length;
    $("bellDot").hidden = !unread; $("bellDot").textContent = unread;
  }

  document.addEventListener("sh:acked", () => { renderList(true); renderHero(); });
  $("hero").addEventListener("click", () => {
    const top = heroTop(); if (top) openAlertDetail(top.id);
  });

  SH.onAlerts((list) => {
    alerts = list;
    renderList(); renderHero();
    const want = new URLSearchParams(location.search).get("alert");
    if (want && !window.__openedFromUrl) { window.__openedFromUrl = true; openAlertDetail(want); }
  });

  /* ── 알림 상세 바텀시트 ── */
  let current = null;
  function openAlertDetail(id) {
    const a = alerts.find((x) => x.id === id);
    if (!a) return;
    current = a;
    $("sheetHead").className = `sheet-head lv-${a.level}`;
    $("shLevel").textContent = a.level;
    $("shStage").outerHTML = `<span class="stage-tag ${a.stage === "checking" ? "checking" : ""}" id="shStage">${a.stage === "checking" ? "확인 중" : a.stage === "final" ? "종합 판정 완료" : ""}</span>`;
    $("shTitle").textContent = a.title;
    const created = SH.hhmm(a.created_at), updated = SH.hhmm(a.updated_at);
    $("shTime").textContent = created === updated ? `${created} 감지` : `${created} 감지 · ${updated} 갱신`;
    $("shLines").innerHTML = a.lines.length ? a.lines.map((l) => {
      const m = SH.SRC[l.src];
      return `<div class="ev-line">${m ? `<span class="chip"><svg class="i"><use href="#${m.icon}"/></svg>${m.ko}</span>` : ""}<span>${esc(l.text)}</span></div>`;
    }).join("") : `<div class="ev-line">${esc(a.message)}</div>`;
    $("shAction").textContent = a.action || "특별히 할 일은 없어요. 기록으로 남겨둘게요.";
    $("shActions").querySelector('[data-act="call"]').style.display = a.level === "위험" ? "" : "none";
    $("shActions").querySelector('[data-act="share"]').classList.toggle("wide", a.level !== "위험");
    $("sheet").classList.add("open"); $("sheetBg").classList.add("open");
  }
  function closeSheet() { $("sheet").classList.remove("open"); $("sheetBg").classList.remove("open"); }
  $("sheetBg").addEventListener("click", closeSheet);
  $("shActions").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (!act || !current) return;
    if (act === "call") SH.toast("📞 112에 연결합니다 (시연 화면)");
    if (act === "share") SH.toast("보호자 2명에게 이 알림을 보냈어요 (시연)");
    if (act === "ack") { SH.ack(current.id); closeSheet(); renderList(true); renderHero(); SH.toast("확인 처리했어요"); }
  });
  window.openAlertDetail = openAlertDetail;

  /* ── 카메라 + 센서 (서버 상태) ── */
  let camUrl = "";
  function showImage(url) {
    if (url === camUrl) return;
    const img = new Image();
    img.onload = () => { camUrl = url; $("camImg").src = url; $("camImg").hidden = false; $("camEmpty").hidden = true; };
    img.src = url;
  }
  function renderCamera(cam) {
    const live = cam.live, cap = cam.capture;
    const badge = $("camBadge"), meta = $("camMeta");
    if (live && live.age < 6) {
      showImage(live.url);
      badge.hidden = false; badge.className = "cam-badge live"; badge.textContent = "LIVE";
      meta.hidden = false; $("camTime").textContent = SH.hhmm(new Date().toISOString());
      $("camState").textContent = "· 실시간 분석 중";
    } else if (cap) {
      showImage(cap.url);
      badge.hidden = false; badge.className = "cam-badge"; badge.textContent = "캡처 화면";
      meta.hidden = false; $("camTime").textContent = "";
      $("camState").textContent = "· 저장된 캡처";
    } else if (live) {
      showImage(live.url);
      badge.hidden = false; badge.className = "cam-badge"; badge.textContent = "연결 끊김";
      meta.hidden = false; $("camTime").textContent = `${Math.round(live.age)}초 전 화면`;
      $("camState").textContent = "· 카메라 신호 없음";
    } else {
      $("camImg").hidden = true; $("camEmpty").hidden = false; badge.hidden = true; meta.hidden = true; camUrl = "";
      $("camEmpty").innerHTML = `<svg class="i"><use href="#i-video"/></svg><br>카메라 화면이 아직 없어요<br><small>서버 카메라 → <code>static/camera/live.jpg</code><br>캡처 이미지 → <code>static/camera/capture.jpg</code></small>`;
      $("camState").textContent = "· 화면 없음";
    }
  }
  const lastSeen = {};
  function renderSensors(st) {
    for (const key of ["vision", "sound", "keypad"]) {
      const el = $(`sn-${key}`); const s = st.sensors[key];
      const span = el.querySelector("span");
      const camLive = key === "vision" && st.camera.live && st.camera.live.age < 6;
      if (s) {
        span.textContent = `${s.ago < 60 ? Math.round(s.ago) + "초" : Math.round(s.ago / 60) + "분"} 전 · ${TYPE_KO[s.type] || s.type}`;
        if (lastSeen[key] !== undefined && lastSeen[key] !== s.t) { el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
        lastSeen[key] = s.t;
      } else {
        span.textContent = camLive ? "실시간 분석 중" : "감지 대기 중";
      }
      el.classList.toggle("on", camLive || (s && s.ago < 120) || key !== "vision");
    }
  }
  SH.onStatus((st) => { renderCamera(st.camera || {}); renderSensors(st); });

  setInterval(() => { if (alerts.length) { renderList(true); renderHero(); } }, 15000);   // "n분 전" 갱신
})();
