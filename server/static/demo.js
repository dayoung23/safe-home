/* 폰 바깥 패널: 시연 시나리오 · 개별 이벤트 · 실시간 판단 로그 · 판단 흐름 하이라이트 */
(() => {
  const $ = (id) => document.getElementById(id);
  const SRC_KO = { sound: "소리", vision: "영상", keypad: "도어락", judge: "판정" };
  const ACT_KO = { record: "기록만", save: "새 알림", update: "알림 갱신", wait_keypad: "도어락 대기", final: "종합 판정" };
  const TYPE_KO = {
    person: "사람 감지", entry: "현관 진입", threat: "양손 든 자세", fall: "쓰러짐", none: "사람 없음", motion: "움직임",
    scream: "비명", glass_break: "유리 깨짐", door_slam: "문 쾅", bang: "쾅", thump: "충격음", shout: "고함", crying: "울음",
    keypad_input: "정상 인증", keypad_anomaly: "리듬 불일치", keypad_failed: "비밀번호 오류", keypad_slow_input: "입력 지연",
    keypad_register: "패턴 등록", door_locked: "잠금", final: "",
  };
  const esc = (s) => { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; };

  /* ── 시나리오 / 이벤트 실행 ── */
  async function run(url, body, label) {
    const { data } = await SH.api(url, { json: body });
    if (data && data.ok) SH.toast(`▶ ${label}`, 1600);
    else SH.toast((data && data.error) || "실행 실패");
  }
  $("scenarios")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-scn]"); if (!b) return;
    run("/api/demo/scenario", { name: b.dataset.scn }, b.querySelector("b").textContent);
  });
  $("events")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-ev]"); if (!b) return;
    run("/api/demo/event", { kind: b.dataset.ev }, b.textContent);
  });
  $("clearBtn")?.addEventListener("click", async () => {
    if (!confirm("알림과 판단 기록을 모두 지울까요? (시연 초기화)")) return;
    const { data } = await SH.api("/api/demo/clear", { json: {} });
    if (data && data.ok) { SH.toast(`알림 ${data.deleted}개를 지웠어요`); SH.resetKnown(); }
    else SH.toast((data && data.error) || "초기화 실패");
  });

  /* ── 판단 흐름 하이라이트 ── */
  const flowTimers = [];
  function light(steps) {
    flowTimers.splice(0).forEach(clearTimeout);
    document.querySelectorAll(".flow-step").forEach((el) => el.classList.remove("lit"));
    steps.forEach((n, i) => flowTimers.push(setTimeout(() => {
      document.querySelectorAll(".flow-step").forEach((el) => el.classList.toggle("lit", el.dataset.step === String(n)));
    }, i * 650)));
    flowTimers.push(setTimeout(() => document.querySelectorAll(".flow-step").forEach((el) => el.classList.remove("lit")), steps.length * 650 + 1800));
  }

  /* ── 실시간 로그 + 사건 창 ── */
  let lastTop = null;
  SH.onStatus((st) => {
    const log = $("log");
    if (log) {
      const rows = st.recent || [];
      log.innerHTML = rows.length ? rows.map((r) => `
        <div class="log-row">
          <span class="t">${r.time}</span>
          <span class="s s-${r.source}">${SRC_KO[r.source] || r.source}${r.demo ? "*" : ""}</span>
          <span class="d">${esc(TYPE_KO[r.type] ?? r.type)}${r.type === "final" ? esc(r.message) : ""}<span class="act ${r.action} lv-${r.level || "none"}">${ACT_KO[r.action] || r.action}${r.level ? " · " + r.level : ""}</span></span>
        </div>`).join("") : '<div class="log-empty">아직 들어온 이벤트가 없어요</div>';

      const top = rows[0];
      const key = top ? `${top.t}:${top.action}` : null;
      if (top && key !== lastTop && lastTop !== null) {
        if (top.action === "final") light([4]);
        else if (top.action === "save" || top.action === "update") light(top.source === "vision" ? [1, 2, 4] : [1, 2, 3, 4]);
        else light([1, 2]);
      }
      lastTop = key ?? lastTop ?? "";
    }
    const inc = (st.incidents || [])[0];
    const box = $("incident");
    if (box) {
      box.classList.toggle("on", !!inc);
      if (inc) {
        const w = st.window_sec || 5;
        $("incidentText").textContent = inc.final ? `사건 진행 중 · ${inc.level}` : `사건 창 ${Math.min(inc.age, w).toFixed(1)}s / ${w}s`;
        $("incidentBar").style.width = `${Math.min(100, (inc.age / w) * 100)}%`;
      }
    }
  });
})();
