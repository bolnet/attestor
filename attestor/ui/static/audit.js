// Attestor — audit dashboard front-end.
// Plain ES modules, no build step. Talks to /audit/* JSON endpoints.

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const tbody = $("#tbody");
  const empty = $("#empty-state");
  const filtersForm = $("#filters");
  const userInput = $("#f-user");
  const nsInput = $("#f-ns");
  const sinceInput = $("#f-since");
  const limitInput = $("#f-limit");
  const resetBtn = $("#reset-btn");

  const modal = $("#modal");
  const modalClose = $("#modal-close");
  const memModal = $("#mem-modal");
  const memClose = $("#mem-close");

  // ── Helpers ───────────────────────────────────────────────────────────
  function fmtMs(v) {
    if (v == null) return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    if (n < 1) return n.toFixed(2) + " ms";
    return n.toFixed(1) + " ms";
  }

  function fmtCost(v) {
    if (v == null) return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    if (n === 0) return "$0.00";
    if (n < 0.001) return "$" + n.toExponential(2);
    return "$" + n.toFixed(4);
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toISOString().replace("T", " ").replace("Z", "");
    } catch (e) {
      return iso;
    }
  }

  function escapeHTML(s) {
    if (s == null) return "";
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function lanePill(lane) {
    const cls = `lane-pill lane-${escapeHTML(lane)}` ||
      "lane-pill lane-other";
    const known = ["vector", "bm25", "graph", "reranker", "mmr"];
    const finalCls = known.includes(lane)
      ? `lane-pill lane-${lane}`
      : "lane-pill lane-other";
    return `<span class="${finalCls}">${escapeHTML(lane)}</span>`;
  }

  // ── Data layer ────────────────────────────────────────────────────────
  async function fetchJSON(url) {
    const res = await fetch(url, { headers: { "accept": "application/json" } });
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText} on ${url}`);
    }
    const body = await res.json();
    if (body && body.ok === false) {
      throw new Error(body.error || "request failed");
    }
    return body.data;
  }

  function buildListURL() {
    const params = new URLSearchParams();
    const u = userInput.value.trim();
    const ns = nsInput.value.trim();
    const since = sinceInput.value.trim();
    const limit = limitInput.value.trim();
    if (u) params.set("user_id", u);
    if (ns) params.set("namespace", ns);
    if (since) params.set("since", since);
    if (limit) params.set("limit", limit);
    const qs = params.toString();
    return "/audit/recalls" + (qs ? "?" + qs : "");
  }

  // ── Render ────────────────────────────────────────────────────────────
  function renderRows(rows) {
    tbody.innerHTML = "";
    empty.classList.toggle("hidden", rows.length > 0);

    if (rows.length === 0) {
      $("#s-count").textContent = "0";
      $("#s-latency").textContent = "—";
      $("#s-cost").textContent = "—";
      $("#s-users").textContent = "0";
      return;
    }

    let totalLatency = 0;
    let totalCost = 0;
    const users = new Set();

    const frag = document.createDocumentFragment();
    for (const r of rows) {
      totalLatency += Number(r.elapsed_ms || 0);
      totalCost += Number(r.cost_usd || 0);
      if (r.user_id) users.add(r.user_id);

      const tr = document.createElement("tr");
      tr.className = "row";
      tr.dataset.recallId = r.recall_id;
      tr.innerHTML = `
        <td class="px-4 py-3 mono text-xs text-slate-600">${escapeHTML(fmtTime(r.started_at))}</td>
        <td class="px-4 py-3 text-sm">${escapeHTML(r.user_id || "—")}</td>
        <td class="px-4 py-3 text-sm">${escapeHTML(r.namespace || "—")}</td>
        <td class="px-4 py-3 text-sm max-w-md truncate" title="${escapeHTML(r.query)}">${escapeHTML(r.query || "—")}</td>
        <td class="px-4 py-3 text-sm">${(r.lanes || []).map(lanePill).join(" ")}</td>
        <td class="px-4 py-3 text-sm text-right mono">${r.result_count}</td>
        <td class="px-4 py-3 text-sm text-right mono">${escapeHTML(fmtMs(r.elapsed_ms))}</td>
        <td class="px-4 py-3 text-sm text-right mono">${escapeHTML(fmtCost(r.cost_usd))}</td>
      `;
      tr.addEventListener("click", () => openRecallDetail(r.recall_id));
      frag.appendChild(tr);
    }
    tbody.appendChild(frag);

    $("#s-count").textContent = String(rows.length);
    $("#s-latency").textContent = (totalLatency / rows.length).toFixed(1);
    $("#s-cost").textContent = totalCost.toFixed(4);
    $("#s-users").textContent = String(users.size);
  }

  async function refresh() {
    try {
      const data = await fetchJSON(buildListURL());
      renderRows(data);
    } catch (err) {
      tbody.innerHTML = "";
      empty.classList.remove("hidden");
      empty.textContent = `Error: ${err.message}`;
    }
  }

  // ── Detail modal ──────────────────────────────────────────────────────
  async function openRecallDetail(rid) {
    try {
      const detail = await fetchJSON(`/audit/recall/${encodeURIComponent(rid)}`);
      $("#m-rid").textContent = detail.recall_id;
      $("#m-query").textContent = detail.query;
      $("#m-user").textContent = detail.user_id || "—";
      $("#m-ns").textContent = detail.namespace || "—";
      $("#m-latency").textContent = fmtMs(detail.elapsed_ms);
      $("#m-cost").textContent = fmtCost(detail.cost_usd);

      const lanesEl = $("#m-lanes");
      lanesEl.innerHTML = (detail.lanes || []).map(lanePill).join(" ") || "—";

      const resultsEl = $("#m-results");
      const completeEv = (detail.events || []).find(
        (e) => e.event === "recall.complete" || e.event === "recall.done",
      ) || {};
      const ids = completeEv.result_ids || (completeEv.final_ids || []).map(
        (x) => (typeof x === "string" ? x : x.id),
      );
      const finalEntries = completeEv.final_ids || ids.map((id) => ({ id }));
      resultsEl.innerHTML = "";
      if (finalEntries.length === 0) {
        resultsEl.innerHTML = '<li class="px-3 py-2 text-sm text-slate-500">No memories returned.</li>';
      } else {
        for (const entry of finalEntries) {
          const id = entry.id || entry;
          const score = entry.score != null ? entry.score : null;
          const source = entry.source || entry.match_source || "—";
          const preview = entry.preview || "";
          const li = document.createElement("li");
          li.className = "px-3 py-2 text-sm flex items-center justify-between hover:bg-slate-50 cursor-pointer";
          li.innerHTML = `
            <div class="min-w-0">
              <div class="mono text-xs text-slate-500">${escapeHTML(id)}</div>
              <div class="text-slate-700 truncate" title="${escapeHTML(preview)}">${escapeHTML(preview || "(no preview)")}</div>
            </div>
            <div class="ml-3 flex items-center gap-3 text-xs text-slate-500">
              <span class="mono">${score != null ? Number(score).toFixed(4) : "—"}</span>
              <span>${escapeHTML(source)}</span>
            </div>`;
          li.addEventListener("click", (ev) => {
            ev.stopPropagation();
            openMemoryAccess(id);
          });
          resultsEl.appendChild(li);
        }
      }

      const eventsEl = $("#m-events");
      eventsEl.innerHTML = "";
      for (const ev of detail.events || []) {
        const block = document.createElement("div");
        block.className = "border border-slate-200 rounded-md overflow-hidden";
        const head = document.createElement("button");
        head.type = "button";
        head.className = "w-full text-left px-3 py-2 bg-slate-50 hover:bg-slate-100 flex items-center justify-between";
        head.innerHTML = `
          <div class="flex items-center gap-3">
            <span class="mono text-xs text-slate-500">#${ev.seq ?? "—"}</span>
            <span class="font-medium text-slate-800">${escapeHTML(ev.event || "(unknown)")}</span>
          </div>
          <span class="mono text-xs text-slate-400">${escapeHTML(fmtTime(ev.ts))}</span>`;
        const body = document.createElement("pre");
        body.className = "evjson hidden whitespace-pre-wrap p-3 bg-white text-slate-700 mono";
        body.textContent = JSON.stringify(ev, null, 2);
        head.addEventListener("click", () => body.classList.toggle("hidden"));
        block.appendChild(head);
        block.appendChild(body);
        eventsEl.appendChild(block);
      }

      modal.classList.remove("hidden");
    } catch (err) {
      alert("Failed to load recall: " + err.message);
    }
  }

  function closeModal() {
    modal.classList.add("hidden");
  }

  async function openMemoryAccess(memoryId) {
    try {
      const rows = await fetchJSON(`/audit/memory/${encodeURIComponent(memoryId)}/access`);
      $("#mm-id").textContent = memoryId;
      const ul = $("#mm-rows");
      ul.innerHTML = "";
      if (!rows.length) {
        ul.innerHTML = '<li class="px-6 py-4 text-sm text-slate-500">No access events recorded.</li>';
      } else {
        for (const r of rows) {
          const li = document.createElement("li");
          li.className = "px-6 py-3 text-sm";
          li.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="mono text-xs text-slate-500">${escapeHTML(fmtTime(r.ts))}</span>
              <span class="text-xs text-slate-400">${escapeHTML(r.recall_id)}</span>
            </div>
            <div class="mt-1 text-slate-700">${escapeHTML(r.query || "—")}</div>
            <div class="mt-1 text-xs text-slate-500">user=${escapeHTML(r.user_id || "—")} · ns=${escapeHTML(r.namespace || "—")}</div>`;
          ul.appendChild(li);
        }
      }
      memModal.classList.remove("hidden");
    } catch (err) {
      alert("Failed to load memory access: " + err.message);
    }
  }

  function closeMemModal() {
    memModal.classList.add("hidden");
  }

  // ── Wire it up ────────────────────────────────────────────────────────
  filtersForm.addEventListener("submit", (e) => {
    e.preventDefault();
    refresh();
  });

  resetBtn.addEventListener("click", () => {
    userInput.value = "";
    nsInput.value = "";
    sinceInput.value = "";
    limitInput.value = "100";
    refresh();
  });

  modalClose.addEventListener("click", closeModal);
  memClose.addEventListener("click", closeMemModal);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
  memModal.addEventListener("click", (e) => { if (e.target === memModal) closeMemModal(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!memModal.classList.contains("hidden")) closeMemModal();
      else if (!modal.classList.contains("hidden")) closeModal();
    }
  });

  // Initial paint.
  refresh();
})();
