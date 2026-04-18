// Attestor — Recall Pipeline Replay
// Renders the 5-layer retrieval cascade step by step.

(function () {
  "use strict";

  const LAYER_ICONS = ["#", "◇", "⬡", "⚡", "◎"];
  const LAYER_COLORS = [
    "var(--brass)",      // Tag Match
    "var(--verdigris)",  // Graph Expansion
    "var(--indigo)",     // Vector Search
    "var(--rust)",       // Fusion + Rank
    "var(--paper)",      // Diversity + Fit
  ];

  function esc(str) {
    const d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }

  async function executeRecall() {
    const query = document.getElementById("recall-query").value.trim();
    if (!query) return;

    const ns = document.getElementById("recall-ns").value.trim() || null;
    const budget = parseInt(document.getElementById("recall-budget").value) || 2000;
    const area = document.getElementById("recall-area");

    area.innerHTML = '<div class="recall-loading">Executing pipeline…</div>';

    let data;
    try {
      const res = await fetch("/ui/recall.json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, namespace: ns, budget }),
      });
      data = await res.json();
      if (data.error) {
        area.innerHTML = `<div class="recall-empty">${esc(data.error)}</div>`;
        return;
      }
    } catch (e) {
      area.innerHTML = `<div class="recall-empty">Request failed: ${esc(e.message)}</div>`;
      return;
    }

    renderTrace(data, area);
    renderConfig(data.config);
  }

  function renderPipelineDiagram(data) {
    const layers = data.layers || [];
    if (layers.length === 0) return "";

    const LAYER_SHORT = ["Tag", "Graph", "Vector", "Fused", "Final"];

    let html = `<div class="recall-diagram">`;
    layers.forEach((layer, i) => {
      const color = LAYER_COLORS[i] || "var(--ghost)";
      const stepDelay = i * 120;
      const arrowDelay = i * 120 + 60;

      if (i > 0) {
        html += `<div class="recall-diagram__arrow" style="animation-delay:${arrowDelay}ms">&rarr;</div>`;
      }

      const msLabel = layer.latency_ms != null
        ? `<div class="recall-diagram__ms">${parseFloat(layer.latency_ms).toFixed(1)} ms</div>`
        : "";

      html += `
        <div class="recall-diagram__step" style="--layer-color:${color}; animation-delay:${stepDelay}ms">
          <div class="recall-diagram__num">Layer ${i + 1}</div>
          <div class="recall-diagram__name">${esc(layer.name)}</div>
          <div class="recall-diagram__count">${layer.count}</div>
          <div class="recall-diagram__count-label">results</div>
          ${msLabel}
        </div>
      `;
    });

    // Summary line: total input candidates -> final output + total latency
    const firstCount = layers.length > 0 ? layers[0].count : 0;
    const lastCount = layers.length > 0 ? layers[layers.length - 1].count : 0;
    const totalMs = data.total_latency_ms != null
      ? ` in <span class="recall-diagram__summary-count">${parseFloat(data.total_latency_ms).toFixed(1)} ms</span>`
      : "";
    html += `
      <div class="recall-diagram__summary">
        <span class="recall-diagram__summary-count">${firstCount}</span> candidates entered
        &rarr;
        <span class="recall-diagram__summary-count">${lastCount}</span> results delivered${totalMs}
      </div>
    `;

    // Funnel metric: Tag: N -> Graph: N -> Vector: N -> Fused: N -> Final: N
    html += `<div class="recall-funnel">`;
    layers.forEach((layer, i) => {
      if (i > 0) {
        html += `<span class="recall-funnel__arrow">&rarr;</span>`;
      }
      const shortName = LAYER_SHORT[i] || layer.name;
      html += `
        <span class="recall-funnel__step">
          <span class="recall-funnel__step-name">${esc(shortName)}</span>
          <span class="recall-funnel__step-count">${layer.count}</span>
        </span>
      `;
    });
    html += `</div>`;

    html += `</div>`;
    return html;
  }

  function renderTrace(data, container) {
    let html = `
      <div class="recall-header">
        <div class="recall-header__query">"${esc(data.query)}"</div>
        <div class="recall-header__meta">
          ${data.namespace ? `namespace: ${esc(data.namespace)} · ` : ""}budget: ${data.token_budget} tokens · ${data.final_count} results
        </div>
      </div>
    `;

    html += renderPipelineDiagram(data);

    html += `<div class="recall-pipeline">
    `;

    data.layers.forEach((layer, i) => {
      const color = LAYER_COLORS[i] || "var(--ghost)";
      const icon = LAYER_ICONS[i] || "·";
      const isLast = i === data.layers.length - 1;

      html += `
        <div class="recall-layer" style="--layer-color: ${color}; animation-delay: ${i * 150}ms">
          <div class="recall-layer__head">
            <span class="recall-layer__icon">${icon}</span>
            <div class="recall-layer__info">
              <div class="recall-layer__name">Layer ${i + 1} · ${esc(layer.name)}</div>
              <div class="recall-layer__desc">${esc(layer.description)}</div>
            </div>
            ${layer.latency_ms != null ? `<div class="recall-layer__ms">${parseFloat(layer.latency_ms).toFixed(1)} ms</div>` : ""}
            <div class="recall-layer__count">${layer.count}</div>
          </div>
      `;

      // Layer-specific metadata
      if (layer.tags && layer.tags.length > 0) {
        html += `<div class="recall-layer__meta">Tags extracted: ${layer.tags.map((t) => `<span class="recall-tag">${esc(t)}</span>`).join(" ")}</div>`;
      }
      if (layer.expanded_entities && layer.expanded_entities.length > 0) {
        html += `<div class="recall-layer__meta">Expanded entities: ${layer.expanded_entities.map((e) => `<span class="recall-tag">${esc(e)}</span>`).join(" ")}</div>`;
      }
      if (layer.fusion_mode) {
        html += `<div class="recall-layer__meta">Mode: ${esc(layer.fusion_mode)}</div>`;
      }
      if (layer.mmr_enabled !== undefined) {
        html += `<div class="recall-layer__meta">MMR: ${layer.mmr_enabled ? "on" : "off"} · Budget: ${layer.budget}</div>`;
      }

      // Results
      if (layer.results && layer.results.length > 0) {
        html += `<div class="recall-layer__results">`;
        layer.results.forEach((r, ri) => {
          const barWidth = Math.max(4, Math.min(100, r.score * 100));
          html += `
            <div class="recall-result ${isLast ? "recall-result--final" : ""}">
              <div class="recall-result__rank">${ri + 1}</div>
              <div class="recall-result__body">
                <div class="recall-result__content">${esc(r.content)}</div>
                <div class="recall-result__meta">
                  <span class="recall-result__source">${esc(r.source)}</span>
                  ${r.entity ? `<span>· ${esc(r.entity)}</span>` : ""}
                  ${r.category ? `<span>· ${esc(r.category)}</span>` : ""}
                </div>
              </div>
              <div class="recall-result__score">
                <div class="recall-result__bar" style="width:${barWidth}%"></div>
                <span>${r.score.toFixed(4)}</span>
              </div>
            </div>
          `;
        });
        html += `</div>`;
      } else {
        html += `<div class="recall-layer__empty">No results from this layer</div>`;
      }

      html += `</div>`; // close recall-layer

      // Connector arrow between layers
      if (!isLast) {
        html += `<div class="recall-connector"><span>▼</span></div>`;
      }
    });

    html += `</div>`; // close recall-pipeline
    container.innerHTML = html;
  }

  function renderConfig(config) {
    if (!config) return;
    const el = document.getElementById("recall-config");
    let html = `<div class="rail__label">Config</div>`;
    html += `<div class="recall-config-list">`;
    for (const [k, v] of Object.entries(config)) {
      html += `<div class="recall-config-item"><span>${esc(k)}</span> ${esc(String(v))}</div>`;
    }
    html += `</div>`;
    el.innerHTML = html;
  }

  // Wire up
  document.getElementById("recall-form").addEventListener("submit", (e) => {
    e.preventDefault();
    executeRecall();
  });

  document.getElementById("recall-go").addEventListener("click", executeRecall);

  // Enter key in query field
  document.getElementById("recall-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      executeRecall();
    }
  });

  // Budget-vs-latency explorer
  async function budgetExplore() {
    const query = document.getElementById("recall-query").value.trim();
    if (!query) return;
    const ns = document.getElementById("recall-ns").value.trim() || "";
    const btn = document.getElementById("budget-explore-btn");
    const container = document.getElementById("budget-explore-area");
    if (!container) return;

    btn.disabled = true;
    btn.textContent = "Running\u2026";
    container.innerHTML = '<div class="recall-loading">Running 5 budget variants\u2026</div>';

    try {
      const params = new URLSearchParams({ q: query });
      if (ns) params.set("namespace", ns);
      const res = await fetch("/ui/recall/budget-explore.json?" + params);
      const data = await res.json();
      if (data.error) {
        container.innerHTML = `<div class="recall-empty">${esc(data.error)}</div>`;
        return;
      }
      renderBudgetChart(data, container);
    } catch (e) {
      container.innerHTML = `<div class="recall-empty">Error: ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Budget Explorer";
    }
  }

  function renderBudgetChart(data, container) {
    const budgets = data.budgets || [];
    if (!budgets.length) { container.innerHTML = ""; return; }

    const maxMs = Math.max.apply(null, budgets.map(b => b.latency_ms)) || 1;
    const maxCount = Math.max.apply(null, budgets.map(b => b.result_count)) || 1;

    let html = '<div class="budget-chart">';
    html += '<div class="budget-chart__title">Budget vs Latency &amp; Results</div>';
    html += '<div class="budget-chart__bars">';

    budgets.forEach((b) => {
      const msPct = Math.min(100, (b.latency_ms / maxMs) * 100);
      const countPct = Math.min(100, (b.result_count / maxCount) * 100);
      const msColor = b.latency_ms < 10 ? "var(--verdigris)" : b.latency_ms < 50 ? "var(--brass)" : "var(--rust)";
      html += `
        <div class="budget-row">
          <div class="budget-row__label">${b.budget.toLocaleString()}</div>
          <div class="budget-row__bars">
            <div class="budget-row__bar budget-row__bar--ms" style="width:${msPct}%;background:${msColor}">
              <span>${b.latency_ms.toFixed(1)} ms</span>
            </div>
            <div class="budget-row__bar budget-row__bar--count" style="width:${countPct}%">
              <span>${b.result_count} results</span>
            </div>
          </div>
        </div>
      `;
    });

    html += '</div></div>';
    container.innerHTML = html;
  }

  // Wire up budget explore button if present
  const budgetBtn = document.getElementById("budget-explore-btn");
  if (budgetBtn) {
    budgetBtn.addEventListener("click", budgetExplore);
  }
})();
