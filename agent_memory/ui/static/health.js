/* health.js — Fetches /ui/health.json and renders the System Health dashboard. */

(function () {
  "use strict";

  const REFRESH_MS = 30_000;
  const area = document.getElementById("health-area");

  /* ---- Helpers ---- */
  function esc(s) {
    if (s == null) return "";
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function fmtBytes(b) {
    if (b == null) return "\u2014";
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / (1024 * 1024)).toFixed(1) + " MB";
  }

  function fmtMs(v) {
    if (v == null) return "\u2014";
    return parseFloat(v).toFixed(1) + " ms";
  }

  function fmtNum(v) {
    if (v == null) return "\u2014";
    return Number(v).toLocaleString();
  }

  /* Map check name to a human-friendly label + accent color variable */
  var COMPONENT_META = {
    "Document Store":     { icon: "D", color: "var(--brass)",     shortName: "SQLiteStore" },
    "Vector Store":       { icon: "V", color: "var(--verdigris)", shortName: "ChromaDB / Vector" },
    "Graph Store":        { icon: "G", color: "var(--indigo)",    shortName: "NetworkX / Graph" },
    "Retrieval Pipeline": { icon: "R", color: "var(--rust)",      shortName: "Retrieval Pipeline" },
  };

  /* ---- Build a single metric row ---- */
  function metricRow(label, value) {
    if (value == null || value === "") return "";
    return (
      '<div class="health-metric">' +
        '<span class="health-metric__label">' + esc(label) + '</span>' +
        '<span class="health-metric__value">' + esc(value) + '</span>' +
      '</div>'
    );
  }

  /* ---- Render one component card ---- */
  function renderCard(check, idx) {
    var meta = COMPONENT_META[check.name] || { icon: "?", color: "var(--ghost)", shortName: check.name };
    var ok = check.status === "ok";
    var dotClass = ok ? "health-dot--ok" : "health-dot--error";
    var borderColor = ok ? meta.color : "var(--rust)";

    var metrics = "";
    metrics += metricRow("Latency", check.latency_ms != null ? fmtMs(check.latency_ms) : null);
    metrics += metricRow("Records", check.memory_count != null ? fmtNum(check.memory_count) : null);
    metrics += metricRow("DB Size", check.db_size_bytes != null ? fmtBytes(check.db_size_bytes) : null);
    metrics += metricRow("Vectors", check.vector_count != null ? fmtNum(check.vector_count) : null);
    metrics += metricRow("Nodes", check.nodes != null ? fmtNum(check.nodes) : null);
    metrics += metricRow("Edges", check.edges != null ? fmtNum(check.edges) : null);
    metrics += metricRow("Layers", check.active_layers != null
      ? check.active_layers + " / " + (check.max_layers || "?")
      : null);
    metrics += metricRow("Embedding", check.embedding_provider);
    metrics += metricRow("Graph File", check.graph_file);

    /* Layer list (retrieval pipeline) */
    var layerList = "";
    if (check.layers && check.layers.length) {
      layerList = '<div class="health-layers">';
      for (var l = 0; l < check.layers.length; l++) {
        layerList += '<span class="health-layer-tag">' + esc(check.layers[l]) + '</span>';
      }
      layerList += '</div>';
    }

    var noteHtml = check.note
      ? '<div class="health-note">' + esc(check.note) + '</div>'
      : "";

    var errorHtml = check.error
      ? '<div class="health-error">' + esc(check.error) + '</div>'
      : "";

    return (
      '<div class="health-card" style="--layer-color:' + borderColor + ';animation-delay:' + (idx * 80) + 'ms">' +
        '<div class="health-card__head">' +
          '<div class="health-card__icon">' + esc(meta.icon) + '</div>' +
          '<div class="health-card__info">' +
            '<div class="health-card__name">' + esc(meta.shortName) + '</div>' +
            '<div class="health-card__check-name">' + esc(check.name) + '</div>' +
          '</div>' +
          '<div class="health-dot ' + dotClass + '"></div>' +
          '<div class="health-card__status">' + esc(check.status).toUpperCase() + '</div>' +
        '</div>' +
        (metrics ? '<div class="health-metrics">' + metrics + '</div>' : '') +
        layerList +
        noteHtml +
        errorHtml +
      '</div>'
    );
  }

  /* ---- Render the full dashboard ---- */
  function render(data) {
    var healthy = data.healthy;
    var checks = data.checks || [];

    var bannerClass = healthy ? "health-banner--ok" : "health-banner--error";
    var bannerText = healthy ? "ALL SYSTEMS OPERATIONAL" : "ISSUES DETECTED";
    var bannerDot = healthy
      ? '<span class="health-dot health-dot--ok"></span>'
      : '<span class="health-dot health-dot--error"></span>';

    var html = '';
    html += '<div class="health-banner ' + bannerClass + '">' + bannerDot + bannerText + '</div>';
    html += '<div class="health-grid">';
    for (var i = 0; i < checks.length; i++) {
      html += renderCard(checks[i], i);
    }
    if (checks.length === 0) {
      html += '<div class="health-empty">No diagnostic checks returned.</div>';
    }
    html += '</div>';
    html += '<div class="health-refresh-note">Auto-refreshes every 30 seconds</div>';

    area.innerHTML = html;
  }

  /* ---- Fetch loop ---- */
  function poll() {
    fetch("/ui/health.json")
      .then(function (r) { return r.json(); })
      .then(function (data) { render(data); })
      .catch(function (err) {
        area.innerHTML =
          '<div class="health-banner health-banner--error">' +
            '<span class="health-dot health-dot--error"></span>FETCH ERROR' +
          '</div>' +
          '<div class="health-error" style="margin:24px 40px;">' + esc(err.message) + '</div>';
      });
  }

  poll();
  setInterval(poll, REFRESH_MS);
})();
