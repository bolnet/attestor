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
    "Document Store":     { icon: "D", color: "var(--brass)",     shortName: "Document Store" },
    "Vector Store":       { icon: "V", color: "var(--verdigris)", shortName: "Vector Store" },
    "Graph Store":        { icon: "G", color: "var(--indigo)",    shortName: "Graph Store" },
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

  /* ---- Sparkline SVG ---- */
  function sparklineSvg(latencies, width, height) {
    if (!latencies || latencies.length < 2) return "";
    var max = Math.max.apply(null, latencies) || 1;
    var step = width / (latencies.length - 1);
    var points = latencies.map(function (v, i) {
      return (i * step).toFixed(1) + "," + (height - (v / max) * (height - 2)).toFixed(1);
    }).join(" ");
    return '<svg class="health-spark" viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none">' +
      '<polyline points="' + points + '" fill="none" stroke="var(--rust)" stroke-width="1.5"/>' +
      '</svg>';
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

    /* Latency percentiles + sparkline (from ops ring buffer) */
    var pctl = data.latency_percentiles;
    var spark = data.latency_sparkline;
    if (pctl || (spark && spark.length >= 2)) {
      html += '<div class="health-latency-panel">';
      html += '<div class="health-latency-panel__title">Operation Latency</div>';
      if (pctl) {
        html += '<div class="health-pctl-row">';
        html += '<div class="health-pctl"><span class="health-pctl__val">' + fmtMs(pctl.p50) + '</span><span class="health-pctl__label">P50</span></div>';
        html += '<div class="health-pctl"><span class="health-pctl__val">' + fmtMs(pctl.p95) + '</span><span class="health-pctl__label">P95</span></div>';
        html += '<div class="health-pctl"><span class="health-pctl__val">' + fmtMs(pctl.p99) + '</span><span class="health-pctl__label">P99</span></div>';
        html += '<div class="health-pctl"><span class="health-pctl__val">' + fmtNum(data.ops_log_size) + '</span><span class="health-pctl__label">Ops</span></div>';
        html += '</div>';
      }
      if (spark && spark.length >= 2) {
        html += '<div class="health-spark-row">';
        html += '<span class="health-spark-label">Last ' + spark.length + ' ops</span>';
        html += sparklineSvg(spark, 500, 36);
        html += '</div>';
      }
      html += '</div>';
    }

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
