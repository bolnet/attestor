/* ops.js — Operations Log flight recorder with auto-refresh. */

(function () {
  "use strict";

  var REFRESH_MS = 5000;
  var area = document.getElementById("ops-area");

  function esc(s) {
    if (s == null) return "";
    var d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function fmtMs(v) {
    if (v == null) return "\u2014";
    var n = parseFloat(v);
    if (n < 1) return n.toFixed(3) + " ms";
    if (n < 100) return n.toFixed(1) + " ms";
    return Math.round(n) + " ms";
  }

  function fmtTs(iso) {
    if (!iso) return "\u2014";
    try {
      var d = new Date(iso);
      return d.toLocaleTimeString(undefined, { hour12: false }) + "." +
        String(d.getMilliseconds()).padStart(3, "0");
    } catch (_) {
      return iso;
    }
  }

  /* Latency bar — max 200ms = full width, clamped */
  function latencyBar(ms) {
    var pct = Math.min(100, (ms / 200) * 100);
    var color = ms < 10 ? "var(--verdigris)" : ms < 50 ? "var(--brass)" : "var(--rust)";
    return '<div class="ops-bar" style="width:' + pct + '%;background:' + color + '"></div>';
  }

  /* Sparkline SVG from latency array */
  function sparklineSvg(latencies, width, height) {
    if (!latencies || latencies.length < 2) return "";
    var max = Math.max.apply(null, latencies) || 1;
    var step = width / (latencies.length - 1);
    var points = latencies.map(function (v, i) {
      return (i * step).toFixed(1) + "," + (height - (v / max) * (height - 2)).toFixed(1);
    }).join(" ");
    return '<svg class="ops-spark" viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none">' +
      '<polyline points="' + points + '" fill="none" stroke="var(--rust)" stroke-width="1.5"/>' +
      '</svg>';
  }

  function render(data) {
    var ops = data.ops || [];

    if (ops.length === 0) {
      area.innerHTML = '<div class="ops-empty">No operations recorded yet. Add memories or run recalls to populate the flight recorder.</div>';
      return;
    }

    /* Summary stats */
    var recallOps = ops.filter(function (o) { return o.op === "recall"; });
    var addOps = ops.filter(function (o) { return o.op === "add"; });
    var latencies = ops.map(function (o) { return o.latency_ms; });
    var sorted = latencies.slice().sort(function (a, b) { return a - b; });
    var n = sorted.length;
    var p50 = sorted[Math.floor(n * 0.5)] || 0;
    var p95 = sorted[Math.floor(n * 0.95)] || 0;
    var p99 = sorted[Math.floor(n * 0.99)] || 0;

    var html = "";

    /* Summary row */
    html += '<div class="ops-summary">';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + ops.length + '</div><div class="ops-stat__label">Total Ops</div></div>';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + recallOps.length + '</div><div class="ops-stat__label">Recalls</div></div>';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + addOps.length + '</div><div class="ops-stat__label">Adds</div></div>';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + fmtMs(p50) + '</div><div class="ops-stat__label">P50</div></div>';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + fmtMs(p95) + '</div><div class="ops-stat__label">P95</div></div>';
    html += '<div class="ops-stat"><div class="ops-stat__val">' + fmtMs(p99) + '</div><div class="ops-stat__label">P99</div></div>';
    html += '</div>';

    /* Sparkline */
    var last50 = latencies.slice(-50);
    if (last50.length >= 2) {
      html += '<div class="ops-sparkline-row">';
      html += '<span class="ops-sparkline-label">Last ' + last50.length + ' ops</span>';
      html += sparklineSvg(last50, 600, 40);
      html += '</div>';
    }

    /* Table */
    html += '<div class="ops-table-wrap">';
    html += '<table class="ops-table">';
    html += '<thead><tr><th>Time</th><th>Op</th><th>Latency</th><th></th><th>Input</th><th>Results</th><th>Stores</th></tr></thead>';
    html += '<tbody>';

    /* Show most recent first */
    for (var i = ops.length - 1; i >= 0; i--) {
      var op = ops[i];
      var opClass = "ops-op--" + (op.op || "unknown");
      html += '<tr>';
      html += '<td class="ops-td--ts">' + esc(fmtTs(op.ts)) + '</td>';
      html += '<td><span class="ops-badge ' + opClass + '">' + esc(op.op) + '</span></td>';
      html += '<td class="ops-td--ms">' + fmtMs(op.latency_ms) + '</td>';
      html += '<td class="ops-td--bar">' + latencyBar(op.latency_ms) + '</td>';
      html += '<td class="ops-td--input">' + esc(op.input || "") + '</td>';
      html += '<td>' + (op.result_count != null ? op.result_count : "\u2014") + '</td>';
      html += '<td class="ops-td--stores">' + (op.stores || []).join(", ") + '</td>';
      html += '</tr>';
    }

    html += '</tbody></table></div>';
    html += '<div class="ops-refresh-note">Auto-refreshes every 5 seconds</div>';

    area.innerHTML = html;
  }

  function poll() {
    fetch("/ui/ops.json")
      .then(function (r) { return r.json(); })
      .then(function (data) { render(data); })
      .catch(function (err) {
        area.innerHTML = '<div class="ops-empty">Fetch error: ' + esc(err.message) + '</div>';
      });
  }

  poll();
  setInterval(poll, REFRESH_MS);
})();
