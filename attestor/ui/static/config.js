// Attestor — Configuration Viewer
// Fetches /ui/config.json and renders system parameters in the Forensic Archive style.

(function () {
  "use strict";

  // ---- Helpers ----

  function esc(str) {
    var d = document.createElement("div");
    d.textContent = str == null ? "" : String(str);
    return d.innerHTML;
  }

  function fmtBytes(bytes) {
    if (bytes == null) return "\u2014";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function kvRow(key, value) {
    return (
      '<dt>' + esc(key) + '</dt>' +
      '<dd>' + (value != null ? esc(String(value)) : '<span style="color:var(--ghost)">\u2014</span>') + '</dd>'
    );
  }

  function kvRowCode(key, value) {
    return (
      '<dt>' + esc(key) + '</dt>' +
      '<dd><code>' + esc(String(value)) + '</code></dd>'
    );
  }

  function sectionHeader(label) {
    return '<div class="cfg-section-label">' + esc(label) + '</div>';
  }

  function statusDot(ok) {
    if (ok) return '<span class="cfg-dot cfg-dot--ok"></span>';
    return '<span class="cfg-dot cfg-dot--err"></span>';
  }

  // ---- Render ----

  function render(data) {
    var area = document.getElementById("cfg-area");
    var html = "";

    // -- Storage Backends --
    html += sectionHeader("Storage Backends");
    html += '<dl class="kv cfg-kv">';

    var roles = data.role_assignments || {};
    html += kvRow("Document Store", roles.document || "\u2014");
    html += kvRow("Vector Store", roles.vector || "\u2014");
    html += kvRow("Graph Store", roles.graph || "\u2014");

    var backends = data.backends || [];
    html += kvRow("Backend List", backends.join(", ") || "\u2014");
    html += kvRowCode("Store Path", data.store_path || "\u2014");

    // Backend file paths
    var paths = data.store_paths || {};
    if (paths.db_path) html += kvRowCode("SQLite Path", paths.db_path);
    if (paths.graph_path) html += kvRowCode("Graph Path", paths.graph_path);
    if (paths.db_size_bytes != null) html += kvRow("SQLite Size", fmtBytes(paths.db_size_bytes));

    html += '</dl>';

    // -- Embedding Provider --
    html += sectionHeader("Embedding Provider");
    html += '<dl class="kv cfg-kv">';

    var emb = data.embedding || {};
    html += kvRow("Provider", emb.provider || "local");
    html += kvRow("Dimension", emb.dimension || "\u2014");
    html += kvRow("Vector Count", emb.vector_count != null ? emb.vector_count : "\u2014");

    html += '</dl>';

    // -- Retrieval Pipeline --
    html += sectionHeader("Retrieval Pipeline");
    html += '<dl class="kv cfg-kv">';

    var ret = data.retrieval || {};
    html += kvRow("Fusion Mode", ret.fusion_mode || "rrf");
    html += kvRow("RRF k", ret.rrf_k != null ? ret.rrf_k : 60);
    html += kvRow("MMR Enabled", ret.enable_mmr != null ? ret.enable_mmr : true);
    html += kvRowCode("MMR Lambda", ret.mmr_lambda != null ? ret.mmr_lambda : 0.7);
    html += kvRow("Graph BFS Depth", ret.graph_bfs_depth != null ? ret.graph_bfs_depth : 2);
    html += kvRow("Min Results", ret.min_results != null ? ret.min_results : 3);
    html += kvRow("Default Token Budget", ret.default_token_budget || "\u2014");
    html += kvRow("Confidence Gate", ret.confidence_gate != null ? ret.confidence_gate : 0.0);
    html += kvRowCode("Confidence Decay Rate", ret.confidence_decay_rate != null ? ret.confidence_decay_rate : 0.001);
    html += kvRowCode("Confidence Boost Rate", ret.confidence_boost_rate != null ? ret.confidence_boost_rate : 0.03);

    var layers = ret.active_layers || [];
    html += kvRow("Active Layers", layers.length + " / " + (ret.max_layers || 3));
    html += kvRow("Layer Names", layers.join(" \u2192 ") || "\u2014");

    html += '</dl>';

    // -- Store Statistics --
    html += sectionHeader("Store Statistics");
    html += '<dl class="kv cfg-kv">';

    var stats = data.stats || {};
    html += kvRow("Total Memories", stats.total_memories != null ? stats.total_memories : "\u2014");
    html += kvRow("Active", stats.active != null ? stats.active : stats.total_active || "\u2014");
    html += kvRow("Superseded", stats.superseded != null ? stats.superseded : "\u2014");

    var ns = stats.namespaces;
    if (Array.isArray(ns)) {
      html += kvRow("Namespaces", ns.join(", ") || "\u2014");
    } else if (ns != null) {
      html += kvRow("Namespaces", ns);
    }

    var categories = stats.categories;
    if (Array.isArray(categories)) {
      html += kvRow("Categories", categories.join(", ") || "\u2014");
    } else if (categories != null) {
      html += kvRow("Categories", categories);
    }

    // Graph stats
    var graph = data.graph_stats || {};
    if (graph.nodes != null) html += kvRow("Graph Nodes", graph.nodes);
    if (graph.edges != null) html += kvRow("Graph Edges", graph.edges);

    html += '</dl>';

    // -- Backend Configs (if any) --
    var bcfg = data.backend_configs || {};
    var bcfgKeys = Object.keys(bcfg);
    if (bcfgKeys.length > 0) {
      html += sectionHeader("Backend Configuration");
      html += '<dl class="kv cfg-kv">';
      bcfgKeys.forEach(function (key) {
        var val = bcfg[key];
        if (typeof val === "object" && val !== null) {
          html += kvRow(key, JSON.stringify(val));
        } else {
          html += kvRow(key, val);
        }
      });
      html += '</dl>';
    }

    area.innerHTML = html;
  }

  function renderHealth(data) {
    var statusEl = document.getElementById("cfg-health-status");
    var checksEl = document.getElementById("cfg-health-checks");

    var health = data.health || {};
    var checks = health.checks || [];
    var healthy = health.healthy !== false;

    statusEl.innerHTML = healthy
      ? '<span style="color:var(--verdigris)">OK</span>'
      : '<span style="color:var(--rust)">DEGRADED</span>';

    var html = '<div class="cfg-health-list">';
    checks.forEach(function (c) {
      var ok = c.status === "ok";
      html += '<div class="cfg-health-item">';
      html += statusDot(ok);
      html += '<span class="cfg-health-name">' + esc(c.name) + '</span>';
      if (!ok && c.error) {
        html += '<span class="cfg-health-err">' + esc(c.error) + '</span>';
      }
      html += '</div>';
    });
    html += '</div>';
    checksEl.innerHTML = html;
  }

  // ---- Fetch ----

  function loadConfig() {
    var area = document.getElementById("cfg-area");
    area.innerHTML = '<div class="cfg-loading">Loading configuration\u2026</div>';

    fetch("/ui/config.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        render(data);
        renderHealth(data);
      })
      .catch(function (err) {
        area.innerHTML =
          '<div class="cfg-loading" style="color:var(--rust)">Failed to load configuration: ' +
          esc(String(err)) + '</div>';
      });
  }

  // ---- Init ----

  document.addEventListener("DOMContentLoaded", function () {
    loadConfig();

    var btn = document.getElementById("cfg-refresh");
    if (btn) btn.addEventListener("click", loadConfig);
  });
})();
