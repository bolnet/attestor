// Memwright — Agent Registry
// Renders namespace-based agent dossier cards with drill-down detail.

(function () {
  "use strict";

  // ---- Helpers ----

  function esc(str) {
    var d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }

  function fmtDate(iso) {
    if (!iso) return "\u2014";
    return iso.slice(0, 10);
  }

  function fmtDateTime(iso) {
    if (!iso) return "\u2014";
    return iso.slice(0, 10) + " " + (iso.slice(11, 16) || "");
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n) + "\u2026" : s;
  }

  // ---- State ----

  var agentsData = null;

  // ---- Fetch ----

  function loadAgents() {
    var area = document.getElementById("agents-area");
    area.innerHTML = '<div class="agents-empty">Loading agent namespaces\u2026</div>';

    fetch("/ui/agents.json")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        agentsData = data;
        renderGrid(data.namespaces, area);
      })
      .catch(function (err) {
        area.innerHTML =
          '<div class="agents-empty">Failed to load: ' + esc(err.message) + "</div>";
      });
  }

  // ---- Grid ----

  function renderGrid(namespaces, container) {
    if (!namespaces || namespaces.length === 0) {
      container.innerHTML =
        '<div class="agents-empty">No agent namespaces found. Memories will appear here once agents write to distinct namespaces.</div>';
      return;
    }

    var html = '<div class="agents-grid">';

    namespaces.forEach(function (ns, i) {
      var catHtml = renderCategoryBar(ns.categories);
      var entitiesHtml = (ns.top_entities || [])
        .slice(0, 5)
        .map(function (e) {
          return '<span class="agents-entity">' + esc(e.name) + " <em>" + e.count + "</em></span>";
        })
        .join("");

      var activityHtml = renderActivity(ns.activity);

      html +=
        '<div class="agents-card" data-ns="' + esc(ns.namespace) + '" style="--i:' + i + '">' +
          '<div class="agents-card__tab">' + esc(ns.namespace) + "</div>" +
          '<div class="agents-card__name">' + esc(ns.namespace) + "</div>" +
          '<div class="agents-card__stat">' +
            '<span class="agents-card__count">' + ns.memory_count + "</span>" +
            '<span class="agents-card__label">memories</span>' +
          "</div>" +
          '<hr class="agents-card__rule">' +
          '<div class="agents-card__section-label">Categories</div>' +
          catHtml +
          '<div class="agents-card__section-label">Top Entities</div>' +
          '<div class="agents-card__entities">' +
            (entitiesHtml || '<span class="agents-card__none">none extracted</span>') +
          "</div>" +
          '<hr class="agents-card__rule">' +
          '<div class="agents-card__foot">' +
            '<div class="agents-card__latest">Latest: ' + fmtDate(ns.latest_date) + "</div>" +
            '<div class="agents-card__activity">' + activityHtml + "</div>" +
          "</div>" +
        "</div>";
    });

    html += "</div>";
    container.innerHTML = html;

    // Attach click listeners
    container.querySelectorAll(".agents-card").forEach(function (card) {
      card.addEventListener("click", function () {
        var nsName = card.getAttribute("data-ns");
        openDetail(nsName);
      });
    });
  }

  function renderCategoryBar(categories) {
    if (!categories || categories.length === 0) {
      return '<div class="agents-catbar"><span class="agents-card__none">uncategorized</span></div>';
    }

    var total = categories.reduce(function (sum, c) { return sum + c.count; }, 0);
    if (total === 0) return '<div class="agents-catbar"></div>';

    var COLORS = {
      general: "var(--ghost)",
      decision: "var(--rust)",
      fact: "var(--verdigris)",
      preference: "var(--brass)",
      procedure: "var(--indigo)",
      context: "var(--paper-shade)",
    };

    var barHtml = '<div class="agents-catbar__bar">';
    categories.forEach(function (c) {
      var pct = Math.max(2, (c.count / total) * 100);
      var color = COLORS[c.name] || "var(--ghost)";
      barHtml +=
        '<div class="agents-catbar__seg" style="width:' + pct + "%;background:" + color + '" title="' + esc(c.name) + ": " + c.count + '"></div>';
    });
    barHtml += "</div>";

    var legendHtml = '<div class="agents-catbar__legend">';
    categories.forEach(function (c) {
      var color = COLORS[c.name] || "var(--ghost)";
      legendHtml +=
        '<span class="agents-catbar__tag"><span class="agents-catbar__dot" style="background:' + color + '"></span>' + esc(c.name) + " " + c.count + "</span>";
    });
    legendHtml += "</div>";

    return '<div class="agents-catbar">' + barHtml + legendHtml + "</div>";
  }

  function renderActivity(activity) {
    if (!activity || activity.length === 0) return "";
    // Simple sparkline using Unicode block chars
    var max = Math.max.apply(null, activity.map(function (a) { return a.count; }));
    if (max === 0) return "";
    var BARS = ["\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"];
    return activity
      .map(function (a) {
        var idx = Math.round((a.count / max) * (BARS.length - 1));
        return '<span class="agents-spark" title="' + esc(a.date) + ": " + a.count + '">' + BARS[idx] + "</span>";
      })
      .join("");
  }

  // ---- Detail Panel ----

  function openDetail(nsName) {
    var overlay = document.getElementById("agents-overlay");
    var panel = document.getElementById("agents-detail");
    overlay.hidden = false;

    panel.innerHTML = '<div class="agents-empty">Loading dossier\u2026</div>';

    fetch("/ui/agents.json?namespace=" + encodeURIComponent(nsName) + "&detail=1")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        renderDetail(data, panel);
      })
      .catch(function (err) {
        panel.innerHTML =
          '<div class="agents-empty">Failed: ' + esc(err.message) + "</div>";
      });
  }

  function renderDetail(data, panel) {
    var ns = data.namespace_detail;
    if (!ns) {
      panel.innerHTML = '<div class="agents-empty">No data returned.</div>';
      return;
    }

    var html =
      '<button class="agents-detail__close" id="agents-detail-close">&times; Close</button>' +
      '<div class="agents-detail__head">' +
        '<div class="agents-detail__name">' + esc(ns.namespace) + "</div>" +
        '<div class="agents-detail__meta">' +
          ns.memory_count + " memories &middot; latest: " + fmtDate(ns.latest_date) +
        "</div>" +
      "</div>";

    // Entity frequency table
    if (ns.entity_freq && ns.entity_freq.length > 0) {
      html +=
        '<div class="agents-detail__section-label">Entity Frequency</div>' +
        '<table class="agents-freq">' +
          "<thead><tr><th>Entity</th><th>Count</th></tr></thead><tbody>";
      ns.entity_freq.forEach(function (e) {
        html += "<tr><td>" + esc(e.name) + "</td><td>" + e.count + "</td></tr>";
      });
      html += "</tbody></table>";
    }

    // Category distribution
    if (ns.categories && ns.categories.length > 0) {
      html += '<div class="agents-detail__section-label">Category Distribution</div>';
      html += renderCategoryBar(ns.categories);
    }

    // Recent memories (paginated, last 20)
    html += '<div class="agents-detail__section-label">Recent Memories</div>';
    if (ns.memories && ns.memories.length > 0) {
      html += '<div class="agents-detail__list">';
      ns.memories.forEach(function (m, i) {
        html +=
          '<div class="agents-mem">' +
            '<div class="agents-mem__rank">' + (i + 1) + "</div>" +
            '<div class="agents-mem__body">' +
              '<div class="agents-mem__content">' + esc(truncate(m.content, 200)) + "</div>" +
              '<div class="agents-mem__meta">' +
                '<span class="agents-mem__cat">' + esc(m.category || "general") + "</span>" +
                (m.entity ? " &middot; " + esc(m.entity) : "") +
                " &middot; " + fmtDateTime(m.created_at) +
              "</div>" +
            "</div>" +
            '<div class="agents-mem__conf">' + Math.round((m.confidence || 1) * 100) + "%</div>" +
          "</div>";
      });
      html += "</div>";

      // Pagination controls
      if (ns.total_count > ns.memories.length) {
        var page = ns.page || 1;
        var totalPages = Math.ceil(ns.total_count / 20);
        html +=
          '<div class="agents-detail__pager">' +
            (page > 1
              ? '<button class="btn agents-pager-btn" data-page="' + (page - 1) + '" data-ns="' + esc(ns.namespace) + '">Prev</button>'
              : "") +
            '<span class="agents-detail__pager-info">Page ' + page + " / " + totalPages + "</span>" +
            (page < totalPages
              ? '<button class="btn agents-pager-btn" data-page="' + (page + 1) + '" data-ns="' + esc(ns.namespace) + '">Next</button>'
              : "") +
          "</div>";
      }
    } else {
      html += '<div class="agents-card__none">No memories in this namespace.</div>';
    }

    panel.innerHTML = html;

    // Wire close button
    document.getElementById("agents-detail-close").addEventListener("click", closeDetail);

    // Wire pager buttons
    panel.querySelectorAll(".agents-pager-btn").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        var pg = btn.getAttribute("data-page");
        var nsName = btn.getAttribute("data-ns");
        loadDetailPage(nsName, parseInt(pg), panel);
      });
    });
  }

  function loadDetailPage(nsName, page, panel) {
    panel.innerHTML = '<div class="agents-empty">Loading page ' + page + "\u2026</div>";
    fetch("/ui/agents.json?namespace=" + encodeURIComponent(nsName) + "&detail=1&page=" + page)
      .then(function (res) { return res.json(); })
      .then(function (data) { renderDetail(data, panel); })
      .catch(function (err) {
        panel.innerHTML = '<div class="agents-empty">Failed: ' + esc(err.message) + "</div>";
      });
  }

  function closeDetail() {
    document.getElementById("agents-overlay").hidden = true;
  }

  // ---- Init ----

  document.getElementById("agents-refresh").addEventListener("click", loadAgents);

  // Close overlay on backdrop click
  document.getElementById("agents-overlay").addEventListener("click", function (e) {
    if (e.target === e.currentTarget) closeDetail();
  });

  // Close on Escape key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeDetail();
  });

  // Load on page ready
  loadAgents();
})();
