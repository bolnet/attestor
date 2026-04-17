// Memwright — Temporal Timeline
// Vertical timeline with supersession chains, as-of replay, and detail panel.

(function () {
  "use strict";

  // ---- Helpers ----

  function esc(str) {
    var d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }

  function excerpt(text, max) {
    if (!text) return "";
    return text.length > max ? text.slice(0, max) + "..." : text;
  }

  function fmtDate(iso) {
    if (!iso) return "";
    return iso.slice(0, 10);
  }

  function fmtTime(iso) {
    if (!iso || iso.length < 16) return "";
    return iso.slice(11, 16);
  }

  function fmtConfidence(val) {
    return Math.round((val || 0) * 100);
  }

  // ---- State ----

  var allMemories = [];

  // ---- Fetch ----

  function buildQueryString() {
    var params = new URLSearchParams();
    var ns = document.getElementById("tl-namespace").value.trim();
    var entity = document.getElementById("tl-entity").value.trim();
    var status = document.getElementById("tl-status").value;
    var from = document.getElementById("tl-from").value;
    var to = document.getElementById("tl-to").value;
    var asOf = document.getElementById("tl-as-of").value;

    if (ns) params.set("namespace", ns);
    if (entity) params.set("entity", entity);
    if (status && status !== "all") params.set("status", status);
    if (from) params.set("from", from);
    if (to) params.set("to", to);
    if (asOf) params.set("as_of", asOf);

    return params.toString();
  }

  async function fetchTimeline() {
    var area = document.getElementById("timeline-area");
    area.innerHTML = '<div class="tl-loading">Reconstructing timeline...</div>';
    hideDetail();

    var qs = buildQueryString();
    var url = "/ui/timeline.json" + (qs ? "?" + qs : "");

    try {
      var res = await fetch(url);
      var data = await res.json();
      if (data.error) {
        area.innerHTML = '<div class="tl-empty">' + esc(data.error) + "</div>";
        return;
      }
      allMemories = data.memories || [];
      renderTimeline(allMemories, area);
    } catch (e) {
      area.innerHTML = '<div class="tl-empty">Request failed: ' + esc(e.message) + "</div>";
    }
  }

  // ---- Render ----

  function renderTimeline(memories, container) {
    if (!memories.length) {
      container.innerHTML = '<div class="tl-empty">No memories match these filters.</div>';
      return;
    }

    var html = '<div class="tl-spine">';

    memories.forEach(function (m, i) {
      var side = i % 2 === 0 ? "left" : "right";
      var isSuperseded = m.status === "superseded";
      var cls = "tl-card tl-card--" + side;
      if (isSuperseded) cls += " tl-card--superseded";

      var confPct = fmtConfidence(m.confidence);
      var stampCls = "tl-stamp";
      if (isSuperseded) stampCls += " tl-stamp--superseded";
      else if (confPct < 60) stampCls += " tl-stamp--decayed";

      html += '<div class="tl-node" style="animation-delay:' + (i * 60) + 'ms">';
      html += '  <div class="tl-node__dot"></div>';
      html += '  <div class="tl-node__date">' + esc(fmtDate(m.created_at)) + "</div>";
      html += '  <div class="' + cls + '" data-idx="' + i + '">';

      // Header row
      html += '    <div class="tl-card__head">';
      html += '      <div class="tl-card__entity">' + esc(m.entity || "---") + "</div>";
      html += '      <div class="' + stampCls + '">';
      html += '        <span class="tl-stamp__pct">' + confPct + "</span>";
      html += '        <span class="tl-stamp__label">CONF</span>';
      html += "      </div>";
      html += "    </div>";

      // Category
      if (m.category) {
        html += '    <div class="tl-card__category">' + esc(m.category) + "</div>";
      }

      // Content excerpt
      html += '    <div class="tl-card__content">' + esc(excerpt(m.content, 150)) + "</div>";

      // Temporal metadata
      html += '    <div class="tl-card__meta">';
      html += '      <span>' + esc(fmtDate(m.created_at));
      if (fmtTime(m.created_at)) html += " " + esc(fmtTime(m.created_at));
      html += "</span>";
      if (m.namespace) html += '  <span>' + esc(m.namespace) + "</span>";
      html += "    </div>";

      // Supersession indicator
      if (m.superseded_by) {
        html += '    <div class="tl-card__superseded">';
        html += '      <span class="tl-card__arrow">&#x2192;</span> ';
        html += "      superseded by " + esc(m.superseded_by.slice(0, 8));
        html += "    </div>";
      }

      // Validity window
      if (m.valid_from || m.valid_until) {
        html += '    <div class="tl-card__validity">';
        html += "      VALID " + esc(fmtDate(m.valid_from) || "...");
        html += " &ndash; " + esc(fmtDate(m.valid_until) || "present");
        html += "    </div>";
      }

      html += "  </div>"; // close tl-card
      html += "</div>"; // close tl-node
    });

    html += "</div>"; // close tl-spine

    container.innerHTML = html;

    // Bind click handlers
    container.querySelectorAll(".tl-card").forEach(function (el) {
      el.addEventListener("click", function () {
        var idx = parseInt(el.getAttribute("data-idx"), 10);
        showDetail(allMemories[idx]);
      });
    });
  }

  // ---- Detail panel ----

  function showDetail(m) {
    var panel = document.getElementById("tl-detail");
    var body = document.getElementById("tl-detail-body");

    var html = "";

    html += '<div class="tl-detail__id">ID: ' + esc(m.id) + "</div>";

    if (m.entity) {
      html += '<div class="tl-detail__row"><span>Entity</span>' + esc(m.entity) + "</div>";
    }
    html += '<div class="tl-detail__row"><span>Category</span>' + esc(m.category || "general") + "</div>";
    html += '<div class="tl-detail__row"><span>Namespace</span>' + esc(m.namespace || "default") + "</div>";
    html += '<div class="tl-detail__row"><span>Status</span>' + esc(m.status) + "</div>";
    html += '<div class="tl-detail__row"><span>Confidence</span>' + fmtConfidence(m.confidence) + "%</div>";
    html += '<div class="tl-detail__row"><span>Created</span>' + esc(m.created_at || "") + "</div>";

    if (m.event_date) {
      html += '<div class="tl-detail__row"><span>Event Date</span>' + esc(m.event_date) + "</div>";
    }
    if (m.valid_from) {
      html += '<div class="tl-detail__row"><span>Valid From</span>' + esc(m.valid_from) + "</div>";
    }
    if (m.valid_until) {
      html += '<div class="tl-detail__row"><span>Valid Until</span>' + esc(m.valid_until) + "</div>";
    }
    if (m.superseded_by) {
      html += '<div class="tl-detail__row tl-detail__row--rust"><span>Superseded By</span>' + esc(m.superseded_by) + "</div>";
    }
    if (m.access_count) {
      html += '<div class="tl-detail__row"><span>Accesses</span>' + m.access_count + "</div>";
    }

    // Tags
    if (m.tags && m.tags.length) {
      html += '<div class="tl-detail__tags">';
      m.tags.forEach(function (t) {
        html += '<span class="tl-detail__tag">' + esc(t) + "</span>";
      });
      html += "</div>";
    }

    // Full content
    html += '<div class="tl-detail__section">Content</div>';
    html += '<div class="tl-detail__content">' + esc(m.content) + "</div>";

    body.innerHTML = html;
    panel.hidden = false;
  }

  function hideDetail() {
    document.getElementById("tl-detail").hidden = true;
  }

  // ---- Wire up ----

  document.getElementById("timeline-form").addEventListener("submit", function (e) {
    e.preventDefault();
    fetchTimeline();
  });

  document.getElementById("tl-go").addEventListener("click", fetchTimeline);

  document.getElementById("tl-detail-close").addEventListener("click", hideDetail);

  // Enter key in any text field triggers fetch
  document.querySelectorAll("#timeline-form .field").forEach(function (el) {
    el.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        fetchTimeline();
      }
    });
  });

  // Auto-load timeline with status=all on page ready
  document.getElementById("tl-status").value = "all";
  fetchTimeline();
})();
