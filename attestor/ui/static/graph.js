// Memwright — Knowledge Graph (Cytoscape.js)
// Forensic Archive aesthetic: aged-paper nodes, oxidized copper edges.

(function () {
  "use strict";

  // ── Palette keyed to entity types ──────────────────────────────
  const TYPE_COLORS = {
    tool:        { bg: "#D4A84B", border: "#A07A20", text: "#0A0C10" }, // brass
    person:      { bg: "#C75146", border: "#8B2E26", text: "#E8DFC6" }, // rust
    concept:     { bg: "#4A7C7E", border: "#2D5556", text: "#E8DFC6" }, // verdigris
    project:     { bg: "#3B4A6B", border: "#242F48", text: "#E8DFC6" }, // indigo
    organization:{ bg: "#9B8E66", border: "#6B6140", text: "#0A0C10" }, // paper-shade
    location:    { bg: "#8B6E4E", border: "#5A4530", text: "#E8DFC6" }, // leather
    event:       { bg: "#7E4A6E", border: "#52304A", text: "#E8DFC6" }, // plum
    general:     { bg: "#E8DFC6", border: "#C9BB93", text: "#0A0C10" }, // paper
  };

  const EDGE_COLORS = {
    USES:        "#D4A84B",
    AUTHORED_BY: "#C75146",
    RELATED_TO:  "#4A7C7E",
    SUPERSEDES:  "#8B2E26",
    CONTAINS:    "#3B4A6B",
    DEPENDS_ON:  "#9B8E66",
  };

  const DEFAULT_EDGE_COLOR = "rgba(232, 223, 198, 0.35)";

  function colorForType(type) {
    return TYPE_COLORS[type] || TYPE_COLORS.general;
  }

  function colorForEdge(type) {
    return EDGE_COLORS[type] || DEFAULT_EDGE_COLOR;
  }

  // ── State ──────────────────────────────────────────────────────
  let cy = null;
  let graphData = null;

  // ── Fetch graph data ───────────────────────────────────────────
  async function loadGraph() {
    const res = await fetch("/ui/graph.json");
    graphData = await res.json();
    return graphData;
  }

  // ── Build Cytoscape elements ───────────────────────────────────
  function buildElements(data) {
    const nodes = data.nodes.map((n) => ({
      data: {
        id: n.id,
        label: n.label,
        type: n.type || "general",
        pagerank: n.pagerank || 0,
      },
    }));

    const edges = data.edges.map((e, i) => ({
      data: {
        id: "e" + i,
        source: e.source,
        target: e.target,
        type: e.type || "RELATED_TO",
      },
    }));

    return [...nodes, ...edges];
  }

  // ── Sizing ─────────────────────────────────────────────────────
  function nodeSizeFn(mode) {
    if (mode === "uniform") return () => 32;
    if (mode === "degree") {
      return (ele) => {
        const d = ele.degree(false) || 1;
        return Math.max(24, Math.min(80, 20 + d * 6));
      };
    }
    // pagerank (default)
    return (ele) => {
      const pr = ele.data("pagerank") || 0;
      // Scale: 0 → 24px, max → 80px
      return Math.max(24, Math.min(80, 24 + pr * 4000));
    };
  }

  // ── Init Cytoscape ─────────────────────────────────────────────
  function initCy(elements) {
    cy = cytoscape({
      container: document.getElementById("cy"),
      elements: elements,
      minZoom: 0.15,
      maxZoom: 4,
      // wheelSensitivity left at default to avoid warning on standard mice

      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            width: 32,
            height: 32,
            "background-color": (ele) => colorForType(ele.data("type")).bg,
            "border-width": 2,
            "border-color": (ele) => colorForType(ele.data("type")).border,
            color: "#E8DFC6",
            "font-family": '"Departure Mono", "IBM Plex Mono", monospace',
            "font-size": 9,
            "text-valign": "bottom",
            "text-margin-y": 6,
            "text-outline-width": 2,
            "text-outline-color": "#0A0C10",
            "text-max-width": 100,
            "text-wrap": "ellipsis",
            "overlay-padding": 4,
            "transition-property": "width, height, border-width, opacity",
            "transition-duration": "0.2s",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#C75146",
            "overlay-color": "#C75146",
            "overlay-opacity": 0.12,
          },
        },
        {
          selector: "node.highlighted",
          style: {
            "border-width": 3,
            "border-color": "#C75146",
            "font-size": 11,
            "z-index": 10,
          },
        },
        {
          selector: "node.dimmed",
          style: {
            opacity: 0.15,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": (ele) => colorForEdge(ele.data("type")),
            "target-arrow-color": (ele) => colorForEdge(ele.data("type")),
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            opacity: 0.6,
            "transition-property": "opacity, width",
            "transition-duration": "0.2s",
          },
        },
        {
          selector: "edge.highlighted",
          style: {
            width: 2.5,
            opacity: 1,
            label: "data(type)",
            "font-family": '"Departure Mono", monospace',
            "font-size": 8,
            color: "rgba(232, 223, 198, 0.7)",
            "text-rotation": "autorotate",
            "text-outline-width": 2,
            "text-outline-color": "#0A0C10",
          },
        },
        {
          selector: "edge.dimmed",
          style: {
            opacity: 0.06,
          },
        },
      ],

      layout: {
        name: "cose",
        animate: true,
        animationDuration: 800,
        nodeRepulsion: () => 8000,
        idealEdgeLength: () => 80,
        gravity: 0.3,
        numIter: 500,
        padding: 60,
      },
    });

    // Fit after layout settles
    cy.one("layoutstop", () => {
      cy.fit(cy.elements(), 50);
    });

    // Apply PageRank sizing
    applySizing("pagerank");

    // ── Interactions ───────────────────────────────────────────
    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      highlightNeighborhood(node);
      showInspector(node);
    });

    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        clearHighlight();
        hideInspector();
      }
    });
  }

  // ── Highlight neighborhood ─────────────────────────────────────
  function highlightNeighborhood(node) {
    clearHighlight();
    const neighborhood = node.closedNeighborhood();
    cy.elements().addClass("dimmed");
    neighborhood.removeClass("dimmed").addClass("highlighted");
  }

  function clearHighlight() {
    cy.elements().removeClass("dimmed highlighted");
  }

  // ── Inspector panel ────────────────────────────────────────────
  function showInspector(node) {
    const d = node.data();
    const degree = node.degree(false);
    const inDeg = node.indegree(false);
    const outDeg = node.outdegree(false);
    const neighbors = node.neighborhood("node");
    const edges = node.connectedEdges();

    const c = colorForType(d.type);

    let html = `
      <div class="graph-inspector__head">
        <span class="graph-inspector__dot" style="background:${c.bg};border-color:${c.border}"></span>
        <div>
          <div class="graph-inspector__name">${esc(d.label)}</div>
          <div class="graph-inspector__type">${esc(d.type)}</div>
        </div>
      </div>
      <div class="graph-inspector__stats">
        <div class="graph-inspector__stat">
          <span class="graph-inspector__stat-val">${d.pagerank.toFixed(4)}</span>
          <span class="graph-inspector__stat-lbl">PageRank</span>
        </div>
        <div class="graph-inspector__stat">
          <span class="graph-inspector__stat-val">${degree}</span>
          <span class="graph-inspector__stat-lbl">Degree</span>
        </div>
        <div class="graph-inspector__stat">
          <span class="graph-inspector__stat-val">${inDeg} / ${outDeg}</span>
          <span class="graph-inspector__stat-lbl">In / Out</span>
        </div>
      </div>
    `;

    // Connected edges
    if (edges.length > 0) {
      html += `<div class="graph-inspector__section-label">Relations</div><ul class="graph-inspector__edges">`;
      edges.forEach((e) => {
        const ed = e.data();
        const isOutgoing = ed.source === d.id;
        const otherLabel = isOutgoing
          ? cy.getElementById(ed.target).data("label")
          : cy.getElementById(ed.source).data("label");
        const arrow = isOutgoing ? "→" : "←";
        html += `<li><span class="graph-inspector__edge-type">${esc(ed.type)}</span> ${arrow} ${esc(otherLabel)}</li>`;
      });
      html += `</ul>`;
    }

    // Neighbor list
    if (neighbors.length > 0) {
      html += `<div class="graph-inspector__section-label">Neighbors · ${neighbors.length}</div><ul class="graph-inspector__neighbors">`;
      neighbors
        .sort((a, b) => (b.data("pagerank") || 0) - (a.data("pagerank") || 0))
        .forEach((n) => {
          const nd = n.data();
          const nc = colorForType(nd.type);
          html += `<li class="graph-inspector__neighbor" data-id="${esc(nd.id)}"><span class="graph-inspector__ndot" style="background:${nc.bg}"></span>${esc(nd.label)}</li>`;
        });
      html += `</ul>`;
    }

    const panel = document.getElementById("inspector");
    panel.innerHTML = html;
    panel.classList.add("is-active");

    // Click neighbor to navigate
    panel.querySelectorAll(".graph-inspector__neighbor").forEach((el) => {
      el.addEventListener("click", () => {
        const target = cy.getElementById(el.dataset.id);
        if (target.length) {
          cy.animate({ center: { eles: target }, zoom: 1.5 }, { duration: 400 });
          highlightNeighborhood(target);
          showInspector(target);
        }
      });
    });
  }

  function hideInspector() {
    const panel = document.getElementById("inspector");
    panel.innerHTML = `<div class="graph-inspector__empty">Select a node to inspect</div>`;
    panel.classList.remove("is-active");
  }

  function esc(str) {
    const d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }

  // ── Sizing ─────────────────────────────────────────────────────
  function applySizing(mode) {
    const fn = nodeSizeFn(mode);
    cy.nodes().forEach((n) => {
      const s = fn(n);
      n.style({ width: s, height: s });
    });
  }

  // ── Layout ─────────────────────────────────────────────────────
  function applyLayout(name) {
    const opts = { name, animate: true, animationDuration: 600, padding: 40 };
    if (name === "cose") {
      opts.nodeRepulsion = () => 8000;
      opts.idealEdgeLength = () => 80;
      opts.gravity = 0.3;
      opts.numIter = 500;
    }
    if (name === "concentric") {
      opts.concentric = (n) => n.data("pagerank") || 0;
      opts.levelWidth = () => 0.5;
    }
    if (name === "breadthfirst") {
      opts.directed = true;
      opts.spacingFactor = 1.2;
    }
    cy.layout(opts).run();
  }

  // ── Legend ──────────────────────────────────────────────────────
  function buildLegend(types) {
    const container = document.getElementById("graph-legend");
    let html = `<div class="rail__label">Legend</div><ul class="graph-legend__list">`;
    const seen = new Set();
    for (const t of Object.keys(types).sort()) {
      seen.add(t);
      const c = colorForType(t);
      html += `<li class="graph-legend__item"><span class="graph-legend__dot" style="background:${c.bg};border-color:${c.border}"></span>${t} · ${types[t]}</li>`;
    }
    html += `</ul>`;
    container.innerHTML = html;
  }

  // ── Type filter dropdown ───────────────────────────────────────
  function populateTypeFilter(types) {
    const sel = document.getElementById("graph-type-filter");
    for (const t of Object.keys(types).sort()) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = `${t} (${types[t]})`;
      sel.appendChild(opt);
    }
  }

  // ── Search ─────────────────────────────────────────────────────
  function setupSearch() {
    const input = document.getElementById("graph-search");
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      if (!q) {
        clearHighlight();
        cy.nodes().show();
        return;
      }
      cy.nodes().forEach((n) => {
        const label = (n.data("label") || "").toLowerCase();
        if (label.includes(q)) {
          n.show();
          n.removeClass("dimmed");
        } else {
          n.addClass("dimmed");
        }
      });
      // Center on first match
      const matches = cy.nodes().filter((n) =>
        (n.data("label") || "").toLowerCase().includes(q)
      );
      if (matches.length === 1) {
        cy.animate({ center: { eles: matches }, zoom: 1.5 }, { duration: 400 });
        highlightNeighborhood(matches[0]);
        showInspector(matches[0]);
      }
    });
  }

  // ── Controls ───────────────────────────────────────────────────
  function setupControls() {
    document.getElementById("graph-fit").addEventListener("click", () => {
      cy.animate({ fit: { eles: cy.elements(), padding: 40 } }, { duration: 400 });
    });

    document.getElementById("graph-reset").addEventListener("click", () => {
      clearHighlight();
      hideInspector();
      document.getElementById("graph-search").value = "";
      document.getElementById("graph-type-filter").value = "";
      cy.nodes().show();
      applyLayout("cose");
    });

    document.getElementById("graph-layout").addEventListener("change", (e) => {
      applyLayout(e.target.value);
    });

    document.getElementById("graph-sizing").addEventListener("change", (e) => {
      applySizing(e.target.value);
    });

    document.getElementById("graph-type-filter").addEventListener("change", (e) => {
      const type = e.target.value;
      if (!type) {
        cy.nodes().show();
        cy.edges().show();
      } else {
        cy.nodes().forEach((n) => {
          if (n.data("type") === type) {
            n.show();
          } else {
            n.hide();
          }
        });
        // Show edges only between visible nodes
        cy.edges().forEach((edge) => {
          if (edge.source().visible() && edge.target().visible()) {
            edge.show();
          } else {
            edge.hide();
          }
        });
      }
    });
  }

  // ── Boot ───────────────────────────────────────────────────────
  async function boot() {
    const data = await loadGraph();

    if (!data.nodes || data.nodes.length === 0) {
      document.getElementById("cy").innerHTML =
        '<div class="graph-empty">No entities in graph yet.<br>Add memories with entities to populate.</div>';
      return;
    }

    const elements = buildElements(data);
    initCy(elements);

    const types = data.stats.types || {};
    buildLegend(types);
    populateTypeFilter(types);
    setupSearch();
    setupControls();
  }

  boot();
})();
