// Minimal UI choreography. HTMX handles fragment swapping;
// this file only adds micro-behaviors that CSS can't express.

// Theme toggle — must be global for onclick="toggleTheme()"
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("memwright-theme", theme);
}

function toggleTheme() {
  var current = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
}

// Apply saved theme immediately (before DOMContentLoaded to avoid flash)
(function () {
  var saved = localStorage.getItem("memwright-theme");
  if (saved) applyTheme(saved);
})();

(function () {
  // 1) Stagger card index CSS var so the load animation waterfalls.
  document.querySelectorAll(".card").forEach((el, i) => {
    el.style.setProperty("--i", Math.min(i, 30));
  });

  // 2) Apply per-card tilt from data attribute.
  document.querySelectorAll("[data-tilt]").forEach((el) => {
    el.style.setProperty("--tilt", el.dataset.tilt + "deg");
  });

  // 3) Tab bar keyboard navigation on the dossier detail page.
  const tabs = document.querySelector(".tabs");
  if (tabs) {
    tabs.addEventListener("keydown", (e) => {
      const items = [...tabs.querySelectorAll(".tabs__item")];
      const idx = items.indexOf(document.activeElement);
      if (idx === -1) return;
      if (e.key === "ArrowRight") items[(idx + 1) % items.length].focus();
      if (e.key === "ArrowLeft") items[(idx - 1 + items.length) % items.length].focus();
    });
  }

  // 4) Live-updating "last captured" tick in the ticker.
  const tick = document.querySelector(".ticker__pulse");
  if (tick) {
    const start = Date.now();
    setInterval(() => {
      const secs = Math.floor((Date.now() - start) / 1000);
      tick.dataset.secs = String(secs);
    }, 1000);
  }

  // 5) Search result highlighting in Evidence Board cards.
  function highlightTerms() {
    const grid = document.getElementById("grid");
    if (!grid) return;
    const raw = (grid.dataset.query || "").trim();
    if (!raw) return;

    const terms = raw.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return;

    // Build one regex that matches any term, longest first to avoid
    // partial-match clobbering (e.g. "react" before "re").
    const sorted = terms.slice().sort((a, b) => b.length - a.length);
    const escaped = sorted.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp("(" + escaped.join("|") + ")", "gi");

    grid.querySelectorAll(".card__body").forEach((el) => {
      // Walk text nodes only — preserves any existing HTML structure.
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);

      nodes.forEach((textNode) => {
        const text = textNode.nodeValue;
        if (!re.test(text)) return;
        re.lastIndex = 0; // reset after .test()

        const frag = document.createDocumentFragment();
        let lastIdx = 0;
        let match;
        while ((match = re.exec(text)) !== null) {
          if (match.index > lastIdx) {
            frag.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
          }
          const mark = document.createElement("mark");
          mark.className = "hl";
          mark.textContent = match[0];
          frag.appendChild(mark);
          lastIdx = re.lastIndex;
        }
        if (lastIdx < text.length) {
          frag.appendChild(document.createTextNode(text.slice(lastIdx)));
        }
        textNode.parentNode.replaceChild(frag, textNode);
      });
    });
  }

  // Run on initial load.
  highlightTerms();

  // Re-run after HTMX swaps (pagination, filter updates).
  document.body.addEventListener("htmx:afterSwap", highlightTerms);
})();
