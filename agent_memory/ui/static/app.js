// Minimal UI choreography. HTMX handles fragment swapping;
// this file only adds micro-behaviors that CSS can't express.

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
})();
