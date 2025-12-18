(() => {
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  // Micro-effet glitch (léger) sur le titre, sans dépendances
  const title = document.querySelector(".title");
  if (!title) return;

  const prefersReduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  if (prefersReduced) return;

  let t = 0;
  const tick = () => {
    t++;
    // Très subtil : une "perturbation" occasionnelle
    if (t % 180 === 0) {
      title.classList.add("glitch-on");
      window.setTimeout(() => title.classList.remove("glitch-on"), 220);
    }
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
})();
