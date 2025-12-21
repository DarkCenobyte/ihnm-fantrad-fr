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

// Lightbox simple (4 captures)
document.querySelectorAll(".shot").forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const src = link.getAttribute("href");

    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(0,0,0,.85)";
    overlay.style.display = "grid";
    overlay.style.placeItems = "center";
    overlay.style.zIndex = "9999";
    overlay.style.padding = "24px";
    overlay.style.cursor = "zoom-out";

    const img = document.createElement("img");
    img.src = src;
    img.alt = link.querySelector("img")?.alt || "Capture d'écran";
    img.style.maxWidth = "min(1100px, 95vw)";
    img.style.maxHeight = "90vh";
    img.style.borderRadius = "16px";
    img.style.border = "1px solid rgba(35,42,51,.9)";
    img.style.boxShadow = "0 18px 50px rgba(0,0,0,.65)";

    overlay.appendChild(img);
    overlay.addEventListener("click", () => overlay.remove());
    document.addEventListener("keydown", function esc(ev){
      if (ev.key === "Escape") overlay.remove();
      document.removeEventListener("keydown", esc);
    });

    document.body.appendChild(overlay);
  });
});
