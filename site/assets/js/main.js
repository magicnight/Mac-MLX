import { legacyLanguageDestination } from "./legacy-language.mjs";

(function () {
  "use strict";

  const themeKey = "macmlx-theme";

  function savedValue(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function saveValue(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      // The page still works when storage is unavailable.
    }
  }

  function migrateLegacyLanguageQuery() {
    const destination = legacyLanguageDestination(window.location.href);
    if (destination === null) return false;
    window.location.replace(destination);
    return true;
  }

  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") {
      document.documentElement.dataset.theme = theme;
    } else {
      delete document.documentElement.dataset.theme;
    }
  }

  function initialiseTheme() {
    const saved = savedValue(themeKey);
    applyTheme(saved);
    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      const current = savedValue(themeKey) || "system";
      const next = current === "system" ? "light" : current === "light" ? "dark" : "system";
      saveValue(themeKey, next === "system" ? null : next);
      applyTheme(next);
    });
  }

  function initialiseCopyButton() {
    const button = document.getElementById("copy-command");
    if (!button) return;
    const originalLabel = button.textContent;
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy || "");
        button.textContent = button.dataset.copySuccess || originalLabel;
        window.setTimeout(() => {
          button.textContent = originalLabel;
        }, 1400);
      } catch {
        button.textContent = button.dataset.copyFailure || originalLabel;
      }
    });
  }

  function initialiseReveal() {
    const nodes = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      nodes.forEach((node) => node.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8%", threshold: 0.08 });

    nodes.forEach((node) => observer.observe(node));
    document.documentElement.classList.add("reveal-ready");
  }

  function initialiseEngineStory() {
    const story = document.querySelector("[data-engine-story]");
    if (!story) return;

    const steps = [...story.querySelectorAll("[data-engine-index]")];
    const panels = [...story.querySelectorAll("[data-engine-panel]")];
    const progress = [...story.querySelectorAll("[data-engine-progress]")];
    if (steps.length === 0) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    function setActive(index) {
      const bounded = Math.max(0, Math.min(steps.length - 1, index));
      story.dataset.engineStep = String(bounded);
      steps.forEach((step) => step.classList.toggle("is-active", Number(step.dataset.engineIndex) === bounded));
      panels.forEach((panel) => panel.classList.toggle("is-active", Number(panel.dataset.enginePanel) === bounded));
      progress.forEach((segment) => segment.classList.toggle("is-active", Number(segment.dataset.engineProgress) === bounded));
    }

    function activateClosestStep() {
      const viewportCenter = window.innerHeight / 2;
      let closestIndex = 0;
      let closestDistance = Infinity;

      steps.forEach((step, index) => {
        const rect = step.getBoundingClientRect();
        const distance = Math.abs(rect.top + rect.height / 2 - viewportCenter);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestIndex = index;
        }
      });

      setActive(closestIndex);
    }

    activateClosestStep();
    steps.find((step) => Number(step.dataset.engineIndex) === Number(story.dataset.engineStep))?.classList.add("is-visible");

    if (!("IntersectionObserver" in window)) {
      story.classList.add("is-static");
      steps.forEach((step) => step.classList.add("is-visible"));
      return;
    }

    if (reducedMotion.matches) {
      steps.forEach((step) => step.classList.add("is-visible"));
      return;
    }

    story.classList.add("is-enhanced");
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("is-visible");
      });
      activateClosestStep();
    }, {
      rootMargin: "-28% 0px -28%",
      threshold: [0, 0.25, 0.5, 0.75, 1],
    });

    steps.forEach((step) => observer.observe(step));
  }

  function boot() {
    initialiseTheme();
    initialiseCopyButton();
    initialiseReveal();
    initialiseEngineStory();
  }

  if (!migrateLegacyLanguageQuery()) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
  }
})();
