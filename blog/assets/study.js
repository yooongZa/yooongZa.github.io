"use strict";

const valueInput = document.querySelector("#input-value");
if (valueInput) {
  valueInput.addEventListener("input", () => {
    const value = Number(valueInput.value);
    document.querySelector("#input-label").textContent = value;
    document.querySelector("#equation-input").textContent = value;
    document.querySelector("#explorer-title").textContent = `새 값이 ${value}이라면?`;
    document.querySelector("#scaled-value").textContent = ((value - 10) / 20).toFixed(2);
  });
}

const audio = document.querySelector("#study-audio");
const rate = document.querySelector("#playback-rate");
if (audio && rate) {
  rate.addEventListener("change", () => { audio.playbackRate = Number(rate.value); });
}

const progress = document.querySelector("#reading-progress");
if (progress) {
  const updateProgress = () => {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${available > 0 ? Math.min(100, window.scrollY / available * 100) : 0}%`;
  };
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
  updateProgress();
}
