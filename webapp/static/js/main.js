/**
 * CR-DiffNet Interactive Comparison Slider & Multi-Spectral Inspector
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sliderContainer = document.getElementById("comparison-slider");
  const layerAfter = document.getElementById("layer-after");
  const sliderHandle = document.getElementById("slider-handle");
  
  const imgCloudy = document.getElementById("img-cloudy");
  const imgRecon = document.getElementById("img-recon");
  
  const metricPsnr = document.getElementById("metric-psnr");
  const metricSsim = document.getElementById("metric-ssim");
  const metricSam = document.getElementById("metric-sam");
  const metricNdvi = document.getElementById("metric-ndvi");
  const metricCbdi = document.getElementById("metric-cbdi");
  const metricLatency = document.getElementById("metric-latency");
  
  const modeButtons = document.querySelectorAll(".mode-btn");
  const coverageSlider = document.getElementById("coverage-slider");
  const coverageVal = document.getElementById("coverage-val");
  const azimuthSlider = document.getElementById("azimuth-slider");
  const azimuthVal = document.getElementById("azimuth-val");
  const btnRegenClouds = document.getElementById("btn-regen-clouds");

  let currentMode = "rgb";
  let sceneImages = {};
  let isDragging = false;

  // ==========================================================================
  // 1. Interactive Slider Dragging Engine
  // ==========================================================================
  function updateSliderPosition(clientX) {
    const rect = sliderContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    
    // Clamp between 2% and 98%
    percentage = Math.max(2, Math.min(98, percentage));
    
    // Update clip path on after image (revealing before image on the left)
    layerAfter.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    sliderHandle.style.left = `${percentage}%`;
  }

  sliderContainer.addEventListener("mousedown", (e) => {
    isDragging = true;
    updateSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    updateSliderPosition(e.clientX);
  });

  // Touch Support for Mobile / Tablets
  sliderContainer.addEventListener("touchstart", (e) => {
    isDragging = true;
    updateSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDragging = false;
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    updateSliderPosition(e.touches[0].clientX);
  });

  // ==========================================================================
  // 2. Fetch Scene Data & Update Display
  // ==========================================================================
  async function loadSceneData() {
    try {
      const res = await fetch("/api/scene-data");
      const data = await res.json();
      
      if (data.status === "success") {
        sceneImages = data.images;
        
        // Update Metrics
        metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        metricSsim.textContent = data.metrics.ssim;
        metricSam.textContent = `${data.metrics.sam_deg}°`;
        metricNdvi.textContent = data.metrics.ndvi_mae;
        metricCbdi.textContent = data.metrics.cbdi_seam;
        metricLatency.textContent = `${data.metrics.latency_ms} ms`;
        
        coverageVal.textContent = `${data.cloud_coverage_pct}%`;
        coverageSlider.value = data.cloud_coverage_pct;

        renderCurrentMode();
      }
    } catch (err) {
      console.error("Failed to load scene data:", err);
    }
  }

  // ==========================================================================
  // 3. Multi-Spectral Layer Switcher
  // ==========================================================================
  function renderCurrentMode() {
    if (!sceneImages.cloudy_rgb) return;

    if (currentMode === "rgb") {
      imgCloudy.src = sceneImages.cloudy_rgb;
      imgRecon.src = sceneImages.recon_rgb;
    } else if (currentMode === "fcir") {
      imgCloudy.src = sceneImages.cloudy_fcir;
      imgRecon.src = sceneImages.recon_fcir;
    } else if (currentMode === "ndvi") {
      imgCloudy.src = sceneImages.cloudy_ndvi;
      imgRecon.src = sceneImages.recon_ndvi;
    } else if (currentMode === "mask") {
      imgCloudy.src = sceneImages.cloudy_rgb;
      imgRecon.src = sceneImages.mask;
    } else if (currentMode === "sar") {
      imgCloudy.src = sceneImages.cloudy_rgb;
      imgRecon.src = sceneImages.sar;
    } else if (currentMode === "dem") {
      imgCloudy.src = sceneImages.cloudy_rgb;
      imgRecon.src = sceneImages.dem;
    }
  }

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
      renderCurrentMode();
    });
  });

  // ==========================================================================
  // 4. Cloud Coverage Slider Controls
  // ==========================================================================
  coverageSlider.addEventListener("input", (e) => {
    coverageVal.textContent = `${e.target.value}%`;
  });

  azimuthSlider.addEventListener("input", (e) => {
    azimuthVal.textContent = `${e.target.value}°`;
  });

  btnRegenClouds.addEventListener("click", async () => {
    btnRegenClouds.disabled = true;
    btnRegenClouds.innerHTML = `<span>⏳</span> Generating Clouds...`;
    
    try {
      await fetch("/api/regenerate-clouds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          coverage: parseFloat(coverageSlider.value) / 100.0,
          sun_azimuth: parseFloat(azimuthSlider.value)
        })
      });
      await loadSceneData();
    } catch (err) {
      console.error("Error regenerating clouds:", err);
    } finally {
      btnRegenClouds.disabled = false;
      btnRegenClouds.innerHTML = `<span>⚡</span> Re-Simulate Cloud Scene`;
    }
  });

  // Initial Load
  loadSceneData();
});
