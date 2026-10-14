/**
 * CR-DiffNet: Operational Generative AI Cloud Removal
 * Frontend Controller for Minimalist Solid UI
 * Inspired by WisprType & Modern Apple / shadcn design principles.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================================================
  // Global Application State
  // ==========================================================================
  let trainingScenes = [];
  let selectedSceneId = "S2A_46RBN";
  let currentMode = "rgb";
  let activeTab = "landing"; // 'landing', 'studio', or 'map'
  let sceneImages = {};
  let isDraggingStudio = false;
  let isDraggingLanding = false;

  // Leaflet Map & GIS
  let map = null;
  let currentBasemap = null;
  let labelsLayer = null;
  const polygonLayers = {};
  const markerLayers = {};

  // ==========================================================================
  // DOM Elements - Navigation & Views
  // ==========================================================================
  const tabBtnLanding = document.getElementById("tab-btn-landing");
  const tabBtnStudio = document.getElementById("tab-btn-studio");
  const tabBtnMap = document.getElementById("tab-btn-map");
  const navBrandLogo = document.getElementById("nav-brand-logo");
  const btnNavStudioCta = document.getElementById("btn-nav-studio-cta");

  const viewLanding = document.getElementById("view-landing");
  const viewStudio = document.getElementById("view-studio");
  const viewGisMap = document.getElementById("view-gis-map");

  // Landing Page Elements
  const btnHeroLaunch = document.getElementById("btn-hero-launch");
  const btnHeroMap = document.getElementById("btn-hero-map");
  const btnStoryTry = document.getElementById("btn-story-try");
  const landingDemoSlider = document.getElementById("landing-demo-slider");
  const landingAfterBox = document.getElementById("landing-after-box");
  const landingSliderHandle = document.getElementById("landing-slider-handle");

  // Map & Drawer Elements
  const btnToggleDrawer = document.getElementById("btn-toggle-drawer");
  const btnCloseDrawer = document.getElementById("btn-close-drawer");
  const scenesDrawer = document.getElementById("scenes-drawer");
  const scenesContainer = document.getElementById("scenes-container");
  const scenesCountBadge = document.getElementById("scenes-count-badge");
  const hudCoords = document.getElementById("hud-coords");
  const hudElev = document.getElementById("hud-elev");

  // Studio Header & Download
  const bannerSceneName = document.getElementById("banner-scene-name");
  const bannerSceneCoords = document.getElementById("banner-scene-coords");
  const btnDownloadResult = document.getElementById("btn-download-result");

  // Upload Elements
  const uploadDropzone = document.getElementById("upload-dropzone");
  const fileInputUpload = document.getElementById("file-input-upload");
  const uploadStatus = document.getElementById("upload-status");
  const uploadStatusText = document.getElementById("upload-status-text");

  // 4-Stage Pipeline Tracker Elements
  const trackerPipelineBadge = document.getElementById("tracker-pipeline-badge");
  const trackerSummaryText = document.getElementById("tracker-summary-text");
  const trackerSavedBox = document.getElementById("tracker-saved-box");
  const trackerSavedPath = document.getElementById("tracker-saved-path");

  const stepCardMask = document.getElementById("step-card-mask");
  const stepBadgeMask = document.getElementById("step-badge-mask");
  const stepDescMask = document.getElementById("step-desc-mask");

  const stepCardStage1 = document.getElementById("step-card-stage1");
  const stepBadgeStage1 = document.getElementById("step-badge-stage1");
  const stepDescStage1 = document.getElementById("step-desc-stage1");

  const stepCardStage2 = document.getElementById("step-card-stage2");
  const stepBadgeStage2 = document.getElementById("step-badge-stage2");
  const stepDescStage2 = document.getElementById("step-desc-stage2");

  const stepCardOutput = document.getElementById("step-card-output");
  const stepBadgeOutput = document.getElementById("step-badge-output");
  const stepDescOutput = document.getElementById("step-desc-output");

  // Studio Slider Canvas
  const sliderContainer = document.getElementById("comparison-slider");
  const layerAfter = document.getElementById("layer-after");
  const sliderHandle = document.getElementById("slider-handle");
  const imgCloudy = document.getElementById("img-cloudy");
  const imgRecon = document.getElementById("img-recon");

  const btnViewSplit = document.getElementById("btn-view-split");
  const btnViewRecon = document.getElementById("btn-view-recon");
  const btnViewStage1 = document.getElementById("btn-view-stage1");
  const btnViewStage2 = document.getElementById("btn-view-stage2");
  const btnViewCloudy = document.getElementById("btn-view-cloudy");

  // Studio Metrics Scorecards
  const metricPsnr = document.getElementById("metric-psnr");
  const metricSsim = document.getElementById("metric-ssim");
  const metricSam = document.getElementById("metric-sam");
  const metricNdvi = document.getElementById("metric-ndvi");
  const metricCbdi = document.getElementById("metric-cbdi");
  const metricLatency = document.getElementById("metric-latency");
  const geminiAssessmentText = document.getElementById("gemini-assessment-text");

  // ==========================================================================
  // 1. Navigation Engine (Landing / Studio / Map)
  // ==========================================================================
  function switchView(target) {
    activeTab = target;

    // Reset active nav tab pills
    [tabBtnLanding, tabBtnStudio, tabBtnMap].forEach((btn) => {
      if (btn) btn.classList.remove("active");
    });

    if (target === "landing") {
      if (tabBtnLanding) tabBtnLanding.classList.add("active");
      if (viewLanding) viewLanding.style.display = "block";
      if (viewStudio) viewStudio.style.display = "none";
      if (viewGisMap) viewGisMap.style.display = "none";
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (target === "studio") {
      if (tabBtnStudio) tabBtnStudio.classList.add("active");
      if (viewStudio) viewStudio.style.display = "block";
      if (viewLanding) viewLanding.style.display = "none";
      if (viewGisMap) viewGisMap.style.display = "none";
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (target === "map") {
      if (tabBtnMap) tabBtnMap.classList.add("active");
      if (viewGisMap) viewGisMap.style.display = "flex";
      if (viewLanding) viewLanding.style.display = "none";
      if (viewStudio) viewStudio.style.display = "none";
      if (map) {
        setTimeout(() => map.invalidateSize(), 80);
      }
    }
  }

  if (tabBtnLanding) tabBtnLanding.addEventListener("click", () => switchView("landing"));
  if (tabBtnStudio) tabBtnStudio.addEventListener("click", () => switchView("studio"));
  if (tabBtnMap) tabBtnMap.addEventListener("click", () => switchView("map"));
  if (navBrandLogo) navBrandLogo.addEventListener("click", () => switchView("landing"));
  if (btnNavStudioCta) btnNavStudioCta.addEventListener("click", () => switchView("studio"));

  if (btnHeroLaunch) btnHeroLaunch.addEventListener("click", () => switchView("studio"));
  if (btnHeroMap) btnHeroMap.addEventListener("click", () => switchView("map"));
  if (btnStoryTry) btnStoryTry.addEventListener("click", () => switchView("studio"));

  // ==========================================================================
  // 2. Landing Page Interactive Split-Slider
  // ==========================================================================
  function setLandingSliderPercentage(pct) {
    pct = Math.max(0, Math.min(100, pct));
    if (landingAfterBox) {
      landingAfterBox.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    }
    if (landingSliderHandle) {
      landingSliderHandle.style.left = `${pct}%`;
    }
  }

  function updateLandingSliderPosition(clientX) {
    if (!landingDemoSlider) return;
    const rect = landingDemoSlider.getBoundingClientRect();
    let x = clientX - rect.left;
    let pct = (x / rect.width) * 100;
    setLandingSliderPercentage(pct);
  }

  if (landingDemoSlider) {
    landingDemoSlider.addEventListener("mousedown", (e) => {
      isDraggingLanding = true;
      updateLandingSliderPosition(e.clientX);
    });

    landingDemoSlider.addEventListener("touchstart", (e) => {
      isDraggingLanding = true;
      updateLandingSliderPosition(e.touches[0].clientX);
    });
  }

  window.addEventListener("mouseup", () => {
    isDraggingLanding = false;
  });

  window.addEventListener("touchend", () => {
    isDraggingLanding = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingLanding) return;
    updateLandingSliderPosition(e.clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDraggingLanding) return;
    updateLandingSliderPosition(e.touches[0].clientX);
  });

  // ==========================================================================
  // 3. Slide-Over Drawer for Map
  // ==========================================================================
  if (btnToggleDrawer && scenesDrawer) {
    btnToggleDrawer.addEventListener("click", (e) => {
      e.stopPropagation();
      scenesDrawer.classList.toggle("open");
    });
  }

  if (btnCloseDrawer && scenesDrawer) {
    btnCloseDrawer.addEventListener("click", (e) => {
      e.stopPropagation();
      scenesDrawer.classList.remove("open");
    });
  }

  // Close drawer if clicking outside on map
  document.addEventListener("click", (e) => {
    if (scenesDrawer && scenesDrawer.classList.contains("open")) {
      if (!scenesDrawer.contains(e.target) && (!btnToggleDrawer || !btnToggleDrawer.contains(e.target))) {
        scenesDrawer.classList.remove("open");
      }
    }
  });

  // ==========================================================================
  // 4. Leaflet GIS Map Initialization
  // ==========================================================================
  function initMap() {
    const mapElement = document.getElementById("training-map");
    if (!mapElement || typeof L === "undefined") return;

    // Center of North East India (Brahmaputra Valley / Assam)
    map = L.map("training-map", {
      center: [26.20, 92.60],
      zoom: 7,
      zoomControl: true,
      attributionControl: false
    });

    currentBasemap = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 18, attribution: "Esri World Imagery" }
    );
    currentBasemap.addTo(map);

    labelsLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 18, opacity: 0.85 }
    );
    labelsLayer.addTo(map);

    map.on("mousemove", (e) => {
      if (hudCoords) {
        hudCoords.textContent = `${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
      }
    });
  }

  // ==========================================================================
  // 5. Render Real Training Scenes on Map & Drawer
  // ==========================================================================
  function renderTrainingScenesOnMap() {
    if (!scenesContainer) return;
    scenesContainer.innerHTML = "";
    if (scenesCountBadge) {
      scenesCountBadge.textContent = `${trainingScenes.length} Scenes`;
    }

    trainingScenes.forEach((scene) => {
      // Scene Card inside Slide-Over Drawer
      const card = document.createElement("div");
      card.className = `scene-card-item ${scene.id === selectedSceneId ? "selected" : ""}`;
      card.id = `sci-${scene.id}`;
      card.innerHTML = `
        <div class="sci-top">
          <span class="sci-name">${scene.name}</span>
          <span class="sci-badge">${scene.mgrs}</span>
        </div>
        <div class="sci-bottom">
          <span>${scene.state} • ${scene.elevation_m}m MSL</span>
          <span class="highlight-cyan">${scene.patches_trained} Patches</span>
        </div>
      `;
      card.addEventListener("click", () => {
        selectScene(scene.id, true);
        if (window.innerWidth < 768 && scenesDrawer) {
          scenesDrawer.classList.remove("open");
        }
      });
      scenesContainer.appendChild(card);

      // Polygon Bounding Box on Leaflet Map
      if (map && scene.bounds) {
        const polyBounds = [
          [scene.bounds[0][0], scene.bounds[0][1]],
          [scene.bounds[0][0], scene.bounds[1][1]],
          [scene.bounds[1][0], scene.bounds[1][1]],
          [scene.bounds[1][0], scene.bounds[0][1]]
        ];

        const isInitialSelected = (scene.id === selectedSceneId);
        const polygon = L.polygon(polyBounds, {
          color: isInitialSelected ? "#0D4A3E" : "#38bdf8",
          weight: isInitialSelected ? 3 : 1.5,
          fillColor: isInitialSelected ? "#0D4A3E" : "#38bdf8",
          fillOpacity: isInitialSelected ? 0.32 : 0.12,
          dashArray: isInitialSelected ? null : "4, 4"
        }).addTo(map);

        const marker = L.circleMarker([scene.lat, scene.lon], {
          radius: isInitialSelected ? 8 : 6,
          color: isInitialSelected ? "#ffffff" : "#0D4A3E",
          fillColor: isInitialSelected ? "#0D4A3E" : "#0284c7",
          fillOpacity: 0.9,
          weight: 2
        }).addTo(map);

        polygon.bindTooltip(
          `<strong>${scene.name}</strong><br><span style="color:#0D4A3E">MGRS: ${scene.mgrs}</span> • ${scene.patches_trained} Patches`,
          { sticky: true, opacity: 0.9 }
        );

        polygon.on("click", () => {
          selectScene(scene.id, true);
        });
        marker.on("click", () => {
          selectScene(scene.id, true);
        });

        polygonLayers[scene.id] = polygon;
        markerLayers[scene.id] = marker;
      }
    });
  }

  function selectScene(sceneId, flyTo = false) {
    selectedSceneId = sceneId;
    const scene = trainingScenes.find(s => s.id === sceneId);
    if (!scene) return;

    // Update active highlight in drawer
    document.querySelectorAll(".scene-card-item").forEach(el => el.classList.remove("selected"));
    const activeCard = document.getElementById(`sci-${sceneId}`);
    if (activeCard) {
      activeCard.classList.add("selected");
      activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Highlight Polygon on Map
    highlightSceneOnMap(sceneId, flyTo);

    // Update HUD
    if (hudElev) hudElev.textContent = `${scene.elevation_m} m MSL`;
    if (hudCoords) hudCoords.textContent = `${scene.lat.toFixed(4)}° N, ${scene.lon.toFixed(4)}° E`;

    // Update Studio Banner
    if (bannerSceneName) bannerSceneName.textContent = scene.name;
    if (bannerSceneCoords) {
      bannerSceneCoords.textContent = `${scene.lat.toFixed(4)}° N, ${scene.lon.toFixed(4)}° E • ${scene.district}, ${scene.state} • ${scene.elevation_m}m MSL`;
    }

    // Load imagery & metrics for selected scene
    loadSceneData(sceneId);
  }

  function highlightSceneOnMap(sceneId, flyTo = false) {
    Object.keys(polygonLayers).forEach((id) => {
      const poly = polygonLayers[id];
      const marker = markerLayers[id];
      if (!poly || !marker) return;

      if (id === sceneId) {
        poly.setStyle({
          color: "#0D4A3E",
          weight: 3,
          fillColor: "#0D4A3E",
          fillOpacity: 0.35,
          dashArray: null
        });
        marker.setStyle({
          radius: 9,
          color: "#ffffff",
          fillColor: "#0D4A3E",
          fillOpacity: 1.0,
          weight: 3
        });
        poly.bringToFront();
      } else {
        poly.setStyle({
          color: "#38bdf8",
          weight: 1.5,
          fillColor: "#38bdf8",
          fillOpacity: 0.12,
          dashArray: "4, 4"
        });
        marker.setStyle({
          radius: 6,
          color: "#38bdf8",
          fillColor: "#0284c7",
          fillOpacity: 0.85,
          weight: 2
        });
      }
    });

    if (flyTo && map && polygonLayers[sceneId]) {
      const bounds = polygonLayers[sceneId].getBounds();
      map.flyToBounds(bounds, { padding: [60, 60], duration: 1.0, maxZoom: 9 });
    }
  }

  // ==========================================================================
  // 6. Interactive Studio Comparison Slider Engine
  // ==========================================================================
  function setSliderSplitPercentage(pct) {
    pct = Math.max(0, Math.min(100, pct));
    if (layerAfter) {
      layerAfter.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    }
    if (sliderHandle) {
      sliderHandle.style.left = `${pct}%`;
    }
  }

  function setActiveViewToggle(activeBtn) {
    [btnViewSplit, btnViewRecon, btnViewStage1, btnViewStage2, btnViewCloudy].forEach(b => {
      if (b) b.classList.remove("active");
    });
    if (activeBtn) activeBtn.classList.add("active");
  }

  if (btnViewSplit) {
    btnViewSplit.addEventListener("click", () => {
      setActiveViewToggle(btnViewSplit);
      setSliderSplitPercentage(50);
      if (sliderHandle) sliderHandle.style.display = "flex";
    });
  }

  if (btnViewRecon) {
    btnViewRecon.addEventListener("click", () => {
      setActiveViewToggle(btnViewRecon);
      setSliderSplitPercentage(0); // 100% cloud-free reconstructed visible
      if (sliderHandle) sliderHandle.style.display = "none";
    });
  }

  if (btnViewStage1) {
    btnViewStage1.addEventListener("click", () => {
      setActiveViewToggle(btnViewStage1);
      if (sceneImages && (sceneImages.stage1_rgb || sceneImages.recon_rgb) && imgRecon) {
        imgRecon.src = sceneImages.stage1_rgb || sceneImages.recon_rgb;
      }
      setSliderSplitPercentage(0);
      if (sliderHandle) sliderHandle.style.display = "none";
    });
  }

  if (btnViewStage2) {
    btnViewStage2.addEventListener("click", () => {
      setActiveViewToggle(btnViewStage2);
      if (sceneImages && (sceneImages.stage2_refined_rgb || sceneImages.recon_rgb) && imgRecon) {
        imgRecon.src = sceneImages.stage2_refined_rgb || sceneImages.recon_rgb;
      }
      setSliderSplitPercentage(0);
      if (sliderHandle) sliderHandle.style.display = "none";
    });
  }

  if (btnViewCloudy) {
    btnViewCloudy.addEventListener("click", () => {
      setActiveViewToggle(btnViewCloudy);
      setSliderSplitPercentage(100); // 100% cloudy visible
      if (sliderHandle) sliderHandle.style.display = "none";
    });
  }

  function updateStudioSliderPosition(clientX) {
    if (!sliderContainer) return;
    const rect = sliderContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    percentage = Math.max(2, Math.min(98, percentage));
    
    if (layerAfter) {
      layerAfter.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    }
    if (sliderHandle) {
      sliderHandle.style.left = `${percentage}%`;
      sliderHandle.style.display = "flex";
    }
    setActiveViewToggle(btnViewSplit);
  }

  if (sliderContainer) {
    sliderContainer.addEventListener("mousedown", (e) => {
      isDraggingStudio = true;
      updateStudioSliderPosition(e.clientX);
    });

    sliderContainer.addEventListener("touchstart", (e) => {
      isDraggingStudio = true;
      updateStudioSliderPosition(e.touches[0].clientX);
    });
  }

  window.addEventListener("mouseup", () => {
    isDraggingStudio = false;
  });

  window.addEventListener("touchend", () => {
    isDraggingStudio = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingStudio) return;
    updateStudioSliderPosition(e.clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDraggingStudio) return;
    updateStudioSliderPosition(e.touches[0].clientX);
  });

  // ==========================================================================
  // 7. Load Scene Multi-Modal Imagery & Physical Metrics
  // ==========================================================================
  async function loadSceneData(sceneId = selectedSceneId) {
    try {
      const res = await fetch(`/api/scene-data?scene_id=${encodeURIComponent(sceneId)}`);
      const data = await res.json();
      
      if (data.status === "success") {
        sceneImages = data.images;
        
        // Update images
        if (imgCloudy && sceneImages.cloudy_rgb) imgCloudy.src = sceneImages.cloudy_rgb;
        if (imgRecon && sceneImages.recon_rgb) imgRecon.src = sceneImages.recon_rgb;

        // Update Scorecard
        if (metricPsnr) metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        if (metricSsim) metricSsim.textContent = data.metrics.ssim;
        if (metricSam) metricSam.textContent = `${data.metrics.sam_deg}°`;
        if (metricNdvi) metricNdvi.textContent = data.metrics.ndvi_mae;
        if (metricCbdi) metricCbdi.textContent = data.metrics.cbdi_seam;
        if (metricLatency) metricLatency.textContent = `${data.metrics.latency_ms} ms`;

        // Update Gemini Multimodal Assessment
        if (data.gemini_report && data.gemini_report.gemini_assessment && geminiAssessmentText) {
          geminiAssessmentText.textContent = data.gemini_report.gemini_assessment;
        }

        // Default slider to 50% split
        setActiveViewToggle(btnViewSplit);
        setSliderSplitPercentage(50);
        if (sliderHandle) sliderHandle.style.display = "flex";
      }
    } catch (err) {
      console.error("Failed to load scene data:", err);
    }
  }

  // ==========================================================================
  // 8. Custom Image Upload & 4-Stage Animated Pipeline Execution
  // ==========================================================================
  if (uploadDropzone && fileInputUpload) {
    uploadDropzone.addEventListener("click", () => fileInputUpload.click());

    uploadDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadDropzone.classList.add("dragover");
    });

    uploadDropzone.addEventListener("dragleave", () => {
      uploadDropzone.classList.remove("dragover");
    });

    uploadDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleImageUpload(e.dataTransfer.files[0]);
      }
    });

    fileInputUpload.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
      }
    });
  }

  function resetTrackerToReady() {
    if (stepCardMask) {
      stepCardMask.className = "stage-step-card";
      if (stepBadgeMask) stepBadgeMask.textContent = "Ready";
      if (stepDescMask) stepDescMask.textContent = "Whiteness + Luminance Haze Seal";
    }
    if (stepCardStage1) {
      stepCardStage1.className = "stage-step-card";
      if (stepBadgeStage1) stepBadgeStage1.textContent = "Ready";
      if (stepDescStage1) stepDescStage1.textContent = "SAR C-Band Radar + DEM Diffusion";
    }
    if (stepCardStage2) {
      stepCardStage2.className = "stage-step-card";
      if (stepBadgeStage2) stepBadgeStage2.textContent = "Ready";
      if (stepDescStage2) stepDescStage2.textContent = "Quad-Stack Multimodal Verification";
    }
    if (stepCardOutput) {
      stepCardOutput.className = "stage-step-card";
      if (stepBadgeOutput) stepBadgeOutput.textContent = "Ready";
      if (stepDescOutput) stepDescOutput.textContent = "Saved in results/ directory";
    }

    if (trackerPipelineBadge) trackerPipelineBadge.textContent = "⚡ 2-Stage Pipeline Ready";
    if (trackerSummaryText) trackerSummaryText.textContent = "Awaiting input satellite image...";
    if (trackerSavedBox) trackerSavedBox.style.display = "none";
    if (btnDownloadResult) btnDownloadResult.style.display = "none";
  }

  async function handleImageUpload(file) {
    if (!file) return;

    // 1. Immediately switch to Studio View so user witnesses execution
    switchView("studio");

    // Hide download button until all 4 stages finish
    if (btnDownloadResult) btnDownloadResult.style.display = "none";

    if (uploadStatus) uploadStatus.style.display = "flex";
    if (uploadStatusText) {
      uploadStatusText.innerHTML = `<span>⏳</span> Processing <strong>${file.name}</strong>: Running Stage 1 (CR-DiffNet SAR) ➔ Stage 2 (Gemini CLI Refiner)...`;
    }

    // 2. Animate Step 1: Optical Cloud Masker active
    if (stepCardMask) {
      stepCardMask.className = "stage-step-card active";
      if (stepBadgeMask) stepBadgeMask.textContent = "Scanning...";
      if (stepDescMask) stepDescMask.textContent = "Optical Penumbra & Cloud Detection";
    }
    if (stepCardStage1) {
      stepCardStage1.className = "stage-step-card";
      if (stepBadgeStage1) stepBadgeStage1.textContent = "Queued";
      if (stepDescStage1) stepDescStage1.textContent = "SAR C-Band Radar + DEM Inpainting";
    }
    if (stepCardStage2) {
      stepCardStage2.className = "stage-step-card";
      if (stepBadgeStage2) stepBadgeStage2.textContent = "Queued";
      if (stepDescStage2) stepDescStage2.textContent = "Quad-Stack Multimodal Verification";
    }
    if (stepCardOutput) {
      stepCardOutput.className = "stage-step-card";
      if (stepBadgeOutput) stepBadgeOutput.textContent = "Queued";
      if (stepDescOutput) stepDescOutput.textContent = "Saved in results/ directory";
    }

    if (trackerPipelineBadge) trackerPipelineBadge.textContent = "⚡ Running 2-Stage Pipeline...";
    if (trackerSummaryText) trackerSummaryText.textContent = `Executing Stage 1 on ${file.name}...`;
    if (trackerSavedBox) trackerSavedBox.style.display = "none";

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/upload-image", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      if (data.status === "success") {
        sceneImages = data.images;
        const s1Latency = (data.stages && data.stages.stage1) ? data.stages.stage1.latency_ms : 65;

        // 3. Mark Step 1 complete & Step 2 complete
        if (stepCardMask) {
          stepCardMask.className = "stage-step-card complete";
          if (stepBadgeMask) stepBadgeMask.textContent = `✓ ${data.cloud_coverage_pct}% Cloud`;
          if (stepDescMask) stepDescMask.textContent = "Optical Penumbra & Haze Masked";
        }

        if (stepCardStage1) {
          stepCardStage1.className = "stage-step-card complete";
          if (stepBadgeStage1) stepBadgeStage1.textContent = `✓ Done (${s1Latency}ms)`;
          if (stepDescStage1) stepDescStage1.textContent = "SAR Radar Conditioned Diffusion";
        }

        // 4. Highlight Step 3 (Stage 2: Gemini Refiner) as active
        if (stepCardStage2) {
          stepCardStage2.className = "stage-step-card active";
          if (stepBadgeStage2) stepBadgeStage2.textContent = "⏳ Gemini CLI Running";
          if (stepDescStage2) stepDescStage2.textContent = "Synthesizing micro-textures on local Mac...";
        }

        if (stepCardOutput) {
          stepCardOutput.className = "stage-step-card";
          if (stepBadgeOutput) stepBadgeOutput.textContent = "⏳ In Progress";
          if (stepDescOutput) stepDescOutput.textContent = "Awaiting Stage 2 refinement";
        }

        if (trackerPipelineBadge) trackerPipelineBadge.textContent = "⚡ Stage 1 Complete ➔ Gemini CLI Running...";
        if (trackerSummaryText) trackerSummaryText.textContent = `Stage 1 cloud removal finished in ${s1Latency}ms. Gemini CLI is synthesizing refined textures in background...`;

        // Update Scorecards
        if (metricPsnr) metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        if (metricSsim) metricSsim.textContent = data.metrics.ssim;
        if (metricSam) metricSam.textContent = `${data.metrics.sam_deg}°`;
        if (metricNdvi) metricNdvi.textContent = data.metrics.ndvi_mae;
        if (metricCbdi) metricCbdi.textContent = data.metrics.cbdi_seam;
        if (metricLatency) metricLatency.textContent = `${s1Latency} ms`;

        // Update Banner
        if (bannerSceneName) bannerSceneName.textContent = data.metadata.name;
        if (bannerSceneCoords) {
          bannerSceneCoords.textContent = `User Upload (${data.filename}) • Detected Cloud: ${data.cloud_coverage_pct}% • Stage 1 (CR-DiffNet) ➔ Stage 2 (Gemini CLI)`;
        }

        // Render images on canvas immediately
        if (imgCloudy && sceneImages.cloudy_rgb) imgCloudy.src = sceneImages.cloudy_rgb;
        if (imgRecon && sceneImages.recon_rgb) imgRecon.src = sceneImages.recon_rgb;

        // Show Stage 1 toggle button
        if (btnViewStage1) {
          btnViewStage1.style.display = "inline-block";
          btnViewStage1.textContent = `⚡ Stage 1 (${s1Latency}ms)`;
        }

        // Set slider to 50% split view
        setActiveViewToggle(btnViewSplit);
        setSliderSplitPercentage(50);
        if (sliderHandle) sliderHandle.style.display = "flex";

        if (uploadStatusText) {
          uploadStatusText.innerHTML = `<span>⚡</span> Stage 1 finished in <strong>${s1Latency}ms</strong>! Surface displayed. Gemini CLI is now synthesizing the refined cloud-free image...`;
        }

        // 5. Poll for Stage 2 Gemini CLI Native completion
        let pollCount = 0;
        const pollInterval = setInterval(async () => {
          pollCount++;
          try {
            const sRes = await fetch("/api/gemini-refine-status");
            const sData = await sRes.json();

            if (sData.status === "generating") {
              const elapsed = sData.elapsed_s || pollCount * 2;
              if (stepBadgeStage2) stepBadgeStage2.textContent = `⏳ Gemini CLI (${elapsed}s)`;
              if (geminiAssessmentText) {
                geminiAssessmentText.textContent = `Gemini CLI Native is running in the background on your Mac without API key. Synthesizing high-frequency ground textures... (Elapsed: ${elapsed}s)`;
              }
            } else if (sData.status === "complete") {
              clearInterval(pollInterval);
              const s2Ms = sData.latency_ms || 2500;

              // Step 3 Complete
              if (stepCardStage2) {
                stepCardStage2.className = "stage-step-card complete";
                if (stepBadgeStage2) stepBadgeStage2.textContent = `✓ Refined (${s2Ms}ms)`;
                if (stepDescStage2) stepDescStage2.textContent = "Gemini CLI Native (Zero API Key)";
              }

              // Step 4 Complete (All 4 stages completed!)
              if (stepCardOutput) {
                stepCardOutput.className = "stage-step-card complete";
                if (stepBadgeOutput) stepBadgeOutput.textContent = "✓ Stored";
                if (stepDescOutput) stepDescOutput.textContent = "results/gemini_refined_cloudfree.png";
              }

              if (trackerPipelineBadge) trackerPipelineBadge.textContent = "✅ All 4 Stages Complete";
              if (trackerSummaryText) trackerSummaryText.textContent = `Gemini CLI refined surface generated and saved. Both Stage 1 and Stage 2 available.`;
              if (trackerSavedBox) trackerSavedBox.style.display = "flex";
              if (trackerSavedPath) trackerSavedPath.textContent = "results/gemini_refined_cloudfree.png";

              // Update Reconstructed Image with Stage 2 Refinement
              if (sData.images && sData.images.stage2_refined) {
                sceneImages.stage2_refined_rgb = sData.images.stage2_refined;
                sceneImages.recon_rgb = sData.images.stage2_refined;
                if (imgRecon) imgRecon.src = sData.images.stage2_refined;
              }

              if (btnViewStage2) {
                btnViewStage2.style.display = "inline-block";
                btnViewStage2.textContent = `🤖 Stage 2 (Gemini Refined)`;
              }

              if (geminiAssessmentText && sData.assessment) {
                geminiAssessmentText.textContent = sData.assessment;
              }

              // ================================================================
              // REVEAL DOWNLOAD BUTTON: Only when all 4 stages are complete!
              // ================================================================
              if (btnDownloadResult) {
                btnDownloadResult.style.display = "inline-flex";
              }

              if (uploadStatusText) {
                uploadStatusText.innerHTML = `<span>✅</span> All 4 Stages Complete! Stage 1 (CR-DiffNet ~${s1Latency}ms) + Stage 2 (Gemini CLI Native) saved in <strong>results/gemini_refined_cloudfree.png</strong>`;
              }

              setTimeout(() => {
                if (uploadStatus) uploadStatus.style.display = "none";
              }, 6000);

            } else if (sData.status === "error") {
              clearInterval(pollInterval);
              console.warn("Gemini CLI polling ended:", sData.message);
              // Fallback: Still allow download of stage 1
              if (stepCardOutput) {
                stepCardOutput.className = "stage-step-card complete";
                if (stepBadgeOutput) stepBadgeOutput.textContent = "✓ Stored";
                if (stepDescOutput) stepDescOutput.textContent = "results/stage1_cloudfree_draft.png";
              }
              if (btnDownloadResult) {
                btnDownloadResult.style.display = "inline-flex";
              }
            }
          } catch (e) {
            console.warn("Poll note:", e);
          }
        }, 2000);

      } else {
        if (uploadStatusText) {
          uploadStatusText.textContent = `⚠️ Error: ${data.message || "Failed to process image"}`;
        }
      }
    } catch (err) {
      console.error("Upload error:", err);
      if (uploadStatusText) {
        uploadStatusText.textContent = "⚠️ Error uploading image. Please try another file.";
      }
    }
  }

  // ==========================================================================
  // 9. Initial Load Sequence
  // ==========================================================================
  async function bootstrapApp() {
    initMap();
    resetTrackerToReady();

    try {
      const res = await fetch("/api/training-scenes");
      const data = await res.json();
      if (data.status === "success" && data.scenes) {
        trainingScenes = data.scenes;
        renderTrainingScenesOnMap();
        selectScene(selectedSceneId, false);
      }
    } catch (err) {
      console.error("Failed to load training scenes metadata:", err);
    }
  }

  bootstrapApp();
});
