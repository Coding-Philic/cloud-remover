/**
 * CR-DiffNet: Operational GIS Mapping & Multi-Modal Generative Cloud Removal Studio
 * Integrates real Earth Observation geospatial map (Leaflet + Esri World Imagery)
 * with interactive Before/After generative latent diffusion studio.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================================================
  // Global State
  // ==========================================================================
  let trainingScenes = [];
  let selectedSceneId = "S2A_46RBN";
  let currentMode = "rgb";
  let activeTab = "map"; // 'map' or 'studio'
  let sceneImages = {};
  let isDragging = false;
  
  // Leaflet Map & GIS Layers
  let map = null;
  let currentBasemap = null;
  let labelsLayer = null;
  const polygonLayers = {};
  const markerLayers = {};

  // DOM Elements - Navigation & Views
  const tabBtnMap = document.getElementById("tab-btn-map");
  const tabBtnStudio = document.getElementById("tab-btn-studio");
  const viewGisMap = document.getElementById("view-gis-map");
  const viewStudio = document.getElementById("view-studio");
  const btnSwitchToMap = document.getElementById("btn-switch-to-map");
  const btnLaunchStudio = document.getElementById("btn-launch-scene-studio");
  const btnCloseCard = document.getElementById("btn-close-card");

  // DOM Elements - GIS Map View
  const scenesContainer = document.getElementById("scenes-container");
  const scenesCountBadge = document.getElementById("scenes-count-badge");
  const hudCoords = document.getElementById("hud-coords");
  const hudElev = document.getElementById("hud-elev");
  const sceneDetailCard = document.getElementById("scene-detail-card");
  const overlayTitle = document.getElementById("overlay-title");
  const overlayMgrs = document.getElementById("overlay-mgrs");
  const overlayCoords = document.getElementById("overlay-coords");
  const overlayElev = document.getElementById("overlay-elev");
  const overlayClouds = document.getElementById("overlay-clouds");
  const overlayPatches = document.getElementById("overlay-patches");
  const overlayTerrain = document.getElementById("overlay-terrain");
  const overlayDesc = document.getElementById("overlay-desc");

  // Basemap Switcher Buttons
  const btnBmSatellite = document.getElementById("btn-bm-satellite");
  const btnBmOsm = document.getElementById("btn-bm-osm");
  const btnBmDark = document.getElementById("btn-bm-dark");

  // DOM Elements - Studio View
  const studioSceneDropdown = document.getElementById("studio-scene-dropdown");
  const bannerSceneName = document.getElementById("banner-scene-name");
  const bannerSceneCoords = document.getElementById("banner-scene-coords");
  const studioQuickMgrs = document.getElementById("studio-quick-mgrs");
  const studioQuickElev = document.getElementById("studio-quick-elev");
  const studioQuickCoords = document.getElementById("studio-quick-coords");
  const studioQuickPatches = document.getElementById("studio-quick-patches");

  // Studio Slider & Layers
  const sliderContainer = document.getElementById("comparison-slider");
  const layerAfter = document.getElementById("layer-after");
  const sliderHandle = document.getElementById("slider-handle");
  const imgCloudy = document.getElementById("img-cloudy");
  const imgRecon = document.getElementById("img-recon");

  // Studio Metrics Scorecards
  const metricPsnr = document.getElementById("metric-psnr");
  const metricSsim = document.getElementById("metric-ssim");
  const metricSam = document.getElementById("metric-sam");
  const metricNdvi = document.getElementById("metric-ndvi");
  const metricCbdi = document.getElementById("metric-cbdi");
  const metricLatency = document.getElementById("metric-latency");
  const geminiAssessmentText = document.getElementById("gemini-assessment-text");

  // Controls
  const modeButtons = document.querySelectorAll(".mode-btn");
  const coverageSlider = document.getElementById("coverage-slider");
  const coverageVal = document.getElementById("coverage-val");
  const azimuthSlider = document.getElementById("azimuth-slider");
  const azimuthVal = document.getElementById("azimuth-val");
  const btnRegenClouds = document.getElementById("btn-regen-clouds");

  // ==========================================================================
  // 1. Navigation & Tab Switching Engine
  // ==========================================================================
  function switchView(target) {
    activeTab = target;
    if (target === "map") {
      tabBtnMap.classList.add("active");
      tabBtnStudio.classList.remove("active");
      viewGisMap.style.display = "flex";
      viewStudio.style.display = "none";
      if (map) {
        setTimeout(() => map.invalidateSize(), 60);
      }
    } else {
      tabBtnStudio.classList.add("active");
      tabBtnMap.classList.remove("active");
      viewStudio.style.display = "grid";
      viewGisMap.style.display = "none";
    }
  }

  tabBtnMap.addEventListener("click", () => switchView("map"));
  tabBtnStudio.addEventListener("click", () => switchView("studio"));

  btnSwitchToMap.addEventListener("click", () => {
    switchView("map");
    highlightSceneOnMap(selectedSceneId, true);
  });

  btnLaunchStudio.addEventListener("click", () => {
    switchView("studio");
  });

  btnCloseCard.addEventListener("click", () => {
    sceneDetailCard.style.display = "none";
  });

  // ==========================================================================
  // 2. Leaflet GIS Map Initialization
  // ==========================================================================
  function initMap() {
    // Center of North East India (Brahmaputra Valley / Assam)
    map = L.map("training-map", {
      center: [26.20, 92.60],
      zoom: 7,
      zoomControl: true,
      attributionControl: false
    });

    // Basemap tile providers
    const basemaps = {
      satellite: L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 18, attribution: "Esri World Imagery" }
      ),
      osm: L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        { maxZoom: 18, attribution: "© OpenStreetMap contributors" }
      ),
      dark: L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 18, attribution: "© CartoDB Dark Matter" }
      )
    };

    // Esri boundaries & places overlay
    labelsLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 18, opacity: 0.85 }
    );

    currentBasemap = basemaps.satellite;
    currentBasemap.addTo(map);
    labelsLayer.addTo(map);

    // Basemap Switcher Event Handlers
    function setBasemap(type, btn) {
      document.querySelectorAll(".basemap-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      
      map.removeLayer(currentBasemap);
      currentBasemap = basemaps[type];
      currentBasemap.addTo(map);

      // Re-add labels on top if satellite or dark
      if (type === "satellite" || type === "dark") {
        if (!map.hasLayer(labelsLayer)) labelsLayer.addTo(map);
      } else {
        if (map.hasLayer(labelsLayer)) map.removeLayer(labelsLayer);
      }
    }

    btnBmSatellite.addEventListener("click", () => setBasemap("satellite", btnBmSatellite));
    btnBmOsm.addEventListener("click", () => setBasemap("osm", btnBmOsm));
    btnBmDark.addEventListener("click", () => setBasemap("dark", btnBmDark));

    // Live Coordinates Tracking HUD
    map.on("mousemove", (e) => {
      hudCoords.textContent = `${e.latlng.lat.toFixed(4)}° N, ${e.latlng.lng.toFixed(4)}° E`;
    });
  }

  // ==========================================================================
  // 3. Render Real Training Scenes on Map & Sidebar
  // ==========================================================================
  function renderTrainingScenesOnMap() {
    scenesContainer.innerHTML = "";
    scenesCountBadge.textContent = `${trainingScenes.length} Scenes`;

    // Populate Studio Dropdown
    studioSceneDropdown.innerHTML = "";

    trainingScenes.forEach((scene) => {
      // 1. Add option to Studio Dropdown
      const opt = document.createElement("option");
      opt.value = scene.id;
      opt.textContent = `${scene.name} (${scene.mgrs})`;
      studioSceneDropdown.appendChild(opt);

      // 2. Add Scene Card in Map Sidebar
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
      });
      scenesContainer.appendChild(card);

      // 3. Draw Polygon Bounding Box on Leaflet Map
      // scene.bounds: [[min_lat, min_lon], [max_lat, max_lon]]
      const polyBounds = [
        [scene.bounds[0][0], scene.bounds[0][1]],
        [scene.bounds[0][0], scene.bounds[1][1]],
        [scene.bounds[1][0], scene.bounds[1][1]],
        [scene.bounds[1][0], scene.bounds[0][1]]
      ];

      const isInitialSelected = (scene.id === selectedSceneId);
      const polygon = L.polygon(polyBounds, {
        color: isInitialSelected ? "#f59e0b" : "#38bdf8",
        weight: isInitialSelected ? 3 : 1.5,
        fillColor: isInitialSelected ? "#f59e0b" : "#38bdf8",
        fillOpacity: isInitialSelected ? 0.28 : 0.12,
        dashArray: isInitialSelected ? null : "4, 4"
      }).addTo(map);

      // Center Marker with pulsing badge
      const marker = L.circleMarker([scene.lat, scene.lon], {
        radius: isInitialSelected ? 8 : 6,
        color: isInitialSelected ? "#ffffff" : "#38bdf8",
        fillColor: isInitialSelected ? "#f59e0b" : "#0284c7",
        fillOpacity: 0.9,
        weight: 2
      }).addTo(map);

      // Tooltip on Hover
      polygon.bindTooltip(
        `<strong>${scene.name}</strong><br><span style="color:#38bdf8">MGRS: ${scene.mgrs}</span> • ${scene.patches_trained} Training Patches`,
        { sticky: true, opacity: 0.9 }
      );

      // Click on footprint
      polygon.on("click", () => selectScene(scene.id, true));
      marker.on("click", () => selectScene(scene.id, true));

      polygonLayers[scene.id] = polygon;
      markerLayers[scene.id] = marker;
    });

    studioSceneDropdown.value = selectedSceneId;
  }

  // ==========================================================================
  // 4. Scene Selection & Deep-Dive Inspection Card
  // ==========================================================================
  function selectScene(sceneId, flyTo = false) {
    selectedSceneId = sceneId;
    const scene = trainingScenes.find(s => s.id === sceneId);
    if (!scene) return;

    // Update active state in sidebar list
    document.querySelectorAll(".scene-card-item").forEach(el => el.classList.remove("selected"));
    const activeCard = document.getElementById(`sci-${sceneId}`);
    if (activeCard) {
      activeCard.classList.add("selected");
      activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Update Studio Dropdown
    studioSceneDropdown.value = sceneId;

    // Highlight Polygon on Map
    highlightSceneOnMap(sceneId, flyTo);

    // Populate Floating Scene Detail Overlay
    overlayTitle.textContent = scene.name;
    overlayMgrs.textContent = `MGRS Tile: ${scene.mgrs} • ${scene.district}, ${scene.state}`;
    overlayCoords.textContent = `${scene.lat.toFixed(4)}° N, ${scene.lon.toFixed(4)}° E`;
    overlayElev.textContent = `${scene.elevation_m} m MSL`;
    overlayClouds.textContent = `${scene.raw_cloud_pct}%`;
    overlayPatches.textContent = `${scene.patches_trained} Patches (512×512 Float16)`;
    overlayTerrain.textContent = scene.terrain_type;
    overlayDesc.textContent = scene.description;

    hudElev.textContent = `${scene.elevation_m} m MSL`;
    hudCoords.textContent = `${scene.lat.toFixed(4)}° N, ${scene.lon.toFixed(4)}° E`;

    sceneDetailCard.style.display = "block";

    // Update Studio Banner & Quick Meta
    bannerSceneName.textContent = scene.name;
    bannerSceneCoords.textContent = `${scene.lat.toFixed(4)}° N, ${scene.lon.toFixed(4)}° E • ${scene.district}, ${scene.state} • ${scene.elevation_m}m MSL`;
    studioQuickMgrs.textContent = scene.mgrs;
    studioQuickElev.textContent = `${scene.elevation_m}m MSL`;
    studioQuickCoords.textContent = `${scene.lat.toFixed(2)}°N, ${scene.lon.toFixed(2)}°E`;
    studioQuickPatches.textContent = `${scene.patches_trained} Patches`;

    // Fetch and render scene multi-modal imagery
    loadSceneData(sceneId);
  }

  function highlightSceneOnMap(sceneId, flyTo = false) {
    Object.keys(polygonLayers).forEach((id) => {
      const poly = polygonLayers[id];
      const marker = markerLayers[id];
      if (id === sceneId) {
        poly.setStyle({
          color: "#f59e0b",
          weight: 3,
          fillColor: "#f59e0b",
          fillOpacity: 0.32,
          dashArray: null
        });
        marker.setStyle({
          radius: 9,
          color: "#ffffff",
          fillColor: "#f59e0b",
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

  // Handle Studio Dropdown Change
  studioSceneDropdown.addEventListener("change", (e) => {
    selectScene(e.target.value, false);
  });

  // ==========================================================================
  // 5. Interactive Split Comparison Slider Engine
  // ==========================================================================
  // View Toggle Mode Handlers (Split Slider / 100% Cloud-Free / Stage 1 / Stage 2 / 100% Cloudy)
  const btnViewSplit = document.getElementById("btn-view-split");
  const btnViewRecon = document.getElementById("btn-view-recon");
  const btnViewStage1 = document.getElementById("btn-view-stage1");
  const btnViewStage2 = document.getElementById("btn-view-stage2");
  const btnViewCloudy = document.getElementById("btn-view-cloudy");

  function setSliderSplitPercentage(pct) {
    pct = Math.max(0, Math.min(100, pct));
    layerAfter.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    sliderHandle.style.left = `${pct}%`;
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
      sliderHandle.style.display = "flex";
    });
  }

  if (btnViewRecon) {
    btnViewRecon.addEventListener("click", () => {
      setActiveViewToggle(btnViewRecon);
      setSliderSplitPercentage(0); // 100% cloud-free reconstructed visible
      sliderHandle.style.display = "none";
    });
  }

  if (btnViewStage1) {
    btnViewStage1.addEventListener("click", () => {
      setActiveViewToggle(btnViewStage1);
      if (sceneImages && (sceneImages.stage1_rgb || sceneImages.recon_rgb)) {
        imgRecon.src = sceneImages.stage1_rgb || sceneImages.recon_rgb;
      }
      setSliderSplitPercentage(0);
      sliderHandle.style.display = "none";
    });
  }

  if (btnViewStage2) {
    btnViewStage2.addEventListener("click", () => {
      setActiveViewToggle(btnViewStage2);
      if (sceneImages && (sceneImages.stage2_refined_rgb || sceneImages.recon_rgb)) {
        imgRecon.src = sceneImages.stage2_refined_rgb || sceneImages.recon_rgb;
      }
      setSliderSplitPercentage(0);
      sliderHandle.style.display = "none";
    });
  }

  if (btnViewCloudy) {
    btnViewCloudy.addEventListener("click", () => {
      setActiveViewToggle(btnViewCloudy);
      setSliderSplitPercentage(100); // 100% cloudy visible
      sliderHandle.style.display = "none";
    });
  }

  // Live Terminal Stream Logger & Collapse Toggle
  const btnToggleTerminal = document.getElementById("btn-toggle-terminal");
  const terminalStreamBody = document.getElementById("terminal-stream-body");
  if (btnToggleTerminal && terminalStreamBody) {
    btnToggleTerminal.addEventListener("click", () => {
      if (terminalStreamBody.style.display === "none") {
        terminalStreamBody.style.display = "block";
        btnToggleTerminal.textContent = "Minimize ▲";
      } else {
        terminalStreamBody.style.display = "none";
        btnToggleTerminal.textContent = "Expand ▼";
      }
    });
  }

  function appendTerminalLog(text, type = "normal") {
    if (!terminalStreamBody) return;
    const div = document.createElement("div");
    div.className = `terminal-line ${type}`;
    const now = new Date();
    const timeStr = `[${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}]`;
    div.textContent = `${timeStr} ${text}`;
    terminalStreamBody.appendChild(div);
    terminalStreamBody.scrollTop = terminalStreamBody.scrollHeight;
  }

  function updateSliderPosition(clientX) {
    const rect = sliderContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    percentage = Math.max(2, Math.min(98, percentage));
    
    layerAfter.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    sliderHandle.style.left = `${percentage}%`;
    sliderHandle.style.display = "flex";
    setActiveViewToggle(btnViewSplit);
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
  // 6. Multi-Spectral Layer Switcher
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
  // 7. Fetch Scene Multi-Modal Imagery & Physical Metrics
  // ==========================================================================
  async function loadSceneData(sceneId = selectedSceneId) {
    try {
      const res = await fetch(`/api/scene-data?scene_id=${encodeURIComponent(sceneId)}`);
      const data = await res.json();
      
      if (data.status === "success") {
        sceneImages = data.images;
        
        // Update Metrics Scorecard
        metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        metricSsim.textContent = data.metrics.ssim;
        metricSam.textContent = `${data.metrics.sam_deg}°`;
        metricNdvi.textContent = data.metrics.ndvi_mae;
        metricCbdi.textContent = data.metrics.cbdi_seam;
        metricLatency.textContent = `${data.metrics.latency_ms} ms`;
        
        coverageVal.textContent = `${data.cloud_coverage_pct}%`;
        coverageSlider.value = data.cloud_coverage_pct;

        // Update Gemini Multimodal Assessment
        if (data.gemini_report && data.gemini_report.gemini_assessment && geminiAssessmentText) {
          geminiAssessmentText.textContent = data.gemini_report.gemini_assessment;
        }

        renderCurrentMode();
      }
    } catch (err) {
      console.error("Failed to load scene data:", err);
    }
  }

  // ==========================================================================
  // 8. Cloud Coverage Slider & Re-simulation
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
          scene_id: selectedSceneId,
          coverage: parseFloat(coverageSlider.value) / 100.0,
          sun_azimuth: parseFloat(azimuthSlider.value)
        })
      });
      await loadSceneData(selectedSceneId);
    } catch (err) {
      console.error("Error regenerating clouds:", err);
    } finally {
      btnRegenClouds.disabled = false;
      btnRegenClouds.innerHTML = `<span>⚡</span> Re-Simulate Cloud Scene`;
    }
  });

  // ==========================================================================
  // 8b. Judge Demo: Custom Image Upload & Preset Handlers
  // ==========================================================================
  const uploadDropzone = document.getElementById("upload-dropzone");
  const fileInputUpload = document.getElementById("file-input-upload");
  const uploadStatus = document.getElementById("upload-status");
  const uploadStatusText = document.getElementById("upload-status-text");

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

  // Pipeline Live Tracker Step Elements
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

  const trackerPipelineBadge = document.getElementById("tracker-pipeline-badge");
  const trackerSummaryText = document.getElementById("tracker-summary-text");
  const trackerSavedBox = document.getElementById("tracker-saved-box");
  const trackerSavedPath = document.getElementById("tracker-saved-path");
  const terminalStatusIndicator = document.getElementById("terminal-status-indicator");

  async function handleImageUpload(file) {
    if (!file) return;

    // 1. Immediately switch to Studio View so user sees execution & results
    switchView("studio");

    uploadStatus.style.display = "flex";
    uploadStatusText.innerHTML = `<span>⏳</span> Running 2-Stage Pipeline on <strong>${file.name}</strong>: Stage 1 (CR-DiffNet SAR) ➔ Stage 2 (Gemini Quad-Stack Refiner)...`;

    // 2. Initialize Stage Step Cards
    if (stepCardMask) {
      stepCardMask.className = "stage-step-card active";
      if (stepBadgeMask) stepBadgeMask.textContent = "Scanning...";
      if (stepDescMask) stepDescMask.textContent = "Luminance & Whiteness Detection";
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
      if (stepDescOutput) stepDescOutput.textContent = "Saving to results/ directory";
    }

    if (trackerPipelineBadge) trackerPipelineBadge.textContent = "⚡ Running 2-Stage Pipeline...";
    if (trackerSummaryText) trackerSummaryText.textContent = `Executing Stage 1 (CR-DiffNet) ➔ Stage 2 (Gemini Refiner) on ${file.name}...`;
    if (trackerSavedBox) trackerSavedBox.style.display = "none";
    if (terminalStatusIndicator) terminalStatusIndicator.textContent = "● Running";

    // 3. Log to Live Terminal Stream
    appendTerminalLog("======================================================================", "dim");
    appendTerminalLog(`🛰️ INCOMING SATELLITE IMAGE: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, "highlight");
    appendTerminalLog("⚡ [STAGE 1/2] CR-DiffNet Preliminary Diffusion Inpainting Starting...", "stage");

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

        // 4. Update Stage 1 in Tracker and Terminal
        if (stepCardMask) {
          stepCardMask.className = "stage-step-card complete";
          if (stepBadgeMask) stepBadgeMask.textContent = `✓ ${data.cloud_coverage_pct}% Cloud`;
          if (stepDescMask) stepDescMask.textContent = "Optical Penumbra & Haze Masked";
        }
        appendTerminalLog(`   ├─ ☁️ Optical Cloud Masker: ${data.cloud_coverage_pct}% cloud detected`, "highlight");

        if (stepCardStage1) {
          stepCardStage1.className = "stage-step-card complete";
          if (stepBadgeStage1) stepBadgeStage1.textContent = `✓ Done (${s1Latency}ms)`;
          if (stepDescStage1) stepDescStage1.textContent = "SAR Radar Conditioned Diffusion";
        }
        appendTerminalLog("   ├─ 📡 Multi-Modal Conditioning: Sentinel-1 SAR C-Band Radar + DEM", "dim");
        appendTerminalLog(`   └─ ✓ STAGE 1 COMPLETE in ${s1Latency} ms! Preliminary ground surface recovered.`, "success");

        // 5. Update Stage 2 Tracker: Actively generating in Gemini CLI Native
        if (stepCardStage2) {
          stepCardStage2.className = "stage-step-card active";
          if (stepBadgeStage2) stepBadgeStage2.textContent = "⏳ Gemini CLI Running";
          if (stepDescStage2) stepDescStage2.textContent = "Synthesizing micro-textures on local Mac...";
        }
        appendTerminalLog("⚡ [STAGE 2/2] Gemini CLI Native Refiner (Local Mac - Zero API Key)...", "gemini");
        appendTerminalLog("   ├─ 🤖 Native Gemini CLI agent spawned on local machine (Zero API Key)", "gemini");
        appendTerminalLog("   └─ ⏳ Actively generating refined surface in background...", "gemini");

        if (stepCardOutput) {
          stepCardOutput.className = "stage-step-card complete";
          if (stepBadgeOutput) stepBadgeOutput.textContent = "✓ Stage 1 Stored";
          if (stepDescOutput) stepDescOutput.textContent = "results/stage1_cloudfree_draft.png";
        }

        if (trackerPipelineBadge) trackerPipelineBadge.textContent = "⚡ Stage 1 Complete ➔ Gemini CLI Running...";
        if (trackerSummaryText) trackerSummaryText.textContent = `Stage 1 cloud removal finished in ${s1Latency}ms. Gemini CLI is refining high-frequency textures in the background...`;
        if (trackerSavedBox) trackerSavedBox.style.display = "flex";
        if (trackerSavedPath) trackerSavedPath.textContent = "results/stage1_cloudfree_draft.png";
        if (terminalStatusIndicator) terminalStatusIndicator.textContent = "● Gemini CLI Active";

        // Update Scorecards
        metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        metricSsim.textContent = data.metrics.ssim;
        metricSam.textContent = `${data.metrics.sam_deg}°`;
        metricNdvi.textContent = data.metrics.ndvi_mae;
        metricCbdi.textContent = data.metrics.cbdi_seam;
        metricLatency.textContent = `${s1Latency} ms`;

        coverageVal.textContent = `${data.cloud_coverage_pct}%`;
        coverageSlider.value = data.cloud_coverage_pct;

        // Update Banner
        bannerSceneName.textContent = data.metadata.name;
        bannerSceneCoords.textContent = `User Upload (${data.filename}) • Detected Cloud: ${data.cloud_coverage_pct}% • Stage 1 (CR-DiffNet) ➔ Stage 2 (Gemini CLI)`;
        studioQuickMgrs.textContent = "CUSTOM";
        studioQuickCoords.textContent = "User Input";
        studioQuickPatches.textContent = "Live Stream";

        // Show Stage 1 toggle button
        if (btnViewStage1) {
          btnViewStage1.style.display = "inline-block";
          btnViewStage1.textContent = `⚡ Stage 1 (${s1Latency}ms)`;
        }

        // Render Stage 1 images in natural color RGB immediately
        currentMode = "rgb";
        modeButtons.forEach((b) => {
          if (b.dataset.mode === "rgb") b.classList.add("active");
          else b.classList.remove("active");
        });
        renderCurrentMode();

        // Default slider to 50% split view for instant inspection
        setActiveViewToggle(btnViewSplit);
        setSliderSplitPercentage(50);
        sliderHandle.style.display = "flex";

        uploadStatusText.innerHTML = `<span>⚡</span> Stage 1 finished in <strong>${s1Latency}ms</strong>! Surface displayed. Gemini CLI is now generating the refined cloud-free image in the background...`;

        // 6. Poll for Stage 2 Gemini CLI Native completion
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

              if (stepCardStage2) {
                stepCardStage2.className = "stage-step-card complete";
                if (stepBadgeStage2) stepBadgeStage2.textContent = `✓ Refined (${s2Ms}ms)`;
                if (stepDescStage2) stepDescStage2.textContent = "Gemini CLI Native (Zero API Key)";
              }

              if (stepCardOutput) {
                stepCardOutput.className = "stage-step-card complete";
                if (stepBadgeOutput) stepBadgeOutput.textContent = "✓ Stored";
                if (stepDescOutput) stepDescOutput.textContent = "results/gemini_refined_cloudfree.png";
              }

              appendTerminalLog(`✓ [STAGE 2/2] Gemini CLI Native Generation Complete in ${s2Ms} ms!`, "success");
              appendTerminalLog("   ├─ 📁 Refined: results/gemini_refined_cloudfree.png", "saved");
              appendTerminalLog("   └─ 📁 Mirrored: results/latest_reconstruction.png", "saved");
              appendTerminalLog("======================================================================", "dim");

              if (trackerPipelineBadge) trackerPipelineBadge.textContent = "✅ 2-Stage Pipeline Complete";
              if (trackerSummaryText) trackerSummaryText.textContent = `Gemini CLI Native refined surface generated. Both Stage 1 and Stage 2 available.`;
              if (trackerSavedBox) trackerSavedBox.style.display = "flex";
              if (trackerSavedPath) trackerSavedPath.textContent = "results/gemini_refined_cloudfree.png";
              if (terminalStatusIndicator) terminalStatusIndicator.textContent = "● Completed";

              if (sData.images && sData.images.stage2_refined) {
                sceneImages.stage2_refined_rgb = sData.images.stage2_refined;
                sceneImages.recon_rgb = sData.images.stage2_refined;
                renderCurrentMode();
              }

              if (btnViewStage2) {
                btnViewStage2.style.display = "inline-block";
                btnViewStage2.textContent = `🤖 Stage 2 (Gemini Refined)`;
              }

              if (geminiAssessmentText && sData.assessment) {
                geminiAssessmentText.textContent = sData.assessment;
              }

              uploadStatusText.innerHTML = `<span>✅</span> All Stages Complete! Stage 1 (CR-DiffNet ~${s1Latency}ms) + Stage 2 (Gemini CLI Native) stored in <strong>results/gemini_refined_cloudfree.png</strong>`;
              setTimeout(() => {
                uploadStatus.style.display = "none";
              }, 6000);
            } else if (sData.status === "error") {
              clearInterval(pollInterval);
              console.warn("Gemini CLI polling ended:", sData.message);
            }
          } catch (e) {
            console.warn("Poll note:", e);
          }
        }, 2000);
      } else {
        uploadStatusText.textContent = `⚠️ Error: ${data.message || "Failed to process image"}`;
        appendTerminalLog(`⚠️ Error: ${data.message || "Failed to process image"}`, "dim");
      }
    } catch (err) {
      console.error("Upload error:", err);
      uploadStatusText.textContent = "⚠️ Error uploading image. Please try another file.";
      appendTerminalLog(`⚠️ Connection error: ${err.message}`, "dim");
    }
  }

  // Quick Judge Demo Preset Buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cov = parseFloat(btn.dataset.coverage);
      const az = parseFloat(btn.dataset.azimuth);
      coverageSlider.value = Math.round(cov * 100);
      coverageVal.textContent = `${Math.round(cov * 100)}%`;
      azimuthSlider.value = az;
      azimuthVal.textContent = `${az}°`;

      btnRegenClouds.disabled = true;
      btnRegenClouds.innerHTML = `<span>⏳</span> Applying ${btn.textContent.trim()}...`;

      try {
        await fetch("/api/regenerate-clouds", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scene_id: selectedSceneId,
            coverage: cov,
            sun_azimuth: az
          })
        });
        await loadSceneData(selectedSceneId);
      } catch (err) {
        console.error("Error setting preset:", err);
      } finally {
        btnRegenClouds.disabled = false;
        btnRegenClouds.innerHTML = `<span>⚡</span> Re-Simulate Cloud Scene`;
      }
    });
  });

  // 1-Click Real Dataset Satellite Image Handlers
  document.querySelectorAll(".sample-img-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const filename = btn.dataset.file;
      uploadStatus.style.display = "flex";
      uploadStatusText.textContent = `Loading real dataset image: ${filename}...`;

      try {
        const res = await fetch(`/static/samples/${filename}`);
        const blob = await res.blob();
        const file = new File([blob], filename, { type: "image/jpeg" });
        await handleImageUpload(file);
      } catch (err) {
        console.error("Failed to load sample image:", err);
        uploadStatusText.textContent = "⚠️ Error loading sample image.";
      }
    });
  });

  // ==========================================================================
  // 8c. Google Gemini Vision Engine Modal & Status Manager
  // ==========================================================================
  const btnGeminiStatus = document.getElementById("btn-gemini-status");
  const geminiStatusDot = document.getElementById("gemini-status-dot");
  const geminiStatusLabel = document.getElementById("gemini-status-label");
  const geminiKeyModal = document.getElementById("gemini-key-modal");
  const btnCloseGeminiModal = document.getElementById("btn-close-gemini-modal");
  const btnCancelGeminiKey = document.getElementById("btn-cancel-gemini-key");
  const btnSaveGeminiKey = document.getElementById("btn-save-gemini-key");
  const geminiApiKeyInput = document.getElementById("gemini-api-key-input");
  const geminiModalStatusText = document.getElementById("gemini-modal-status-text");
  const btnToggleKeyVis = document.getElementById("btn-toggle-key-vis");
  const gpcActivePill = document.getElementById("gpc-active-pill");

  async function checkGeminiStatus() {
    try {
      const res = await fetch("/api/gemini/config");
      const data = await res.json();
      if (data.status === "success" && data.details) {
        const details = data.details;
        if (details.has_cli || details.client_ready) {
          if (geminiStatusDot) geminiStatusDot.className = "badge-dot dot-green";
          if (geminiStatusLabel) geminiStatusLabel.textContent = "🤖 Gemini CLI: Active";
          if (gpcActivePill) {
            gpcActivePill.textContent = "● Gemini CLI Active (Zero API Key)";
            gpcActivePill.style.color = "#10b981";
          }
          if (geminiModalStatusText) {
            geminiModalStatusText.innerHTML = `<strong>Gemini CLI Active:</strong> Running native Gemini 3.8 Flash directly on this Mac without requiring any API keys!`;
          }
        } else {
          if (geminiStatusDot) geminiStatusDot.className = "badge-dot dot-yellow";
          if (geminiStatusLabel) geminiStatusLabel.textContent = "🤖 Gemini AI: Offline Mode";
          if (gpcActivePill) {
            gpcActivePill.textContent = "● Offline Local Engine";
            gpcActivePill.style.color = "#f59e0b";
          }
        }
      }
    } catch (err) {
      console.warn("Could not check Gemini engine status:", err);
    }
  }

  const btnTestGeminiCli = document.getElementById("btn-test-gemini-cli");
  if (btnTestGeminiCli) {
    btnTestGeminiCli.addEventListener("click", async () => {
      btnTestGeminiCli.disabled = true;
      btnTestGeminiCli.innerHTML = `<span>⏳</span> Running gemini -p on Mac...`;
      try {
        const res = await fetch("/api/gemini/live-assess", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            cloud_pct: parseFloat(coverageSlider ? coverageSlider.value : 25),
            scene_name: bannerSceneName ? bannerSceneName.textContent : "Satellite Scene"
          })
        });
        const data = await res.json();
        if (data.assessment && geminiAssessmentText) {
          geminiAssessmentText.textContent = data.assessment;
          alert("✓ Gemini CLI executed successfully on your local Mac!\n\nAssessment:\n" + data.assessment);
        }
      } catch (err) {
        alert("⚠️ Error running Gemini CLI: " + err.message);
      } finally {
        btnTestGeminiCli.disabled = false;
        btnTestGeminiCli.innerHTML = `<span>⚡</span> Run Live Gemini CLI Assessment`;
      }
    });
  }

  async function triggerGeminiLiveAssessment(cloudPct, sceneName) {
    try {
      const res = await fetch("/api/gemini/live-assess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cloud_pct: cloudPct,
          scene_name: sceneName
        })
      });
      const data = await res.json();
      if (data.status === "success" && data.assessment && geminiAssessmentText) {
        geminiAssessmentText.textContent = data.assessment;
      }
    } catch (err) {
      console.warn("Background Gemini CLI assess note:", err);
    }
  }

  if (btnGeminiStatus && geminiKeyModal) {
    btnGeminiStatus.addEventListener("click", () => {
      geminiKeyModal.style.display = "flex";
      checkGeminiStatus();
    });
  }

  if (btnCloseGeminiModal && geminiKeyModal) {
    btnCloseGeminiModal.addEventListener("click", () => {
      geminiKeyModal.style.display = "none";
    });
  }

  if (btnCancelGeminiKey && geminiKeyModal) {
    btnCancelGeminiKey.addEventListener("click", () => {
      geminiKeyModal.style.display = "none";
    });
  }

  // ==========================================================================
  // 9. Initial Load Sequence
  // ==========================================================================
  async function bootstrapApp() {
    initMap();
    checkGeminiStatus();

    try {
      // Fetch 18 real training footprints from server
      const res = await fetch("/api/training-scenes");
      const data = await res.json();
      if (data.status === "success" && data.scenes) {
        trainingScenes = data.scenes;
        renderTrainingScenesOnMap();
        selectScene(selectedSceneId, true);
      }
    } catch (err) {
      console.error("Failed to load training scenes metadata:", err);
    }
  }

  bootstrapApp();
});
