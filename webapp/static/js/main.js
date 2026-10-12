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
  function updateSliderPosition(clientX) {
    const rect = sliderContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    percentage = Math.max(2, Math.min(98, percentage));
    
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

  async function handleImageUpload(file) {
    if (!file) return;
    uploadStatus.style.display = "flex";
    uploadStatusText.textContent = `Analyzing ${file.name}... Detecting cloud boundaries & running CR-DiffNet...`;

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

        // Update Scorecards
        metricPsnr.textContent = `${data.metrics.psnr_db} dB`;
        metricSsim.textContent = data.metrics.ssim;
        metricSam.textContent = `${data.metrics.sam_deg}°`;
        metricNdvi.textContent = data.metrics.ndvi_mae;
        metricCbdi.textContent = data.metrics.cbdi_seam;
        metricLatency.textContent = `${data.metrics.latency_ms} ms`;

        coverageVal.textContent = `${data.cloud_coverage_pct}%`;
        coverageSlider.value = data.cloud_coverage_pct;

        // Update Banner
        bannerSceneName.textContent = data.metadata.name;
        bannerSceneCoords.textContent = `User Upload (${data.filename}) • Detected Cloud: ${data.cloud_coverage_pct}% • Zero-Shot Generative Inpainting`;
        studioQuickMgrs.textContent = "CUSTOM";
        studioQuickCoords.textContent = "User Input";
        studioQuickPatches.textContent = "Live Stream";

        renderCurrentMode();

        uploadStatusText.textContent = `✓ Successfully reconstructed ${file.name}! Detected ${data.cloud_coverage_pct}% cloud cover.`;
        setTimeout(() => {
          uploadStatus.style.display = "none";
        }, 4000);
      } else {
        uploadStatusText.textContent = `⚠️ Error: ${data.message || "Failed to process image"}`;
      }
    } catch (err) {
      console.error("Upload error:", err);
      uploadStatusText.textContent = "⚠️ Error uploading image. Please try another file.";
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
  // 9. Initial Load Sequence
  // ==========================================================================
  async function bootstrapApp() {
    initMap();

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
