(() => {
  const INPUT_STATE_KEY = "figureweave_input_state_v1";
  const GEMINI_IMAGE_MODELS = {
    balanced: "gemini-3.1-flash-image-preview",
    max_quality: "gemini-3-pro-image-preview",
  };
  const SAM_PROMPT_PRESETS = {
    simple_flowchart: "plot,chart,heatmap,matrix,image",
    complex_paper: "module,block,encoder,head,panel,plot,heatmap,matrix",
  };
  const IMAGE_PROVIDER_CONFIGS = {
    gemini: {
      apiPlaceholder: "AIza...",
      imageProvider: "gemini",
      hint: "Google Gemini official API. Best fit when you want Gemini image generation modes directly in this app.",
    },
    openai: {
      apiPlaceholder: "sk-proj-...",
      imageProvider: "openai",
      imageModel: "gpt-4.1",
      hint: "OpenAI official API. Uses the OpenAI image generation tool path for the draft figure.",
    },
  };
  const SVG_PROVIDER_CONFIGS = {
    gemini: {
      apiPlaceholder: "AIza...",
      svgProvider: "gemini",
      svgModel: "gemini-3.1-pro-preview",
      hint: "Google Gemini official API for multimodal SVG reasoning and reconstruction.",
    },
    openai: {
      apiPlaceholder: "sk-proj-...",
      svgProvider: "openai",
      svgModel: "gpt-4.1",
      hint: "OpenAI official API for multimodal SVG reasoning and reconstruction.",
    },
    anthropic: {
      apiPlaceholder: "sk-ant-...",
      svgProvider: "anthropic",
      svgModel: "claude-sonnet-4-20250514",
      hint: "Anthropic official API for Claude-based SVG understanding and reconstruction.",
    },
  };

  const page = document.body.dataset.page;
  if (page === "input") {
    initInputPage();
  } else if (page === "canvas") {
    initCanvasPage();
  }

  function $(id) {
    return document.getElementById(id);
  }

  function initInputPage() {
    const confirmBtn = $("confirmBtn");
    const errorMsg = $("errorMsg");
    const uploadZone = $("uploadZone");
    const referenceFile = $("referenceFile");
    const referencePreview = $("referencePreview");
    const referenceStatus = $("referenceStatus");
    const figureCaption = $("figureCaption");
    const imageSizeGroup = $("imageSizeGroup");
    const imageSizeInput = $("imageSize");
    const generationModeGroup = $("generationModeGroup");
    const generationModeInput = $("generationMode");
    const figureModeInput = $("figureMode");
    const numCandidatesInput = $("numCandidates");
    const imageProviderInput = $("imageProvider");
    const imageProviderHint = $("imageProviderHint");
    const imageApiKeyInput = $("imageApiKey");
    const svgProviderInput = $("svgProvider");
    const svgProviderHint = $("svgProviderHint");
    const svgApiKeyInput = $("svgApiKey");
    const samBackend = $("samBackend");
    const samPrompt = $("samPrompt");
    const samApiKeyGroup = $("samApiKeyGroup");
    const samApiKeyInput = $("samApiKey");
    let uploadedReferencePath = null;

    function loadInputState() {
      try {
        const raw = window.sessionStorage.getItem(INPUT_STATE_KEY);
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch (_err) {
        return null;
      }
    }

    function saveInputState() {
      const state = {
        methodText: $("methodText")?.value ?? "",
        figureCaption: figureCaption?.value ?? "",
        imageProvider: imageProviderInput?.value ?? "gemini",
        imageApiKey: imageApiKeyInput?.value ?? "",
        svgProvider: svgProviderInput?.value ?? "gemini",
        svgApiKey: svgApiKeyInput?.value ?? "",
        optimizeIterations: $("optimizeIterations")?.value ?? "0",
        numCandidates: numCandidatesInput?.value ?? "1",
        figureMode: figureModeInput?.value ?? "simple_flowchart",
        imageSize: imageSizeInput?.value ?? "2K",
        generationMode: generationModeInput?.value ?? "balanced",
        samBackend: samBackend?.value ?? "local",
        samPrompt: samPrompt?.value ?? "plot,chart,heatmap,matrix,image",
        samApiKey: samApiKeyInput?.value ?? "",
        referencePath: uploadedReferencePath,
        referenceUrl: referencePreview?.src ?? "",
        referenceStatus: referenceStatus?.textContent ?? "",
      };
      try {
        window.sessionStorage.setItem(INPUT_STATE_KEY, JSON.stringify(state));
      } catch (_err) {
        // Ignore storage failures (e.g. private mode / quota)
      }
    }

    function applyInputState() {
      const state = loadInputState();
      if (!state) {
        return;
      }
      if (typeof state.methodText === "string") {
        $("methodText").value = state.methodText;
      }
      if (typeof state.figureCaption === "string" && figureCaption) {
        figureCaption.value = state.figureCaption;
      }
      const legacyMap = { bianxie: "openai", openrouter: "gemini", claude: "anthropic" };
      if (typeof state.imageProvider === "string" && imageProviderInput) {
        imageProviderInput.value = legacyMap[state.imageProvider] || state.imageProvider;
      } else if (typeof state.provider === "string" && imageProviderInput) {
        imageProviderInput.value = legacyMap[state.provider] || state.provider;
      }
      if (typeof state.imageApiKey === "string" && imageApiKeyInput) {
        imageApiKeyInput.value = state.imageApiKey;
      } else if (typeof state.apiKey === "string" && imageApiKeyInput) {
        imageApiKeyInput.value = state.apiKey;
      }
      if (typeof state.svgProvider === "string" && svgProviderInput) {
        svgProviderInput.value = legacyMap[state.svgProvider] || state.svgProvider;
      } else if (typeof state.provider === "string" && svgProviderInput) {
        svgProviderInput.value = legacyMap[state.provider] || state.provider;
      }
      if (typeof state.svgApiKey === "string" && svgApiKeyInput) {
        svgApiKeyInput.value = state.svgApiKey;
      } else if (typeof state.apiKey === "string" && svgApiKeyInput) {
        svgApiKeyInput.value = state.apiKey;
      }
      if (typeof state.optimizeIterations === "string" && $("optimizeIterations")) {
        $("optimizeIterations").value = state.optimizeIterations;
      }
      if (typeof state.numCandidates === "string" && numCandidatesInput) {
        numCandidatesInput.value = state.numCandidates;
      }
      if (typeof state.figureMode === "string" && figureModeInput) {
        figureModeInput.value = state.figureMode;
      }
      if (typeof state.imageSize === "string" && imageSizeInput) {
        imageSizeInput.value = state.imageSize;
      }
      if (typeof state.generationMode === "string" && generationModeInput) {
        generationModeInput.value = state.generationMode;
      }
      if (typeof state.samBackend === "string" && samBackend) {
        samBackend.value = state.samBackend;
      }
      if (typeof state.samPrompt === "string" && samPrompt) {
        samPrompt.value = state.samPrompt;
      }
      if (typeof state.samApiKey === "string" && samApiKeyInput) {
        samApiKeyInput.value = state.samApiKey;
      }
      if (typeof state.referencePath === "string" && state.referencePath) {
        uploadedReferencePath = state.referencePath;
      }
      if (
        referencePreview &&
        typeof state.referenceUrl === "string" &&
        state.referenceUrl
      ) {
        referencePreview.src = state.referenceUrl;
        referencePreview.classList.add("visible");
      }
      if (
        referenceStatus &&
        typeof state.referenceStatus === "string" &&
        state.referenceStatus
      ) {
        referenceStatus.textContent = state.referenceStatus;
      }
    }

    function syncImageSizeVisibility() {
      const provider = imageProviderInput?.value ?? "gemini";
      const show = provider === "gemini";
      if (imageSizeGroup) {
        imageSizeGroup.hidden = !show;
      }
      if (generationModeGroup) {
        generationModeGroup.hidden = !show;
      }
      saveInputState();
    }

    function syncProviderPresentation() {
      const imageProvider = imageProviderInput?.value ?? "gemini";
      const imageConfig = IMAGE_PROVIDER_CONFIGS[imageProvider] || IMAGE_PROVIDER_CONFIGS.gemini;
      if (imageProviderHint) {
        imageProviderHint.textContent = imageConfig.hint;
      }
      if (imageApiKeyInput) {
        imageApiKeyInput.placeholder = imageConfig.apiPlaceholder;
      }

      const svgProvider = svgProviderInput?.value ?? "gemini";
      const svgConfig = SVG_PROVIDER_CONFIGS[svgProvider] || SVG_PROVIDER_CONFIGS.gemini;
      if (svgProviderHint) {
        svgProviderHint.textContent = svgConfig.hint;
      }
      if (svgApiKeyInput) {
        svgApiKeyInput.placeholder = svgConfig.apiPlaceholder;
      }
      saveInputState();
    }

    function syncSamApiKeyVisibility() {
      const shouldShow =
        samBackend &&
        (samBackend.value === "fal" || samBackend.value === "roboflow");
      if (samApiKeyGroup) {
        samApiKeyGroup.hidden = !shouldShow;
      }
      if (!shouldShow && samApiKeyInput) {
        samApiKeyInput.value = "";
      }
      saveInputState();
    }

    function syncFigureModeDefaults() {
      if (!figureModeInput || !samPrompt) {
        return;
      }
      samPrompt.value =
        SAM_PROMPT_PRESETS[figureModeInput.value] || SAM_PROMPT_PRESETS.simple_flowchart;
      saveInputState();
    }

    applyInputState();

    if (samBackend) {
      samBackend.addEventListener("change", syncSamApiKeyVisibility);
      syncSamApiKeyVisibility();
    }
    if (figureModeInput) {
      figureModeInput.addEventListener("change", syncFigureModeDefaults);
    }
    if (imageProviderInput) {
      imageProviderInput.addEventListener("change", syncImageSizeVisibility);
      imageProviderInput.addEventListener("change", syncProviderPresentation);
    }
    if (svgProviderInput) {
      svgProviderInput.addEventListener("change", syncProviderPresentation);
    }
    if (imageProviderInput || svgProviderInput) {
      syncImageSizeVisibility();
      syncProviderPresentation();
    }

    if (uploadZone && referenceFile) {
      uploadZone.addEventListener("click", () => referenceFile.click());
      uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragging");
      });
      uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragging");
      });
      uploadZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragging");
        const file = event.dataTransfer.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
      referenceFile.addEventListener("change", async () => {
        const file = referenceFile.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
    }

    const autoSaveFields = [
      $("methodText"),
      figureCaption,
      imageProviderInput,
      imageApiKeyInput,
      svgProviderInput,
      svgApiKeyInput,
      $("optimizeIterations"),
      numCandidatesInput,
      figureModeInput,
      $("imageSize"),
      generationModeInput,
      samPrompt,
      samApiKeyInput,
    ];
    for (const field of autoSaveFields) {
      if (!field) {
        continue;
      }
      field.addEventListener("input", saveInputState);
      field.addEventListener("change", saveInputState);
    }

    confirmBtn.addEventListener("click", async () => {
      errorMsg.textContent = "";
      const methodText = $("methodText").value.trim();
      if (!methodText) {
        errorMsg.textContent = "Please provide method text.";
        return;
      }

      confirmBtn.disabled = true;
      confirmBtn.textContent = "Starting...";

      const imageProviderKey = imageProviderInput?.value || "gemini";
      const svgProviderKey = svgProviderInput?.value || "gemini";
      const rawImageApiKey = imageApiKeyInput?.value.trim() || "";
      const rawSvgApiKey = svgApiKeyInput?.value.trim() || "";
      const resolvedImageApiKey =
        rawImageApiKey || (imageProviderKey === svgProviderKey ? rawSvgApiKey : "");
      const resolvedSvgApiKey =
        rawSvgApiKey || (svgProviderKey === imageProviderKey ? rawImageApiKey : "");

      const payload = {
        method_text: methodText,
        figure_caption: figureCaption?.value.trim() || null,
        figure_mode: figureModeInput?.value || "simple_flowchart",
        provider: "gemini",
        image_provider: (IMAGE_PROVIDER_CONFIGS[imageProviderKey] || IMAGE_PROVIDER_CONFIGS.gemini).imageProvider,
        image_api_key: resolvedImageApiKey || null,
        svg_provider: (SVG_PROVIDER_CONFIGS[svgProviderKey] || SVG_PROVIDER_CONFIGS.gemini).svgProvider,
        svg_api_key: resolvedSvgApiKey || null,
        optimize_iterations: parseInt($("optimizeIterations").value, 10),
        num_candidates: parseInt(numCandidatesInput?.value || "1", 10),
        reference_image_path: uploadedReferencePath,
        sam_backend: $("samBackend").value,
        sam_prompt: $("samPrompt").value.trim() || null,
        sam_api_key: $("samApiKey").value.trim() || null,
      };
      const imageProviderConfig =
        IMAGE_PROVIDER_CONFIGS[imageProviderKey] || IMAGE_PROVIDER_CONFIGS.gemini;
      const svgProviderConfig =
        SVG_PROVIDER_CONFIGS[svgProviderKey] || SVG_PROVIDER_CONFIGS.gemini;
      if (imageProviderKey === "gemini") {
        payload.image_size = imageSizeInput?.value || "2K";
        payload.image_model =
          GEMINI_IMAGE_MODELS[generationModeInput?.value || "balanced"] ||
          GEMINI_IMAGE_MODELS.balanced;
      } else {
        payload.image_model = imageProviderConfig.imageModel || null;
      }
      if (svgProviderConfig.svgModel) {
        payload.svg_model = svgProviderConfig.svgModel;
      }
      if (payload.sam_backend === "local") {
        payload.sam_api_key = null;
      }
      saveInputState();

      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }

        const data = await response.json();
        window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
      } catch (err) {
        errorMsg.textContent = err.message || "Failed to start job";
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Confirm -> Canvas";
      }
    });
  }

  async function uploadReference(file, confirmBtn, previewEl, statusEl) {
    if (!file.type.startsWith("image/")) {
      statusEl.textContent = "Only image files are supported.";
      return null;
    }

    confirmBtn.disabled = true;
    statusEl.textContent = "Uploading reference...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Upload failed");
      }

      const data = await response.json();
      statusEl.textContent = `Using uploaded reference: ${data.name}`;
      if (previewEl) {
        previewEl.src = data.url || "";
        previewEl.classList.add("visible");
      }
      return {
        path: data.path || null,
        url: data.url || "",
        name: data.name || "",
      };
    } catch (err) {
      statusEl.textContent = err.message || "Upload failed";
      return null;
    } finally {
      confirmBtn.disabled = false;
    }
  }

  async function initCanvasPage() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job");
    const statusText = $("statusText");
    const jobIdEl = $("jobId");
    const artifactPanel = $("artifactPanel");
    const artifactList = $("artifactList");
    const toggle = $("artifactToggle");
    const logToggle = $("logToggle");
    const logToggleSidebar = $("logToggleSidebar");
    const backToConfigBtn = $("backToConfigBtn");
    const logPanel = $("logPanel");
    const logBody = $("logBody");
    const iframe = $("svgEditorFrame");
    const fallback = $("svgFallback");
    const fallbackObject = $("fallbackObject");

    if (!jobId) {
      statusText.textContent = "Missing job id";
      return;
    }

    jobIdEl.textContent = jobId;

    if (toggle && artifactPanel) {
      toggle.addEventListener("click", () => {
        artifactPanel.classList.toggle("open");
      });
    }

    const toggleLogPanel = () => {
      if (logPanel) {
        logPanel.classList.toggle("open");
      }
    };
    if (logToggle) {
      logToggle.addEventListener("click", toggleLogPanel);
    }
    if (logToggleSidebar) {
      logToggleSidebar.addEventListener("click", toggleLogPanel);
    }
    if (backToConfigBtn) {
      backToConfigBtn.addEventListener("click", () => {
        window.location.href = "/";
      });
    }

    let svgEditAvailable = false;
    let svgEditPath = null;
    try {
      const configRes = await fetch("/api/config");
      if (configRes.ok) {
        const config = await configRes.json();
        svgEditAvailable = Boolean(config.svgEditAvailable);
        svgEditPath = config.svgEditPath || null;
      }
    } catch (err) {
      svgEditAvailable = false;
    }

    if (svgEditAvailable && svgEditPath) {
      iframe.src = svgEditPath;
    } else {
      fallback.classList.add("active");
      iframe.style.display = "none";
    }

    let svgReady = false;
    let pendingSvgText = null;

    iframe.addEventListener("load", () => {
      svgReady = true;
      if (pendingSvgText) {
        tryLoadSvg(pendingSvgText);
        pendingSvgText = null;
      }
    });

    const stepMap = {
      figure: { step: 1, label: "Figure generated" },
      samed: { step: 2, label: "SAM3 segmentation" },
      icon_raw: { step: 3, label: "Icons extracted" },
      icon_nobg: { step: 3, label: "Icons refined" },
      template_svg: { step: 4, label: "Template SVG ready" },
      optimized_template_svg: { step: 4, label: "Optimized template ready" },
      final_svg: { step: 5, label: "Final SVG ready" },
    };

    let currentStep = 0;
    let autoLoadedSvg = false;

    const artifacts = new Set();
    const eventSource = new EventSource(`/api/events/${jobId}`);
    let isFinished = false;

    eventSource.addEventListener("artifact", async (event) => {
      const data = JSON.parse(event.data);
      if (!artifacts.has(data.path)) {
        artifacts.add(data.path);
        addArtifactCard(artifactList, data, loadSvgAsset);
      }

      if ((data.kind === "template_svg" || data.kind === "final_svg") && (data.primary || !autoLoadedSvg)) {
        await loadSvgAsset(data.url);
        autoLoadedSvg = true;
      }

      if (stepMap[data.kind] && stepMap[data.kind].step > currentStep) {
        currentStep = stepMap[data.kind].step;
        statusText.textContent = `Step ${currentStep}/5 - ${stepMap[data.kind].label}`;
      }
    });

    eventSource.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.state === "started") {
        statusText.textContent = "Running";
      } else if (data.state === "finished") {
        isFinished = true;
        if (typeof data.code === "number" && data.code !== 0) {
          statusText.textContent = `Failed (code ${data.code})`;
        } else {
          statusText.textContent = "Done";
        }
      }
    });

    eventSource.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      appendLogLine(logBody, data);
    });

    eventSource.onerror = () => {
      if (isFinished) {
        eventSource.close();
        return;
      }
      statusText.textContent = "Disconnected";
    };

    async function loadSvgAsset(url) {
      let svgText = "";
      try {
        const response = await fetch(url);
        svgText = await response.text();
      } catch (err) {
        return;
      }

      if (svgEditAvailable) {
        if (!svgEditPath) {
          return;
        }
        if (!svgReady) {
          pendingSvgText = svgText;
          return;
        }

        const loaded = tryLoadSvg(svgText);
        if (!loaded) {
          iframe.src = `${svgEditPath}?url=${encodeURIComponent(url)}`;
        }
      } else {
        fallbackObject.data = url;
      }
    }

    function tryLoadSvg(svgText) {
      if (!iframe.contentWindow) {
        return false;
      }

      const win = iframe.contentWindow;
      if (win.svgEditor && typeof win.svgEditor.loadFromString === "function") {
        win.svgEditor.loadFromString(svgText);
        return true;
      }
      if (win.svgCanvas && typeof win.svgCanvas.setSvgString === "function") {
        win.svgCanvas.setSvgString(svgText);
        return true;
      }
      return false;
    }
  }

  function appendLogLine(container, data) {
    const line = `[${data.stream}] ${data.line}`;
    const lines = container.textContent.split("\n").filter(Boolean);
    lines.push(line);
    if (lines.length > 200) {
      lines.splice(0, lines.length - 200);
    }
    container.textContent = lines.join("\n");
    container.scrollTop = container.scrollHeight;
  }

  function addArtifactCard(container, data, loadSvgAsset) {
    const card = document.createElement("a");
    card.className = "artifact-card";
    card.href = data.url;
    card.target = "_blank";
    card.rel = "noreferrer";

    const visualKinds = new Set([
      "figure",
      "samed",
      "icon_raw",
      "icon_nobg",
      "template_svg",
      "optimized_template_svg",
      "final_svg",
    ]);
    let previewEl;
    if (visualKinds.has(data.kind)) {
      const img = document.createElement("img");
      img.src = data.url;
      img.alt = data.name;
      img.loading = "lazy";
      previewEl = img;
    } else {
      const thumb = document.createElement("div");
      thumb.className = "artifact-thumb";
      thumb.textContent = (data.candidate_label || "TXT").slice(0, 3).toUpperCase();
      previewEl = thumb;
    }

    const meta = document.createElement("div");
    meta.className = "artifact-meta";

    const name = document.createElement("div");
    name.className = "artifact-name";
    name.textContent = data.display_name || data.name;

    const badge = document.createElement("div");
    badge.className = "artifact-badge";
    badge.textContent = formatKind(data.kind, data.candidate_label);

    meta.appendChild(name);
    meta.appendChild(badge);
    card.appendChild(previewEl);
    card.appendChild(meta);
    if (
      loadSvgAsset &&
      (data.kind === "template_svg" ||
        data.kind === "optimized_template_svg" ||
        data.kind === "final_svg")
    ) {
      card.addEventListener("click", async (event) => {
        event.preventDefault();
        await loadSvgAsset(data.url);
      });
    }
    container.prepend(card);
  }

  function formatKind(kind, candidateLabel) {
    const prefix = candidateLabel ? `${candidateLabel} / ` : "";
    switch (kind) {
      case "figure":
        return `${prefix}figure`;
      case "samed":
        return `${prefix}samed`;
      case "icon_raw":
        return `${prefix}icon raw`;
      case "icon_nobg":
        return `${prefix}icon no-bg`;
      case "template_svg":
        return `${prefix}template`;
      case "optimized_template_svg":
        return `${prefix}template+`;
      case "final_svg":
        return `${prefix}final`;
      case "candidate_manifest":
        return "manifest";
      case "candidate_error":
        return `${prefix}error`;
      case "log":
        return "log";
      default:
        return `${prefix}artifact`;
    }
  }
})();
