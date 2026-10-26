// ── DOM ELEMENTS & GLOBAL STATE ──────────────────────────────────
const fileInput         = document.getElementById('fileInput');
const dropzone          = document.getElementById('dropzone');
const previewArea       = document.getElementById('previewArea');
const previewImg        = document.getElementById('previewImg');
const scanOverlay       = document.getElementById('scanOverlay');
const scanBeam          = document.getElementById('scanBeam');
const scanPct           = document.getElementById('scanPct');
const doneBadge         = document.getElementById('doneBadge');
const metaName          = document.getElementById('metaName');
const metaSize          = document.getElementById('metaSize');
const runBtn            = document.getElementById('runBtn');
const changeImgBtn      = document.getElementById('changeImgBtn');

// Navigation & Status
const serverPill        = document.getElementById('serverPill');
const sDot              = document.getElementById('sDot');
const sLabel            = document.getElementById('sLabel');
const modelStatusPill   = document.getElementById('modelStatusPill');
const modelStatusLabel  = document.getElementById('modelStatusLabel');
const historyCount      = document.getElementById('historyCount');

// Mode Switcher Tabs
const tabSingle         = document.getElementById('tabSingle');
const tabBatch          = document.getElementById('tabBatch');
const uploadCard        = document.getElementById('uploadCard');
const batchCard         = document.getElementById('batchCard');

// Model Selector Buttons
const modelBtns         = document.querySelectorAll('.model-btn');
let selectedModel       = 'ensemble';

// Single Analysis Modal Elements
const modalBg           = document.getElementById('modalBg');
const modalScanning     = document.getElementById('modalScanning');
const modalScanImg      = document.getElementById('modalScanImg');
const scanningSteps     = document.getElementById('scanningSteps');
const scanningLabel     = document.getElementById('scanningLabel');
const scanningFill      = document.getElementById('scanningFill');
const closeModalBtn     = document.getElementById('closeModalBtn');
const analyseAgainBtn   = document.getElementById('analyseAgainBtn');

// Results Elements
const modalResult       = document.getElementById('modalResult');
const mrTitle           = document.getElementById('mrTitle');
const mrChip            = document.getElementById('mrChip');
const mrModelBadge      = document.getElementById('mrModelBadge');
const mrVerdictOverlay  = document.getElementById('mrVerdictOverlay');
const ringArc           = document.getElementById('ringArc');
const ringPct           = document.getElementById('ringPct');
const mrBarFill         = document.getElementById('mrBarFill');
const mrBarSubtext      = document.getElementById('mrBarSubtext');
const nnDenseVal        = document.getElementById('nnDenseVal');
const nnResVal          = document.getElementById('nnResVal');
const nnAgreementVal    = document.getElementById('nnAgreementVal');
const mrMetrics         = document.getElementById('mrMetrics');
const mrTags            = document.getElementById('mrTags');
const disagreementBanner= document.getElementById('disagreementBanner');
const disagreementText  = document.getElementById('disagreementText');

// Latency Strip Elements
const tInference        = document.getElementById('tInference');
const tForensics        = document.getElementById('tForensics');
const tGradcam          = document.getElementById('tGradcam');
const tTotal            = document.getElementById('tTotal');

// Action Buttons
const downloadReportBtn = document.getElementById('downloadReportBtn');
const viewMetadataBtn   = document.getElementById('viewMetadataBtn');

// Grad-CAM Studio Elements
const camLayerTag       = document.getElementById('camLayerTag');
const camTabs           = document.querySelectorAll('.cam-tab');
const viewOriginalImg   = document.getElementById('viewOriginalImg');
const viewHeatmapImg    = document.getElementById('viewHeatmapImg');
const viewOverlayImg    = document.getElementById('viewOverlayImg');
const splitContainer    = document.getElementById('splitContainer');
const splitOriginalImg  = document.getElementById('splitOriginalImg');
const splitHeatmapImg   = document.getElementById('splitHeatmapImg');
const splitOverlayWrap  = document.getElementById('splitOverlayWrap');
const splitHandle       = document.getElementById('splitHandle');
const camOpacityBar     = document.getElementById('camOpacityBar');
const opacitySlider     = document.getElementById('opacitySlider');
const opacityVal        = document.getElementById('opacityVal');
const hrList            = document.getElementById('hrList');

// Submodals
const navDiagnosticsBtn = document.getElementById('navDiagnosticsBtn');
const navBenchmarksBtn  = document.getElementById('navBenchmarksBtn');
const navHistoryBtn     = document.getElementById('navHistoryBtn');
const diagnosticsModal  = document.getElementById('diagnosticsModal');
const benchmarksModal   = document.getElementById('benchmarksModal');
const metadataModal     = document.getElementById('metadataModal');
const historyModal      = document.getElementById('historyModal');
const closeDiagBtn      = document.getElementById('closeDiagBtn');
const closeBenchBtn     = document.getElementById('closeBenchBtn');
const closeMetaBtn      = document.getElementById('closeMetaBtn');
const closeHistBtn      = document.getElementById('closeHistBtn');
const clearHistoryBtn   = document.getElementById('clearHistoryBtn');

// Batch Elements
const batchFileInput    = document.getElementById('batchFileInput');
const batchQueue        = document.getElementById('batchQueue');
const runBatchBtn       = document.getElementById('runBatchBtn');
const batchTableWrap    = document.getElementById('batchTableWrap');
const batchTableBody    = document.getElementById('batchTableBody');
const downloadBatchCsvBtn = document.getElementById('downloadBatchCsvBtn');

// ── GLOBAL APPLICATION STATE ────────────────────────────────────
let currentFile = null;
let currentResultData = null;
let activeView = 'overlay';
let sessionHistory = [];
let batchFiles = [];
let batchResults = [];

// ── PARTICLE CANVAS BACKGROUND ──────────────────────────────────
const canvas = document.getElementById('bgCanvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  let pts = [];

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    initPts();
  });

  function initPts() {
    const count = Math.floor((canvas.width * canvas.height) / 14000);
    pts = Array.from({ length: count }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.4 + 0.4,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      o: Math.random() * 0.4 + 0.1,
    }));
  }
  initPts();

  function drawParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pts.forEach(p => {
      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232,200,74,${p.o})`;
      ctx.fill();
    });
    requestAnimationFrame(drawParticles);
  }
  drawParticles();
}

// ── TOAST NOTIFICATIONS ─────────────────────────────────────────
function toast(icon, title, sub, duration = 3000) {
  const wrap = document.getElementById('toastWrap');
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `
    <div class="t-icon">${icon}</div>
    <div class="t-text">
      <div class="t-title">${title}</div>
      <div class="t-sub">${sub}</div>
    </div>
  `;
  wrap.appendChild(t);
  setTimeout(() => {
    t.classList.add('bye');
    setTimeout(() => t.remove(), 250);
  }, duration);
}

// ── HEALTH & MODEL STATUS CHECK ────────────────────────────────
async function checkSystemHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();

    sDot.className = 'sdot online';
    sLabel.textContent = `online (${data.device})`;

    if (data.is_trained) {
      modelStatusPill.className = 'nav-badge-pill trained';
      modelStatusLabel.textContent = 'Trained Models Active';
    } else {
      modelStatusPill.className = 'nav-badge-pill demo';
      modelStatusLabel.textContent = 'Heuristic Forensics Mode';
    }
  } catch (e) {
    sDot.className = 'sdot offline';
    sLabel.textContent = 'server offline';
    modelStatusPill.className = 'nav-badge-pill';
    modelStatusLabel.textContent = 'Disconnected';
  }
}
checkSystemHealth();
setInterval(checkSystemHealth, 15000);

// ── MODE TABS SWITCHING ────────────────────────────────────────
tabSingle.addEventListener('click', () => {
  tabSingle.classList.add('active');
  tabBatch.classList.remove('active');
  uploadCard.style.display = 'block';
  batchCard.style.display = 'none';
});

tabBatch.addEventListener('click', () => {
  tabBatch.classList.add('active');
  tabSingle.classList.remove('active');
  uploadCard.style.display = 'none';
  batchCard.style.display = 'block';
});

// ── MODEL SELECTOR ─────────────────────────────────────────────
modelBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    modelBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedModel = btn.dataset.model;
    toast('⚙️', 'Model Selected', btn.querySelector('.mb-name').textContent, 1800);
  });
});

// ── SINGLE FILE DROPZONE & PREVIEW ─────────────────────────────
dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
});

changeImgBtn.addEventListener('click', () => {
  fileInput.value = '';
  currentFile = null;
  previewArea.style.display = 'none';
  dropzone.style.display    = 'block';
  runBtn.disabled           = true;
});

function handleFileSelect(file) {
  if (!file.type.startsWith('image/')) {
    toast('⚠️', 'Invalid File Type', 'Please upload a valid image file (PNG, JPG, WEBP, TIFF).', 3500);
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    toast('⚠️', 'File Too Large', 'Maximum image size is 25MB.', 3500);
    return;
  }
  loadImage(file);
}

function loadImage(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    metaName.textContent = file.name;
    metaSize.textContent = (file.size / 1024).toFixed(1) + ' KB';

    dropzone.style.display    = 'none';
    previewArea.style.display = 'block';
    doneBadge.style.display   = 'none';
    scanOverlay.classList.add('active');
    scanBeam.classList.add('running');
    scanPct.textContent = '0%';
    runBtn.disabled = true;

    let pct = 0;
    const t = setInterval(() => {
      pct += Math.random() * 20 + 10;
      if (pct >= 100) {
        pct = 100;
        clearInterval(t);
        scanBeam.classList.remove('running');
        scanOverlay.classList.remove('active');
        doneBadge.style.display = 'flex';
        runBtn.disabled = false;
        toast('🛡️', 'Image Loaded', file.name, 2000);
      }
      scanPct.textContent = Math.floor(pct) + '%';
    }, 30);
  };
  reader.readAsDataURL(file);
}

// ── DEMO SAMPLE BUTTONS ────────────────────────────────────────
const btnSampleAI = document.getElementById('btnSampleAI');
const btnSampleReal = document.getElementById('btnSampleReal');
if (btnSampleAI) btnSampleAI.addEventListener('click', () => loadDemoSample('ai'));
if (btnSampleReal) btnSampleReal.addEventListener('click', () => loadDemoSample('real'));

function loadDemoSample(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 300;
  canvas.height = 300;
  const ctx = canvas.getContext('2d');

  if (type === 'ai') {
    const grad = ctx.createRadialGradient(150, 150, 20, 150, 150, 140);
    grad.addColorStop(0, '#ff9a9e');
    grad.addColorStop(0.5, '#fecfef');
    grad.addColorStop(1, '#a1c4fd');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 300, 300);

    ctx.beginPath();
    ctx.ellipse(150, 140, 60, 80, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#fce4ec';
    ctx.fill();

    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Demo AI Synthetic Sample', 150, 260);
  } else {
    ctx.fillStyle = '#334155';
    ctx.fillRect(0, 0, 300, 300);

    const skyGrad = ctx.createLinearGradient(0, 0, 0, 180);
    skyGrad.addColorStop(0, '#64748b');
    skyGrad.addColorStop(1, '#94a3b8');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, 300, 180);

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 180, 300, 120);

    // Natural camera sensor noise simulation
    const imgData = ctx.getImageData(0, 0, 300, 300);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = (Math.random() - 0.5) * 35;
      d[i]   = Math.min(255, Math.max(0, d[i] + n));
      d[i+1] = Math.min(255, Math.max(0, d[i+1] + n));
      d[i+2] = Math.min(255, Math.max(0, d[i+2] + n));
    }
    ctx.putImageData(imgData, 0, 0);

    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Demo Camera Sensor Sample', 150, 270);
  }

  canvas.toBlob(blob => {
    const filename = type === 'ai' ? 'demo_synthetic_sample.png' : 'demo_camera_sample.jpg';
    const file = new File([blob], filename, { type: type === 'ai' ? 'image/png' : 'image/jpeg' });
    loadImage(file);
    toast(type === 'ai' ? '🤖' : '📷', 'Demo Sample Loaded', filename, 2200);
  });
}

// ── SCANNING & RESULT MODAL ────────────────────────────────────
function openModal() {
  modalBg.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  modalBg.classList.remove('open');
  document.body.style.overflow = '';
}

closeModalBtn.addEventListener('click', closeModal);
analyseAgainBtn.addEventListener('click', closeModal);
modalBg.addEventListener('click', e => {
  if (e.target === modalBg) closeModal();
});

function getPipelineSteps() {
  return [
    { name: 'Image Preprocessing', msg: 'Validating format and preparing normalized vision tensors…' },
    { name: 'Dual Neural Inference', msg: 'Executing DenseNet-121 and ResNet-50 forward passes…' },
    { name: 'Forensic Vision Suite', msg: 'Computing 2D FFT, ELA residuals, and Laplacian sensor noise…' },
    { name: 'Grad-CAM Synthesis', msg: 'Backpropagating convolutional feature activations…' },
    { name: 'Ensemble Calibration', msg: 'Synthesizing uncertainty-aware classification report…' }
  ];
}

function buildSteps(steps) {
  scanningSteps.innerHTML = steps.map((s, i) =>
    `<div class="s-step" id="ss${i}">${s.name}</div>`
  ).join('');
}

function setStep(steps, i) {
  steps.forEach((_, j) => {
    const el = document.getElementById(`ss${j}`);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (j < i)  el.classList.add('done');
    if (j === i) el.classList.add('active');
  });
  scanningFill.style.width = ((i / (steps.length - 1)) * 100) + '%';
  scanningLabel.textContent = steps[Math.min(i, steps.length - 1)].msg;
}

// ── RUN SINGLE IMAGE ANALYSIS ──────────────────────────────────
runBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!currentFile) return;
  runBtn.disabled = true;

  const steps = getPipelineSteps();
  modalScanning.style.display = 'flex';
  modalResult.style.display   = 'none';
  modalScanImg.src            = previewImg.src;
  buildSteps(steps);
  setStep(steps, 0);
  openModal();

  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, steps.length - 1);
    setStep(steps, stepIdx);
  }, 450);

  try {
    const fd = new FormData();
    fd.append('image', currentFile);
    fd.append('model', selectedModel);

    const res = await fetch('/analyze', {
      method: 'POST',
      body: fd
    });

    clearInterval(stepTimer);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    setStep(steps, steps.length - 1);
    scanningFill.style.width = '100%';
    scanningLabel.textContent = 'Forensic synthesis complete!';

    await new Promise(r => setTimeout(r, 250));

    renderResult(data);
    modalScanning.style.display = 'none';
    modalResult.style.display   = 'grid';

    // Record in session history
    addToHistory(data, currentFile.name);

    toast(
      data.verdict_code === 'ai' ? '🤖' : data.verdict_code === 'real' ? '🛡️' : '⚠️',
      data.result,
      `${data.model_used} — ${data.confidence.toFixed(1)}% Confidence`,
      3600
    );

  } catch (err) {
    clearInterval(stepTimer);
    closeModal();
    toast('⚠️', 'Analysis Failed', err.message || 'Server error occurred', 5000);
    console.error('[DeepShield Analysis Error]', err);
  } finally {
    runBtn.disabled = false;
  }
}

// ── RENDER RESULTS ─────────────────────────────────────────────
function renderResult(data) {
  currentResultData = data;
  const code = data.verdict_code; // 'real', 'ai', or 'uncertain'
  const conf = data.confidence;
  const f    = data.forensics || {};
  const ms   = data.model_scores || {};
  const gcam = data.gradcam || {};
  const tm   = data.timing_ms || {};
  const ma   = data.model_agreement || {};

  // Header & Model Badge
  camLayerTag.textContent = gcam.target_layer || 'features.norm5';
  mrModelBadge.textContent = data.model_used + (data.is_demo_mode ? ' (Demo Forensics)' : '');
  mrTitle.textContent      = data.result;
  mrTitle.className        = `mr-title ${code}`;
  mrChip.textContent       = code === 'ai' ? 'Synthetic' : code === 'real' ? 'Authentic' : 'Uncertain';
  mrChip.className         = `mr-chip ${code}`;

  mrVerdictOverlay.textContent = `${data.result} (${conf.toFixed(1)}%)`;
  mrVerdictOverlay.className   = `mr-verdict-overlay ${code}`;

  // Disagreement Banner
  if (ma.disagreement_warning) {
    disagreementBanner.classList.add('show');
    disagreementText.textContent = ma.explanation;
  } else {
    disagreementBanner.classList.remove('show');
  }

  // Load Grad-CAM studio viewers
  viewOriginalImg.src  = previewImg.src;
  splitOriginalImg.src = previewImg.src;
  viewHeatmapImg.src   = gcam.heatmap_base64 || previewImg.src;
  viewOverlayImg.src   = gcam.overlay_base64 || previewImg.src;
  splitHeatmapImg.src  = gcam.overlay_base64 || previewImg.src;

  switchCamView('overlay');

  // Confidence Circular Gauge
  const circ = 238.7;
  const offset = circ - (conf / 100) * circ;
  ringArc.style.strokeDashoffset = circ;
  ringArc.style.stroke = code === 'ai' ? '#f43f5e' : code === 'real' ? '#10b981' : '#e8c84a';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    ringArc.style.strokeDashoffset = offset;
  }));
  countUp(ringPct, 0, conf, 1000, v => v.toFixed(1) + '%');

  // Confidence Bar
  mrBarFill.style.width = '0%';
  mrBarFill.className   = `mr-bar-fill ${code}`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    mrBarFill.style.width = (data.ai_probability != null ? data.ai_probability : conf) + '%';
  }));

  mrBarSubtext.textContent = ma.explanation || (
    code === 'ai'
      ? 'Elevated synthetic feature activations detected across deep convolutional layers.'
      : 'Evenly distributed spatial frequencies and consistent optical sensor noise confirmed.'
  );

  // Neural Backbone Breakdown Cards
  nnDenseVal.textContent = ms.densenet121_ai != null ? `${ms.densenet121_ai.toFixed(1)}% AI` : '—';
  nnResVal.textContent   = ms.resnet50_ai != null ? `${ms.resnet50_ai.toFixed(1)}% AI` : '—';
  nnAgreementVal.textContent = ma.agree ? 'Concordant' : `Divergent (Δ${ma.disagreement_delta}%)`;
  nnAgreementVal.style.color = ma.agree ? 'var(--green)' : 'var(--accent)';

  // Hot Regions List
  renderHotRegions(gcam.hot_regions || []);

  // Forensic Metrics Grid
  const metrics = [
    { icon: '⚡', name: 'ELA Residual',    val: f.ela_mean != null ? `${f.ela_mean.toFixed(1)}` : '—', bar: Math.min((f.ela_mean || 0) / 20 * 100, 100), col: '#4ecdc4' },
    { icon: '🌊', name: '2D FFT HF Ratio', val: f.dct_hf_ratio != null ? f.dct_hf_ratio.toFixed(4) : '—', bar: (f.dct_hf_ratio || 0) * 100, col: '#e8c84a' },
    { icon: '📉', name: 'Spectral Slope',  val: f.spectral_slope != null ? `${f.spectral_slope.toFixed(2)}` : '—', bar: Math.min(Math.abs(f.spectral_slope || 1.8) / 3 * 100, 100), col: '#f5da6c' },
    { icon: '📡', name: 'Noise Level',     val: f.noise_var != null ? `${f.noise_var.toFixed(1)}` : '—', bar: Math.min((f.noise_var || 0) / 200 * 100, 100), col: '#52d68a' },
    { icon: '🎨', name: 'Channel Corr',    val: f.channel_corr != null ? f.channel_corr.toFixed(3) : '—', bar: (f.channel_corr || 0) * 100, col: code === 'ai' ? '#f07070' : '#52d68a' },
    { icon: '📷', name: 'Metadata Status', val: f.has_exif ? 'EXIF Present' : 'No EXIF (Web/Re-saved)', bar: f.has_exif ? 100 : 30, col: f.has_exif ? '#52d68a' : '#8b93a8' },
  ];

  mrMetrics.innerHTML = metrics.map((m, i) => `
    <div class="mc" style="animation-delay:${i * 0.04}s">
      <div class="mc-icon">${m.icon}</div>
      <div class="mc-name">${m.name}</div>
      <div class="mc-val">${m.val}</div>
      <div class="mc-bar"><div class="mc-bar-fill" style="width:0%;background:${m.col}" data-w="${m.bar}"></div></div>
    </div>
  `).join('');

  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll('.mc-bar-fill').forEach(el => {
      el.style.width = el.dataset.w + '%';
    });
  }));

  // Forensic Tags
  mrTags.innerHTML = buildForensicTags(f, ms, code).map(([cls, txt]) =>
    `<span class="stag ${cls}">${txt}</span>`
  ).join('');

  // Latency Breakdown
  tInference.textContent = `${tm.inference || 0}ms`;
  tForensics.textContent = `${tm.forensics || 0}ms`;
  tGradcam.textContent   = `${tm.gradcam || 0}ms`;
  tTotal.textContent     = `${tm.total || 0}ms`;
}

// ── RENDER HOT REGIONS ────────────────────────────────────────
function renderHotRegions(regions) {
  if (!regions.length) {
    hrList.innerHTML = `<div class="hr-desc">No abnormal gradient activations detected.</div>`;
    return;
  }
  hrList.innerHTML = regions.map(r => `
    <div class="hr-item">
      <div class="hr-top-row">
        <span class="hr-reg-name">📍 ${r.region}</span>
        <span class="hr-peak">Activation: ${r.peak_activation}</span>
      </div>
      <div class="hr-desc">${r.explanation}</div>
    </div>
  `).join('');
}

// ── GRAD-CAM VIEW SWITCHER ────────────────────────────────────
camTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    camTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    switchCamView(tab.dataset.view);
  });
});

function switchCamView(view) {
  activeView = view;
  viewOriginalImg.style.display = 'none';
  viewHeatmapImg.style.display  = 'none';
  viewOverlayImg.style.display  = 'none';
  splitContainer.style.display  = 'none';
  camOpacityBar.style.display   = 'none';

  if (view === 'overlay') {
    viewOverlayImg.style.display = 'block';
    camOpacityBar.style.display  = 'flex';
  } else if (view === 'heatmap') {
    viewHeatmapImg.style.display = 'block';
  } else if (view === 'original') {
    viewOriginalImg.style.display = 'block';
  } else if (view === 'split') {
    splitContainer.style.display = 'block';
    updateSplit(50);
  }
}

// Opacity Slider
opacitySlider.addEventListener('input', e => {
  const val = e.target.value;
  opacityVal.textContent = val + '%';
  viewOverlayImg.style.opacity = val / 100;
});

// Interactive Split Slider
let isDraggingSplit = false;
function updateSplit(pct) {
  const clamped = Math.max(5, Math.min(95, pct));
  splitOverlayWrap.style.width = clamped + '%';
  splitHandle.style.left        = clamped + '%';
}
splitHandle.addEventListener('mousedown', () => { isDraggingSplit = true; });
splitContainer.addEventListener('mousedown', e => {
  isDraggingSplit = true;
  handleSplitMove(e);
});
window.addEventListener('mouseup', () => { isDraggingSplit = false; });
window.addEventListener('mousemove', e => { if (isDraggingSplit) handleSplitMove(e); });
function handleSplitMove(e) {
  const rect = splitContainer.getBoundingClientRect();
  const x = e.clientX - rect.left;
  updateSplit((x / rect.width) * 100);
}
splitHandle.addEventListener('touchstart', () => { isDraggingSplit = true; });
splitContainer.addEventListener('touchmove', e => {
  if (!isDraggingSplit) return;
  const rect = splitContainer.getBoundingClientRect();
  const touch = e.touches[0];
  const x = touch.clientX - rect.left;
  updateSplit((x / rect.width) * 100);
});
window.addEventListener('touchend', () => { isDraggingSplit = false; });

// ── COUNT UP ANIMATION ────────────────────────────────────────
function countUp(el, from, to, dur, fmt) {
  const start = performance.now();
  (function step(now) {
    const t    = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  })(performance.now());
}

// ── FORENSIC TAGS GENERATOR ───────────────────────────────────
function buildForensicTags(f, ms, code) {
  const t = [];
  if (ms.densenet121_ai != null) {
    t.push(ms.densenet121_ai > 60 ? ['bad', 'DenseNet: Elevated Synthetic Score'] : ['ok', 'DenseNet: Natural Pattern Flow']);
  }
  if (ms.resnet50_ai != null) {
    t.push(ms.resnet50_ai > 60 ? ['bad', 'ResNet: Structural Anomaly'] : ['ok', 'ResNet: Coherent Structure']);
  }
  if (f.noise_var != null) {
    t.push(f.noise_var < 10 ? ['bad', 'Low Sensor Noise (<10)'] : f.noise_var > 400 ? ['warn', 'High Uniform Noise'] : ['ok', 'Natural Sensor Noise Variance']);
  }
  if (f.dct_hf_ratio != null) {
    t.push(f.dct_hf_ratio < 0.25 ? ['bad', 'High-Frequency Deficit'] : ['ok', 'Healthy Frequency Spectrum']);
  }
  if (f.ai_software_detected) {
    t.push(['bad', `AI Software Tag: ${f.software || 'Generator Detected'}`]);
  } else if (f.has_exif) {
    t.push(['ok', 'Hardware EXIF Detected']);
  }
  t.push(code === 'ai' ? ['bad', '🤖 Synthetic Origin Likely'] : code === 'real' ? ['ok', '🛡️ Authentic Pattern Alignment'] : ['warn', '⚠️ Inconclusive / Ambiguous Evidence']);
  return t;
}

// ── FEATURE B: DOWNLOAD FORENSIC REPORT ────────────────────────
downloadReportBtn.addEventListener('click', () => {
  if (!currentResultData) return;
  const report = {
    title: "DeepShield Digital Image Forensic Analysis Report",
    generated_at: new Date().toISOString(),
    filename: currentFile ? currentFile.name : "unknown",
    classification: {
      verdict: currentResultData.result,
      verdict_code: currentResultData.verdict_code,
      confidence_percent: currentResultData.confidence,
      ai_probability_percent: currentResultData.ai_probability,
      authentic_probability_percent: currentResultData.authentic_probability,
      model_agreement: currentResultData.model_agreement
    },
    neural_network_scores: currentResultData.model_scores,
    forensic_measurements: currentResultData.forensics,
    execution_latency_ms: currentResultData.timing_ms,
    limitations_and_disclaimer: (
      "This report provides algorithmic visual forensic estimations. " +
      "Grad-CAM visualizes neural network attention and does not independently prove authenticity. " +
      "EXIF metadata and frequency spectra can be altered by standard web compression and social media re-saving."
    )
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `DeepShield_Report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  toast('📄', 'Report Downloaded', 'Forensic JSON report saved successfully', 2500);
});

// ── FEATURE F: VIEW METADATA MODAL ─────────────────────────────
viewMetadataBtn.addEventListener('click', () => {
  if (!currentResultData) return;
  const imgInfo = currentResultData.image_info || {};
  const f = currentResultData.forensics || {};

  document.getElementById('metaDim').textContent = `${imgInfo.width || '—'} × ${imgInfo.height || '—'} px`;
  document.getElementById('metaFormat').textContent = imgInfo.format || (currentFile ? currentFile.type : '—');
  document.getElementById('metaMode').textContent = imgInfo.mode || 'RGB';
  document.getElementById('metaFilesize').textContent = currentFile ? `${(currentFile.size / 1024).toFixed(1)} KB` : '—';

  document.getElementById('metaCam').textContent = f.camera_model || 'None detected in EXIF';
  document.getElementById('metaSoftware').textContent = f.software || 'None detected';
  document.getElementById('metaStatusDesc').textContent = f.status_description || '—';

  metadataModal.classList.add('open');
});
closeMetaBtn.addEventListener('click', () => metadataModal.classList.remove('open'));

// ── FEATURE G: MODEL DIAGNOSTICS MODAL ─────────────────────────
navDiagnosticsBtn.addEventListener('click', async () => {
  try {
    const res = await fetch('/diagnostics');
    const data = await res.json();

    document.getElementById('diagDevice').textContent = data.device || 'CPU';
    document.getElementById('diagCuda').textContent = data.cuda_available ? 'Available (Enabled)' : 'Not Available (CPU Mode)';
    document.getElementById('diagTorch').textContent = data.torch_version || '—';
    document.getElementById('diagMode').textContent = data.active_mode === 'production_binary' ? 'Production Binary Classifiers' : 'Demo Forensics Mode';

    const ck = data.checkpoints || {};
    const dCk = ck.densenet121 || {};
    const rCk = ck.resnet50 || {};

    document.getElementById('diagDense').textContent = dCk.loaded ? `Loaded (${dCk.type})` : 'Missing (ImageNet Backbone Fallback)';
    document.getElementById('diagRes').textContent = rCk.loaded ? `Loaded (${rCk.type})` : 'Missing (ImageNet Backbone Fallback)';
    document.getElementById('diagCkptDir').textContent = data.checkpoint_directory || 'checkpoints/';

    diagnosticsModal.classList.add('open');
  } catch (e) {
    toast('⚠️', 'Diagnostics Error', 'Failed fetching system diagnostics', 3000);
  }
});
closeDiagBtn.addEventListener('click', () => diagnosticsModal.classList.remove('open'));

// ── FEATURE E: BENCHMARK EVALUATION MODAL ──────────────────────
navBenchmarksBtn.addEventListener('click', async () => {
  try {
    const res = await fetch('/benchmarks');
    const data = await res.json();

    if (data.status === 'evaluated' && data.metrics) {
      const m = data.metrics;
      document.getElementById('bmAcc').textContent = `${m.accuracy}%`;
      document.getElementById('bmAuc').textContent = `${m.roc_auc}`;
      document.getElementById('bmPrec').textContent = `${m.precision}%`;
      document.getElementById('bmRec').textContent = `${m.recall}%`;
      document.getElementById('bmF1').textContent = `${m.f1_score}%`;
      document.getElementById('bmFpr').textContent = `${m.false_positive_rate}%`;

      const cm = m.confusion_matrix || {};
      document.getElementById('cmTN').textContent = cm.true_negatives != null ? cm.true_negatives : '—';
      document.getElementById('cmFP').textContent = cm.false_positives != null ? cm.false_positives : '—';
      document.getElementById('cmFN').textContent = cm.false_negatives != null ? cm.false_negatives : '—';
      document.getElementById('cmTP').textContent = cm.true_positives != null ? cm.true_positives : '—';
    } else {
      document.getElementById('bmAcc').textContent = 'Pending';
      document.getElementById('bmAuc').textContent = 'Pending';
      document.getElementById('bmPrec').textContent = 'Pending';
      document.getElementById('bmRec').textContent = 'Pending';
      document.getElementById('bmF1').textContent = 'Pending';
      document.getElementById('bmFpr').textContent = 'Pending';
      document.getElementById('cmTN').textContent = '—';
      document.getElementById('cmFP').textContent = '—';
      document.getElementById('cmFN').textContent = '—';
      document.getElementById('cmTP').textContent = '—';
    }

    benchmarksModal.classList.add('open');
  } catch (e) {
    toast('⚠️', 'Benchmark Error', 'Could not load benchmark metrics', 3000);
  }
});
closeBenchBtn.addEventListener('click', () => benchmarksModal.classList.remove('open'));

// ── FEATURE D: ANALYSIS HISTORY ────────────────────────────────
function addToHistory(data, filename) {
  const item = {
    id: Date.now(),
    filename: filename,
    verdict: data.result,
    verdict_code: data.verdict_code,
    confidence: data.confidence,
    time: new Date().toLocaleTimeString()
  };
  sessionHistory.unshift(item);
  if (sessionHistory.length > 20) sessionHistory.pop();
  historyCount.textContent = sessionHistory.length;
}

navHistoryBtn.addEventListener('click', () => {
  const list = document.getElementById('historyList');
  if (sessionHistory.length === 0) {
    list.innerHTML = `<p style="color:var(--txt-muted);font-size:12px;text-align:center;padding:20px;">No analyses in session history yet.</p>`;
  } else {
    list.innerHTML = sessionHistory.map(h => `
      <div class="history-item">
        <div>
          <strong style="color:var(--txt-white);">${h.filename}</strong>
          <div style="font-size:11px;color:var(--txt-muted);margin-top:2px;">${h.time}</div>
        </div>
        <div style="text-align:right;">
          <span class="badge-res ${h.verdict_code}">${h.verdict}</span>
          <div style="font-size:11px;color:var(--txt-muted);margin-top:2px;">${h.confidence.toFixed(1)}% conf</div>
        </div>
      </div>
    `).join('');
  }
  historyModal.classList.add('open');
});
closeHistBtn.addEventListener('click', () => historyModal.classList.remove('open'));
clearHistoryBtn.addEventListener('click', () => {
  sessionHistory = [];
  historyCount.textContent = '0';
  document.getElementById('historyList').innerHTML = `<p style="color:var(--txt-muted);font-size:12px;text-align:center;padding:20px;">Session history cleared.</p>`;
});

// Close submodals when clicking backdrop
[diagnosticsModal, benchmarksModal, metadataModal, historyModal].forEach(m => {
  m.addEventListener('click', e => {
    if (e.target === m) m.classList.remove('open');
  });
});

// ── FEATURE C: BATCH ANALYSIS ──────────────────────────────────
batchFileInput.addEventListener('change', () => {
  batchFiles = Array.from(batchFileInput.files).slice(0, 10);
  renderBatchQueue();
});

function renderBatchQueue() {
  if (batchFiles.length === 0) {
    batchQueue.style.display = 'none';
    runBatchBtn.disabled = true;
    return;
  }
  batchQueue.style.display = 'block';
  batchQueue.innerHTML = batchFiles.map(f => `
    <div class="batch-queue-item">
      <span>📄 ${f.name}</span>
      <span style="color:var(--txt-muted);">${(f.size / 1024).toFixed(1)} KB</span>
    </div>
  `).join('');
  runBatchBtn.disabled = false;
  runBatchBtn.querySelector('span').textContent = `Analyze ${batchFiles.length} Images`;
}

runBatchBtn.addEventListener('click', async () => {
  if (batchFiles.length === 0) return;
  runBatchBtn.disabled = true;
  runBatchBtn.querySelector('span').textContent = 'Analyzing Batch…';

  try {
    const fd = new FormData();
    batchFiles.forEach(f => fd.append('images', f));
    fd.append('model', selectedModel);

    const res = await fetch('/analyze_batch', {
      method: 'POST',
      body: fd
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();

    batchResults = data.results || [];
    renderBatchTable(batchResults);
    toast('📊', 'Batch Complete', `Analyzed ${batchResults.length} images successfully.`, 3000);
  } catch (e) {
    toast('⚠️', 'Batch Failed', e.message || 'Error occurred during batch analysis.', 4000);
  } finally {
    runBatchBtn.disabled = false;
    runBatchBtn.querySelector('span').textContent = `Analyze ${batchFiles.length} Images`;
  }
});

function renderBatchTable(results) {
  batchTableWrap.style.display = 'block';
  downloadBatchCsvBtn.style.display = 'inline-flex';

  batchTableBody.innerHTML = results.map(r => {
    if (r.status === 'error') {
      return `<tr><td>${r.filename}</td><td colspan="6" style="color:var(--red);">Error: ${r.error}</td></tr>`;
    }
    const ms = r.model_scores || {};
    const ma = r.model_agreement || {};
    return `
      <tr>
        <td><strong>${r.filename}</strong></td>
        <td><span class="badge-res ${r.verdict_code}">${r.result}</span></td>
        <td>${r.ai_probability != null ? r.ai_probability.toFixed(1) + '%' : '—'}</td>
        <td>${r.confidence != null ? r.confidence.toFixed(1) + '%' : '—'}</td>
        <td>${ms.densenet121_ai != null ? ms.densenet121_ai.toFixed(1) + '%' : '—'}</td>
        <td>${ms.resnet50_ai != null ? ms.resnet50_ai.toFixed(1) + '%' : '—'}</td>
        <td>${ma.agree ? '✓ Concordant' : '⚠️ Divergent'}</td>
      </tr>
    `;
  }).join('');
}

downloadBatchCsvBtn.addEventListener('click', () => {
  if (batchResults.length === 0) return;
  let csv = 'Filename,Verdict,VerdictCode,AI_Probability,Confidence,DenseNet_AI,ResNet_AI,Agreement\n';
  batchResults.forEach(r => {
    if (r.status === 'success') {
      const ms = r.model_scores || {};
      const ma = r.model_agreement || {};
      csv += `"${r.filename}","${r.result}","${r.verdict_code}",${r.ai_probability},${r.confidence},${ms.densenet121_ai || 0},${ms.resnet50_ai || 0},${ma.agree ? 'Concordant' : 'Divergent'}\n`;
    }
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `DeepShield_Batch_Results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  toast('📥', 'CSV Exported', 'Batch evaluation CSV report downloaded', 2200);
});