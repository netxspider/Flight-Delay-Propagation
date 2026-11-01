// Flight Delay Propagation Research Dashboard Client Logic

const API_BASE = window.location.origin;

// State management
const state = {
  activeView: "overview",
  dataset: {
    partition: "test",
    page: 1,
    pageSize: 15,
    totalPages: 1,
    totalRecords: 0,
    sortBy: "NEXT_ARR_DELAY",
    order: "desc",
    carrier: "",
    origin: "",
    dest: "",
    status: "all",
    deficit: "all",
    search: ""
  },
  threshold: 0.35,
  modelsData: null,
  summaryData: null,
  figures: {
    catalog: [],
    filtered: [],
    activeCategory: "all",
    searchQuery: "",
    currentModalIndex: 0,
    zoomLevel: 1.0
  }
};

// Initialize application on DOM loaded
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initPresetButtons();
  initPredictionForm();
  initThresholdSlider();
  initDatasetFilters();
  initFiguresGallery();
  
  // Load initial backend data
  loadOverviewData();
  loadDatasetSummary();
  loadDatasetSample();
  loadModelsMetrics();
  loadFiguresCatalog();

  // Run sample prediction
  runPrediction();
});

// Navigation Handling
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = item.getAttribute("data-view");
      switchView(targetView);
    });
  });

  // Handle URL hash routing
  if (window.location.hash) {
    const hashView = window.location.hash.replace("#", "");
    const validViews = ["overview", "dataset", "methodology", "eda", "models", "results", "visualizations", "conclusion"];
    if (validViews.includes(hashView)) {
      switchView(hashView);
    }
  }
}

function switchView(viewName) {
  state.activeView = viewName;
  window.location.hash = viewName;

  // Update active nav class
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.getAttribute("data-view") === viewName);
  });

  // Update visible page section
  document.querySelectorAll(".dashboard-page").forEach(page => {
    page.classList.toggle("active", page.id === `${viewName}-view`);
  });

  if (viewName === "dataset") {
    loadDatasetSample();
  }

  // Update top bar title
  const titles = {
    overview: "01 \u2014 Overview: Research Question & Abstract",
    dataset: "02 \u2014 Dataset: Source, Scale & Composition",
    methodology: "03 \u2014 Methodology: Aircraft Rotation Reconstruction",
    eda: "04 \u2014 Exploratory Analysis: Delay Propagation Patterns",
    models: "05 \u2014 Models: Training & Model Comparison",
    results: "06 \u2014 Results: Test Metrics & Evaluation",
    visualizations: "07 \u2014 Visualizations: Figures & Interactive Charts",
    conclusion: "08 \u2014 Conclusion: Findings & Limitations"
  };
  const titleEl = document.getElementById("top-bar-page-title");
  if (titleEl) {
    titleEl.textContent = titles[viewName] || "Research Dashboard";
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function openFigureModalByFilename(filename) {
  let catalog = state.figures.catalog;
  if (!catalog || catalog.length === 0) {
    try {
      const res = await fetch(`${API_BASE}/api/figures`);
      const data = await res.json();
      if (data.figures && data.figures.length > 0) {
        state.figures.catalog = data.figures;
        catalog = data.figures;
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!catalog || catalog.length === 0) return;
  const idx = catalog.findIndex(f => f.filename === filename);
  if (idx !== -1) {
    state.figures.filtered = catalog;
    openFigureModal(idx);
  }
}
window.openFigureModalByFilename = openFigureModalByFilename;

// Overview Data Loader
async function loadOverviewData() {
  try {
    const res = await fetch(`${API_BASE}/api/overview`);
    const data = await res.json();
    if (data.key_metrics) {
      const elTot = document.getElementById("stat-total-rotations");
      if (elTot) elTot.textContent = data.key_metrics.total_rotations.toLocaleString();
      const elAuc = document.getElementById("stat-test-auc");
      if (elAuc) elAuc.textContent = data.key_metrics.test_roc_auc.toFixed(4);
      const elR2 = document.getElementById("stat-test-r2");
      if (elR2) elR2.textContent = (data.key_metrics.test_r2 * 100).toFixed(1) + "%";
      const elDef = document.getElementById("stat-deficit-delay-rate");
      if (elDef) elDef.textContent = (data.key_metrics.deficit_delay_rate * 100).toFixed(1) + "%";
    }
  } catch (err) {
    console.error("Error loading overview data:", err);
  }
}

// Dataset Summary Loader
async function loadDatasetSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/dataset/summary`);
    const data = await res.json();
    state.summaryData = data;
    renderDatasetSummary(data);
  } catch (err) {
    console.error("Error loading dataset summary:", err);
  }
}

function renderDatasetSummary(data) {
  // Render pipeline table
  const tbody = document.getElementById("pipeline-stages-tbody");
  if (tbody && data.pipeline_stages) {
    tbody.innerHTML = data.pipeline_stages.map(stage => `
      <tr>
        <td style="font-weight: 500; color: #fff;">${stage.stage}</td>
        <td style="font-family: monospace; text-align: right;">${stage.count.toLocaleString()}</td>
        <td style="font-family: monospace; text-align: right; color: var(--accent-blue);">${stage.percentage.toFixed(2)}%</td>
      </tr>
    `).join("");
  }

  // Populate airport dropdowns
  const originSelect = document.getElementById("filter-origin");
  const destSelect = document.getElementById("filter-dest");
  const predOrigin = document.getElementById("pred-origin");
  const predDest = document.getElementById("pred-dest");

  if (data.top_airports) {
    const opts = data.top_airports.map(a => `<option value="${a}">${a}</option>`).join("");
    if (originSelect) originSelect.innerHTML = `<option value="">All Origins</option>` + opts;
    if (destSelect) destSelect.innerHTML = `<option value="">All Destinations</option>` + opts;
    if (predOrigin) predOrigin.innerHTML = opts;
    if (predDest) predDest.innerHTML = opts;
    if (predOrigin) predOrigin.value = "ORD";
    if (predDest) predDest.value = "DFW";
  }
}

// Dataset Sample Table Loader (DuckDB API)
async function loadDatasetSample() {
  const tbody = document.getElementById("dataset-sample-tbody");
  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 28px; color: var(--text-muted);">
      <div class="radar-beacon" style="display: inline-block; margin-right: 8px;"></div> Querying Parquet partition via DuckDB engine...
    </td></tr>`;
  }

  const p = state.dataset;
  const params = new URLSearchParams({
    partition: p.partition || "test",
    page: p.page || 1,
    page_size: p.pageSize || 15,
    sort_by: p.sortBy || "NEXT_ARR_DELAY",
    order: p.order || "desc",
    carrier: p.carrier || "",
    status: p.status || "all",
    deficit: p.deficit || "all",
    search: p.search || ""
  });

  try {
    const res = await fetch(`${API_BASE}/api/dataset/sample?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();
    if (data.error) {
      if (tbody) tbody.innerHTML = `<tr><td colspan="10" style="color: var(--accent-rose); text-align: center; padding: 20px;">Query error: ${data.error}</td></tr>`;
      return;
    }

    state.dataset.totalRecords = data.total || 0;
    state.dataset.totalPages = data.total_pages || 1;

    renderSampleTable(data);
  } catch (err) {
    console.error("DuckDB query failed:", err);
    if (tbody) tbody.innerHTML = `<tr><td colspan="10" style="color: var(--accent-rose); text-align: center; padding: 20px;">Error querying dataset. Ensure server.py is running.</td></tr>`;
  }
}

function renderSampleTable(data) {
  const tbody = document.getElementById("dataset-sample-tbody");
  if (!tbody) return;

  const records = data.data || [];
  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 28px; color: var(--text-muted);">No matching flights found for the current filter criteria.</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map(row => {
    const isDelayed = row.NEXT_ARR_DEL15 === 1;
    const isDeficit = row.IS_TURNAROUND_DEFICIT === 1;
    const cushion = Number(row.ESTIMATED_REMAINING_TURNAROUND_MINUTES || 0);
    const prevDelay = Number(row.PREV_ARR_DELAY || 0);
    const buffer = Number(row.SCHEDULED_TURNAROUND_BUFFER || 0);
    const nextDelay = Number(row.NEXT_ARR_DELAY || 0);
    const depHour = Number(row.NEXT_CRS_DEP_HOUR || 0);

    return `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 600; color: #fff;">${row.TAIL_NUM || "N/A"}</td>
        <td><span class="badge-pill" style="color: var(--accent-cyan); font-size: 11px; padding: 2px 7px;">${row.NEXT_CARRIER || "N/A"}</span></td>
        <td><strong style="color: #fff;">${row.NEXT_ORIGIN || "---"}</strong> &rarr; <strong style="color: #fff;">${row.NEXT_DEST || "---"}</strong></td>
        <td style="font-family: var(--font-mono); text-align: right; color: ${prevDelay >= 15 ? 'var(--accent-rose)' : 'var(--text-secondary)'};">
          ${prevDelay > 0 ? '+' : ''}${prevDelay.toFixed(0)}m
        </td>
        <td style="font-family: var(--font-mono); text-align: right;">${buffer.toFixed(0)}m</td>
        <td style="font-family: var(--font-mono); text-align: right; font-weight: 600; color: ${cushion < 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">
          ${cushion > 0 ? '+' : ''}${cushion.toFixed(0)}m
        </td>
        <td style="text-align: center;">
          ${isDeficit 
            ? `<span class="badge-pill" style="color: var(--accent-rose); border-color: rgba(244,63,94,0.3); background: rgba(244,63,94,0.08); font-size: 10.5px;">DEFICIT</span>` 
            : `<span class="badge-pill" style="color: var(--accent-emerald); border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.08); font-size: 10.5px;">BUFFER OK</span>`}
        </td>
        <td style="font-family: var(--font-mono); text-align: right;">${depHour}:00</td>
        <td style="font-family: var(--font-mono); text-align: right; font-weight: 700; color: ${nextDelay >= 15 ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">
          ${nextDelay > 0 ? '+' : ''}${nextDelay.toFixed(0)}m
        </td>
        <td style="text-align: center;">
          ${isDelayed 
            ? `<span class="badge-pill" style="color: var(--accent-rose); border-color: rgba(244,63,94,0.3); background: rgba(244,63,94,0.08); font-size: 10.5px;">DELAYED</span>` 
            : `<span class="badge-pill" style="color: var(--accent-emerald); border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.08); font-size: 10.5px;">ON-TIME</span>`}
        </td>
      </tr>
    `;
  }).join("");

  // Update pagination indicators safely
  const page = data.page || 1;
  const totalPages = data.total_pages || 1;
  const totalRecords = data.total || 0;

  const elCurrent = document.getElementById("page-current");
  if (elCurrent) elCurrent.textContent = page.toLocaleString();

  const elTotal = document.getElementById("page-total");
  if (elTotal) elTotal.textContent = totalPages.toLocaleString();

  const elCount = document.getElementById("record-count");
  if (elCount) elCount.textContent = totalRecords.toLocaleString();

  const elFilterCount = document.getElementById("dataset-filter-count");
  if (elFilterCount) elFilterCount.textContent = `${totalRecords.toLocaleString()} Matching Flights`;

  const btnPrev = document.getElementById("btn-prev");
  if (btnPrev) btnPrev.disabled = (page <= 1);

  const btnNext = document.getElementById("btn-next");
  if (btnNext) btnNext.disabled = (page >= totalPages);

  const btnFirst = document.getElementById("btn-first");
  if (btnFirst) btnFirst.disabled = (page <= 1);

  const btnLast = document.getElementById("btn-last");
  if (btnLast) btnLast.disabled = (page >= totalPages);
}

// Dataset Filters
function initDatasetFilters() {
  const partitionSelect = document.getElementById("filter-partition");
  const carrierSelect = document.getElementById("filter-carrier");
  const statusSelect = document.getElementById("filter-status");
  const deficitSelect = document.getElementById("filter-deficit");
  const searchInput = document.getElementById("filter-search");
  const resetBtn = document.getElementById("filter-reset-btn");

  const triggerUpdate = () => {
    state.dataset.page = 1;
    if (partitionSelect) state.dataset.partition = partitionSelect.value;
    if (carrierSelect) state.dataset.carrier = carrierSelect.value;
    if (statusSelect) state.dataset.status = statusSelect.value;
    if (deficitSelect) state.dataset.deficit = deficitSelect.value;
    if (searchInput) state.dataset.search = searchInput.value.trim();
    loadDatasetSample();
  };

  [partitionSelect, carrierSelect, statusSelect, deficitSelect].forEach(el => {
    if (el) el.addEventListener("change", triggerUpdate);
  });

  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(triggerUpdate, 300);
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (partitionSelect) partitionSelect.value = "test";
      if (carrierSelect) carrierSelect.value = "";
      if (statusSelect) statusSelect.value = "all";
      if (deficitSelect) deficitSelect.value = "all";
      if (searchInput) searchInput.value = "";
      triggerUpdate();
    });
  }

  // Sortable headers
  document.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const sortField = th.getAttribute("data-col") || th.getAttribute("data-sort");
      if (!sortField) return;
      if (state.dataset.sortBy === sortField) {
        state.dataset.order = (state.dataset.order === "desc") ? "asc" : "desc";
      } else {
        state.dataset.sortBy = sortField;
        state.dataset.order = "desc";
      }
      loadDatasetSample();
    });
  });

  // Pagination buttons
  const btnFirst = document.getElementById("btn-first");
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  const btnLast = document.getElementById("btn-last");

  if (btnFirst) {
    btnFirst.addEventListener("click", () => {
      if (state.dataset.page > 1) {
        state.dataset.page = 1;
        loadDatasetSample();
      }
    });
  }
  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (state.dataset.page > 1) {
        state.dataset.page--;
        loadDatasetSample();
      }
    });
  }
  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (state.dataset.page < state.dataset.totalPages) {
        state.dataset.page++;
        loadDatasetSample();
      }
    });
  }
  if (btnLast) {
    btnLast.addEventListener("click", () => {
      if (state.dataset.page < state.dataset.totalPages) {
        state.dataset.page = state.dataset.totalPages;
        loadDatasetSample();
      }
    });
  }
}

// Models & Metrics Loader
async function loadModelsMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    state.modelsData = data;
    renderModelsSection(data);
  } catch (err) {
    console.error("Error loading model metrics:", err);
  }
}

function renderModelsSection(data) {
  // Classification Table
  const clfTbody = document.getElementById("models-classification-tbody");
  if (clfTbody && data.classification_validation) {
    clfTbody.innerHTML = data.classification_validation.map(row => {
      const isSelected = row.estimator.includes("*");
      return `
        <tr style="${isSelected ? 'background: rgba(56, 189, 248, 0.08); font-weight: 600;' : ''}">
          <td style="color: ${isSelected ? 'var(--accent-blue)' : '#fff'};">${row.estimator}</td>
          <td><span class="brand-badge" style="padding: 2px 6px;">${row.group}</span></td>
          <td style="text-align: right; font-family: monospace;">${row.accuracy.toFixed(2)}%</td>
          <td style="text-align: right; font-family: monospace;">${row.precision.toFixed(2)}%</td>
          <td style="text-align: right; font-family: monospace;">${row.recall.toFixed(2)}%</td>
          <td style="text-align: right; font-family: monospace;">${row.f1.toFixed(4)}</td>
          <td style="text-align: right; font-family: monospace; color: var(--accent-blue);">${row.roc_auc.toFixed(4)}</td>
          <td style="text-align: right; font-family: monospace; color: var(--accent-emerald);">${row.ap.toFixed(4)}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${row.fit_time}</td>
        </tr>
      `;
    }).join("");
  }

  // Regression Table
  const regTbody = document.getElementById("models-regression-tbody");
  if (regTbody && data.regression_validation) {
    regTbody.innerHTML = data.regression_validation.map(row => {
      const isSelected = row.model.includes("*");
      return `
        <tr style="${isSelected ? 'background: rgba(56, 189, 248, 0.08); font-weight: 600;' : ''}">
          <td style="color: ${isSelected ? 'var(--accent-blue)' : '#fff'};">${row.model}</td>
          <td><span class="brand-badge" style="padding: 2px 6px;">${row.group}</span></td>
          <td style="text-align: right; font-family: monospace;">${row.mae.toFixed(2)} min</td>
          <td style="text-align: right; font-family: monospace;">${row.rmse.toFixed(2)} min</td>
          <td style="text-align: right; font-family: monospace; color: ${row.r2 > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${row.r2.toFixed(4)}</td>
          <td style="text-align: right; font-family: monospace; color: var(--text-muted);">${row.fit_time}</td>
        </tr>
      `;
    }).join("");
  }

  // Delay Strata Table
  const strataTbody = document.getElementById("delay-strata-tbody");
  if (strataTbody && data.delay_strata_test) {
    strataTbody.innerHTML = data.delay_strata_test.map(row => `
      <tr>
        <td style="font-weight: 600; color: #fff;">${row.stratum}</td>
        <td style="text-align: right; font-family: monospace;">${row.count.toLocaleString()}</td>
        <td style="text-align: right; font-family: monospace;">${row.share_pct.toFixed(2)}%</td>
        <td style="text-align: right; font-family: monospace; color: ${row.mae > 30 ? 'var(--accent-rose)' : 'var(--text-primary)'};">${row.mae.toFixed(2)} m</td>
        <td style="text-align: right; font-family: monospace;">${row.rmse.toFixed(2)} m</td>
        <td style="text-align: right; font-family: monospace; color: ${row.bias > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${row.bias > 0 ? '+' : ''}${row.bias.toFixed(2)} m</td>
      </tr>
    `).join("");
  }

  // Shift-Share Summary
  if (data.shift_share_decomposition) {
    const ssd = data.shift_share_decomposition;
    document.getElementById("ss-total-delta").textContent = `+${ssd.total_delta_mae.toFixed(2)} min`;
    document.getElementById("ss-comp-effect").textContent = `+${ssd.formulation_a.composition_shift_effect.toFixed(2)} min (${ssd.formulation_a.composition_shift_pct.toFixed(1)}%)`;
    document.getElementById("ss-within-effect").textContent = `+${ssd.formulation_a.within_band_effect.toFixed(2)} min (${ssd.formulation_a.within_band_pct.toFixed(1)}%)`;
  }

  // Permutation Importance Table
  const permTbody = document.getElementById("permutation-tbody");
  if (permTbody && data.permutation_importance) {
    permTbody.innerHTML = data.permutation_importance.map(row => `
      <tr>
        <td style="font-family: monospace; font-weight: 700; color: var(--accent-blue); text-align: center;">#${row.rank}</td>
        <td style="font-family: monospace; font-weight: 600; color: #fff;">${row.feature}</td>
        <td><span class="brand-badge" style="font-size: 10px;">${row.domain}</span></td>
        <td style="text-align: right; font-family: monospace; color: var(--accent-emerald);">+${row.ap_drop.toFixed(4)}</td>
        <td style="text-align: right; font-family: monospace; color: var(--accent-amber);">+${row.mae_increase.toFixed(2)} min</td>
        <td style="text-align: right; font-family: monospace; color: var(--accent-blue);">${row.r2_drop > 0 ? '+' : ''}${row.r2_drop.toFixed(4)}</td>
      </tr>
    `).join("");
  }

  // Test Generalization Table
  const testGenTbody = document.getElementById("test-gen-tbody");
  if (testGenTbody && data.test_generalization) {
    testGenTbody.innerHTML = data.test_generalization.map(row => {
      const isPositive = String(row.delta).startsWith("+");
      const deltaColor = (row.metric.includes("MAE") || row.metric.includes("RMSE") || row.metric.includes("Log Loss"))
        ? (isPositive ? "var(--accent-rose)" : "var(--accent-emerald)")
        : (isPositive ? "var(--accent-emerald)" : "var(--accent-rose)");
      return `
        <tr>
          <td><strong style="color: #fff;">${row.metric}</strong></td>
          <td style="text-align: right; font-family: var(--font-mono);">${row.validation}</td>
          <td style="text-align: right; font-family: var(--font-mono); font-weight: 700; color: #fff;">${row.test}</td>
          <td style="text-align: right; font-family: var(--font-mono); font-weight: 600; color: ${deltaColor};">${row.delta}</td>
          <td style="color: var(--text-secondary); font-size: 12px;">${row.interpretation}</td>
        </tr>
      `;
    }).join("");
  }
}

// Interactive Threshold Simulator
function initThresholdSlider() {
  const slider = document.getElementById("threshold-slider");
  const valueDisplay = document.getElementById("threshold-val-display");

  if (!slider) return;

  slider.addEventListener("input", () => {
    const t = parseFloat(slider.value);
    state.threshold = t;
    if (valueDisplay) valueDisplay.textContent = t.toFixed(2);
    updateThresholdSimulation(t);
  });
}

function updateThresholdSimulation(t) {
  // Approximate empirical response curve from test dataset evaluation
  // Total N = 562,202, Positives = 148,865 (26.48%), Negatives = 413,337
  let recall, precision, fpr;

  if (t <= 0.35) {
    // Between 0.10 and 0.35
    const frac = (t - 0.10) / 0.25;
    recall = 0.85 - frac * (0.85 - 0.5324);
    precision = 0.42 + frac * (0.7252 - 0.42);
    fpr = 0.28 - frac * (0.28 - 0.0726);
  } else if (t <= 0.50) {
    // Between 0.35 and 0.50
    const frac = (t - 0.35) / 0.15;
    recall = 0.5324 - frac * (0.5324 - 0.4088);
    precision = 0.7252 + frac * (0.8668 - 0.7252);
    fpr = 0.0726 - frac * (0.0726 - 0.0210);
  } else {
    // Above 0.50
    const frac = (t - 0.50) / 0.40;
    recall = 0.4088 - frac * 0.32;
    precision = Math.min(0.96, 0.8668 + frac * 0.08);
    fpr = Math.max(0.005, 0.0210 - frac * 0.018);
  }

  const f1 = (2 * precision * recall) / (precision + recall);
  const totalPos = 148865;
  const totalNeg = 413337;
  const tp = Math.round(recall * totalPos);
  const fn = totalPos - tp;
  const fp = Math.round(fpr * totalNeg);
  const tn = totalNeg - fp;

  document.getElementById("sim-prec").textContent = (precision * 100).toFixed(1) + "%";
  document.getElementById("sim-recall").textContent = (recall * 100).toFixed(1) + "%";
  document.getElementById("sim-f1").textContent = f1.toFixed(4);
  document.getElementById("sim-fpr").textContent = (fpr * 100).toFixed(1) + "%";

  document.getElementById("cm-tp").textContent = tp.toLocaleString();
  document.getElementById("cm-fp").textContent = fp.toLocaleString();
  document.getElementById("cm-fn").textContent = fn.toLocaleString();
  document.getElementById("cm-tn").textContent = tn.toLocaleString();

  // Highlight badge if optimal
  const badge = document.getElementById("threshold-badge");
  if (badge) {
    if (Math.abs(t - 0.35) < 0.02) {
      badge.textContent = "Recommended Operational Point (T = 0.35)";
      badge.style.color = "var(--accent-blue)";
    } else if (Math.abs(t - 0.39) < 0.02) {
      badge.textContent = "Mathematical F1 Peak on Grid (T = 0.39)";
      badge.style.color = "var(--accent-emerald)";
    } else if (Math.abs(t - 0.50) < 0.02) {
      badge.textContent = "Default SciKit-Learn Threshold (T = 0.50)";
      badge.style.color = "var(--text-muted)";
    } else {
      badge.textContent = `Custom Operating Point (T = ${t.toFixed(2)})`;
      badge.style.color = "var(--accent-amber)";
    }
  }
}

function setSimulatorThreshold(val) {
  const slider = document.getElementById("threshold-slider");
  if (slider) {
    slider.value = val;
    state.threshold = val;
    const valueDisplay = document.getElementById("threshold-val-display");
    if (valueDisplay) valueDisplay.textContent = Number(val).toFixed(2);
    updateThresholdSimulation(val);
  }
}
window.setSimulatorThreshold = setSimulatorThreshold;

// Interactive Live Prediction Playground
function initPredictionForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });

  // Attach dynamic range slider number updates
  const bindSlider = (sliderId, textId, suffix = "") => {
    const s = document.getElementById(sliderId);
    const t = document.getElementById(textId);
    if (s && t) {
      s.addEventListener("input", () => {
        t.textContent = `${s.value}${suffix}`;
        runPrediction();
      });
    }
  };

  bindSlider("pred-prev-delay", "pred-prev-delay-val", "m");
  bindSlider("pred-buffer", "pred-buffer-val", "m");
  bindSlider("pred-hour", "pred-hour-val", ":00");
  bindSlider("pred-distance", "pred-distance-val", " mi");
  bindSlider("pred-delayed-today", "pred-delayed-today-val", " flights");
}

function initPresetButtons() {
  const presets = {
    deficit: {
      prev_delay: 75,
      buffer: 45,
      carrier: "AA",
      origin: "ORD",
      dest: "DFW",
      hour: 16,
      distance: 802,
      delayed_today: 2
    },
    cushion_ok: {
      prev_delay: 20,
      buffer: 65,
      carrier: "DL",
      origin: "ATL",
      dest: "MCO",
      hour: 11,
      distance: 404,
      delayed_today: 0
    },
    fatigued: {
      prev_delay: 45,
      buffer: 40,
      carrier: "WN",
      origin: "DEN",
      dest: "PHX",
      hour: 20,
      distance: 602,
      delayed_today: 3
    },
    ontime: {
      prev_delay: 0,
      buffer: 55,
      carrier: "UA",
      origin: "SFO",
      dest: "SEA",
      hour: 9,
      distance: 679,
      delayed_today: 0
    }
  };

  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      const p = presets[type];
      if (!p) return;

      document.getElementById("pred-prev-delay").value = p.prev_delay;
      document.getElementById("pred-prev-delay-val").textContent = `${p.prev_delay}m`;

      document.getElementById("pred-buffer").value = p.buffer;
      document.getElementById("pred-buffer-val").textContent = `${p.buffer}m`;

      document.getElementById("pred-carrier").value = p.carrier;
      document.getElementById("pred-origin").value = p.origin;
      document.getElementById("pred-dest").value = p.dest;

      document.getElementById("pred-hour").value = p.hour;
      document.getElementById("pred-hour-val").textContent = `${p.hour}:00`;

      document.getElementById("pred-distance").value = p.distance;
      document.getElementById("pred-distance-val").textContent = `${p.distance} mi`;

      document.getElementById("pred-delayed-today").value = p.delayed_today;
      document.getElementById("pred-delayed-today-val").textContent = `${p.delayed_today} flights`;

      runPrediction();
    });
  });
}

async function runPrediction() {
  const prevDelay = parseFloat(document.getElementById("pred-prev-delay")?.value || 0);
  const buffer = parseFloat(document.getElementById("pred-buffer")?.value || 50);
  const carrier = document.getElementById("pred-carrier")?.value || "AA";
  const origin = document.getElementById("pred-origin")?.value || "ORD";
  const dest = document.getElementById("pred-dest")?.value || "DFW";
  const hour = parseInt(document.getElementById("pred-hour")?.value || 15, 10);
  const distance = parseFloat(document.getElementById("pred-distance")?.value || 800);
  const delayedToday = parseInt(document.getElementById("pred-delayed-today")?.value || 0, 10);

  const payload = {
    prev_arr_delay: prevDelay,
    prev_dep_delay: Math.max(0, prevDelay - 8),
    scheduled_turnaround_buffer: buffer,
    carrier: carrier,
    origin: origin,
    dest: dest,
    crs_dep_hour: hour,
    day_of_week: 5,
    month: 12,
    distance: distance,
    aircraft_delayed_flights_today: delayedToday,
    aircraft_consecutive_del15: delayedToday > 0 ? 1 : 0,
    aircraft_cumulative_delay_today: delayedToday * 30.0 + prevDelay,
    aircraft_delay_rate_today: delayedToday > 0 ? (delayedToday / 4.0) : 0.0
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    renderPredictionResult(result);
  } catch (err) {
    console.error("Prediction API failed:", err);
  }
}

function renderPredictionResult(res) {
  // Gauge Percentage
  const pctEl = document.getElementById("gauge-pct-display");
  const circleEl = document.getElementById("gauge-circle-box");
  const verdictEl = document.getElementById("pred-verdict-banner");
  const delayEl = document.getElementById("pred-continuous-delay-val");
  const cushionEl = document.getElementById("pred-cushion-val");
  const recommendationEl = document.getElementById("pred-recommendation-text");
  const driversEl = document.getElementById("pred-drivers-list");

  if (pctEl) pctEl.textContent = `${res.probability_pct}%`;
  if (circleEl) {
    circleEl.style.borderColor = res.risk_color;
    circleEl.style.boxShadow = `0 0 25px ${res.risk_color}33`;
  }

  if (verdictEl) {
    verdictEl.textContent = `${res.risk_level} — ${res.operating_verdict}`;
    verdictEl.style.color = res.risk_color;
  }

  if (delayEl) {
    delayEl.textContent = `${res.predicted_delay_minutes > 0 ? '+' : ''}${res.predicted_delay_minutes.toFixed(1)} min`;
    delayEl.style.color = res.predicted_delay_minutes >= 15 ? "var(--accent-rose)" : "var(--accent-emerald)";
  }

  if (cushionEl) {
    const c = res.turnaround_analysis.remaining_cushion;
    cushionEl.textContent = `${c > 0 ? '+' : ''}${c.toFixed(0)} min (${res.turnaround_analysis.is_deficit ? 'DEFICIT' : 'SUFFICIENT'})`;
    cushionEl.style.color = c < 0 ? "var(--accent-crimson)" : "var(--accent-emerald)";
  }

  if (recommendationEl) {
    recommendationEl.textContent = res.operational_recommendation;
  }

  if (driversEl && res.key_drivers) {
    driversEl.innerHTML = res.key_drivers.map(d => `
      <li style="margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
        <span style="color: var(--accent-blue); font-size: 14px;">&#9654;</span> ${d}
      </li>
    `).join("");
  }
}

// Copy Google Drive Snippet helper
function copySnippet(elementId) {
  const el = document.getElementById(elementId);
  if (el) {
    navigator.clipboard.writeText(el.innerText).then(() => {
      alert("Code snippet copied to clipboard!");
    });
  }
}
window.copySnippet = copySnippet;

// =============================================================================
// RESEARCH FIGURES & VISUALIZATIONS GALLERY (32+ FIGURES)
// =============================================================================

function initFiguresGallery() {
  // Category Pill Buttons
  const catPills = document.querySelectorAll("#figure-category-pills .cat-pill");
  catPills.forEach(pill => {
    pill.addEventListener("click", () => {
      catPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.figures.activeCategory = pill.getAttribute("data-cat");
      filterAndRenderFigures();
    });
  });

  // Search Input Listener
  const searchInput = document.getElementById("figures-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      state.figures.searchQuery = e.target.value.toLowerCase().trim();
      filterAndRenderFigures();
    });
  }

  // Clear Search Button
  const clearBtn = document.getElementById("clear-figures-search");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (searchInput) searchInput.value = "";
      state.figures.searchQuery = "";
      filterAndRenderFigures();
    });
  }

  // Keyboard navigation for Lightbox modal
  document.addEventListener("keydown", (e) => {
    const modal = document.getElementById("figure-modal");
    if (modal && modal.style.display !== "none") {
      if (e.key === "Escape") {
        closeFigureModal();
      } else if (e.key === "ArrowLeft") {
        navigateFigure(-1);
      } else if (e.key === "ArrowRight") {
        navigateFigure(1);
      }
    }
  });
}

async function loadFiguresCatalog() {
  try {
    const res = await fetch(`${API_BASE}/api/figures`);
    const data = await res.json();
    if (data.figures && data.figures.length > 0) {
      state.figures.catalog = data.figures;
      filterAndRenderFigures();
    }
  } catch (err) {
    console.error("Failed to load figures catalog:", err);
  }
}

function filterAndRenderFigures() {
  const { catalog, activeCategory, searchQuery } = state.figures;
  
  let filtered = catalog;

  // Filter by category
  if (activeCategory && activeCategory !== "all") {
    filtered = filtered.filter(f => f.category_id === activeCategory);
  }

  // Filter by search query
  if (searchQuery) {
    filtered = filtered.filter(f => {
      const titleMatch = (f.title || "").toLowerCase().includes(searchQuery);
      const descMatch = (f.description || "").toLowerCase().includes(searchQuery);
      const stageMatch = (f.stage || "").toLowerCase().includes(searchQuery);
      const catMatch = (f.category || "").toLowerCase().includes(searchQuery);
      const fileMatch = (f.filename || "").toLowerCase().includes(searchQuery);
      return titleMatch || descMatch || stageMatch || catMatch || fileMatch;
    });
  }

  state.figures.filtered = filtered;

  // Update gallery counter badge
  const counterBadge = document.getElementById("gallery-count-badge");
  if (counterBadge) {
    counterBadge.textContent = `Showing ${filtered.length} of ${catalog.length} Figures`;
  }

  renderFiguresGrid(filtered);
}

function renderFiguresGrid(figures) {
  const grid = document.getElementById("figures-gallery-grid");
  if (!grid) return;

  if (figures.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 48px; background: var(--bg-card); border-radius: var(--radius-md); border: 1px dashed var(--border-color);">
        <p style="font-size: 16px; color: var(--text-muted); margin-bottom: 8px;">No research figures match the selected filter criteria.</p>
        <button class="btn-secondary" onclick="resetFigureFilters()">Reset All Filters</button>
      </div>
    `;
    return;
  }

  grid.innerHTML = figures.map((fig, idx) => {
    // Stage color accents
    let stageColor = "var(--accent-blue)";
    let stageBg = "rgba(56, 189, 248, 0.1)";
    let stageBorder = "rgba(56, 189, 248, 0.3)";

    if (fig.stage.includes("Stage 6") || fig.stage.includes("Test")) {
      stageColor = "var(--accent-emerald)";
      stageBg = "rgba(16, 185, 129, 0.1)";
      stageBorder = "rgba(16, 185, 129, 0.3)";
    } else if (fig.stage.includes("Tuning") || fig.stage.includes("Stage 5")) {
      stageColor = "var(--accent-purple)";
      stageBg = "rgba(168, 85, 247, 0.1)";
      stageBorder = "rgba(168, 85, 247, 0.3)";
    } else if (fig.stage.includes("IEEE") || fig.stage.includes("Paper")) {
      stageColor = "var(--accent-amber)";
      stageBg = "rgba(245, 158, 11, 0.1)";
      stageBorder = "rgba(245, 158, 11, 0.3)";
    }

    const imgPath = `figures/${fig.filename}`;

    return `
      <div class="figure-gallery-card">
        <div class="figure-card-img-wrap" onclick="openFigureModal(${idx})">
          <img src="${imgPath}" alt="${fig.title}" loading="lazy" onerror="this.src='figures/fig01_target_distributions.png'">
          <div class="figure-card-overlay">
            <button class="figure-overlay-btn">&#128269; View Full Resolution</button>
          </div>
        </div>

        <div class="figure-card-body">
          <div class="figure-card-meta">
            <span class="badge-pill" style="font-size: 10px; padding: 2px 8px; color: ${stageColor}; background: ${stageBg}; border-color: ${stageBorder};">
              ${fig.stage}
            </span>
            <span style="font-size: 11px; color: var(--text-muted);">${fig.source}</span>
          </div>

          <h4 class="figure-card-title">${fig.title}</h4>
          
          <p class="figure-card-desc" title="${fig.description}">
            ${fig.description}
          </p>

          <div class="figure-card-takeaway">
            <strong>Key Insight:</strong> ${fig.takeaway}
          </div>

          <div class="figure-card-footer">
            <span style="font-family: monospace; font-size: 10px; color: var(--accent-blue);">${fig.filename}</span>
            <button class="btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="openFigureModal(${idx})">
              Inspect &rarr;
            </button>
          </div>
        </div>
      </div>
    `;
  }).join("");
}

function resetFigureFilters() {
  state.figures.activeCategory = "all";
  state.figures.searchQuery = "";
  const searchInput = document.getElementById("figures-search-input");
  if (searchInput) searchInput.value = "";
  
  const catPills = document.querySelectorAll("#figure-category-pills .cat-pill");
  catPills.forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-cat") === "all");
  });

  filterAndRenderFigures();
}
window.resetFigureFilters = resetFigureFilters;

// Lightbox Modal Functions
function openFigureModal(index) {
  const figures = state.figures.filtered;
  if (!figures || index < 0 || index >= figures.length) return;

  state.figures.currentModalIndex = index;
  state.figures.zoomLevel = 1.0;

  const fig = figures[index];
  const modal = document.getElementById("figure-modal");
  if (!modal) return;

  // Set modal elements
  document.getElementById("modal-category-badge").textContent = fig.category;
  document.getElementById("modal-stage-badge").textContent = fig.stage;
  document.getElementById("modal-figure-title").textContent = fig.title;
  document.getElementById("modal-figure-desc").textContent = fig.description;
  document.getElementById("modal-figure-takeaway").textContent = fig.takeaway;
  document.getElementById("modal-filename").textContent = fig.filename;
  document.getElementById("modal-source").textContent = fig.source;
  document.getElementById("modal-figure-counter").textContent = `Figure ${index + 1} of ${figures.length}`;

  const imgEl = document.getElementById("modal-figure-img");
  imgEl.src = `figures/${fig.filename}`;
  imgEl.style.transform = `scale(1.0)`;

  const downloadBtn = document.getElementById("modal-download-btn");
  if (downloadBtn) {
    downloadBtn.href = `figures/${fig.filename}`;
    downloadBtn.download = fig.filename;
  }

  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}
window.openFigureModal = openFigureModal;

function closeFigureModal() {
  const modal = document.getElementById("figure-modal");
  if (modal) {
    modal.style.display = "none";
    document.body.style.overflow = "auto";
  }
}
window.closeFigureModal = closeFigureModal;

function navigateFigure(direction) {
  const figures = state.figures.filtered;
  if (!figures || figures.length === 0) return;

  let newIndex = state.figures.currentModalIndex + direction;
  if (newIndex < 0) newIndex = figures.length - 1;
  if (newIndex >= figures.length) newIndex = 0;

  openFigureModal(newIndex);
}
window.navigateFigure = navigateFigure;

function zoomFigure(delta) {
  const imgEl = document.getElementById("modal-figure-img");
  if (!imgEl) return;

  let newZoom = state.figures.zoomLevel + delta;
  newZoom = Math.max(0.5, Math.min(3.0, newZoom));
  state.figures.zoomLevel = newZoom;
  imgEl.style.transform = `scale(${newZoom})`;
}
window.zoomFigure = zoomFigure;

function resetFigureZoom() {
  state.figures.zoomLevel = 1.0;
  const imgEl = document.getElementById("modal-figure-img");
  if (imgEl) {
    imgEl.style.transform = `scale(1.0)`;
  }
}
window.resetFigureZoom = resetFigureZoom;


