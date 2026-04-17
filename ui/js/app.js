/**
 * SSNS-Clean Visual IDE — main orchestrator (v9 layout, range-slider config).
 *
 * State:
 *   data            full training_log.json (or null until first load)
 *   snapshotIndex   index into state.data.snapshots
 *   batchIndex      index into snapshot.samples
 *   showWeights     boolean — whether weight matrices are rendered
 *   polling         interval id during /api/run_training (or null)
 */
import {
    getTrainingData, getTrainingStatus,
    postRunTraining, postStopTraining, postManualTest, postStressTest,
} from "/ui/js/api.js";
import { sigmoidArr } from "/ui/js/extract.js";
import { MatrixHeatmap, ClusterStripHeatmap, VectorHeatmap } from "/ui/js/heatmap.js";
import { LossChart } from "/ui/js/lossChart.js";

const tooltip = document.getElementById("tooltip");

// ---------------------------------------------------------------------------
// State.
// ---------------------------------------------------------------------------

const state = {
    data:            null,
    snapshotIndex:   0,
    batchIndex:      0,
    showWeights:     false,
    polling:         null,   // /api/training_data interval id (snapshots)
    statusPolling:   null,   // /api/training_status interval id (progress)
    autoStressDone:  false,  // already auto-ran stress for this completion?
    seenRunning:     false,  // status confirmed running:true at least once?
    runEpochs:       0,      // epochs requested for the current run
    runSubmittedAt:  0,      // wall-clock ms when /api/run_training POST went out
};

// ---------------------------------------------------------------------------
// Heatmaps (3 weight matrices + 2 cluster-strip outputs).
// ---------------------------------------------------------------------------

// Top-down per column: Input X -> W1 -> Hidden H -> W2 -> Output Y.
const hmXT  = new VectorHeatmap(document.getElementById("hm-x-t"),
    { tooltip, label: "X" });
const hmW1T = new MatrixHeatmap(document.getElementById("hm-w1t"),
    { tooltip, label: "W1_T", symmetric: true });
const hmHT  = new VectorHeatmap(document.getElementById("hm-ht"),
    { tooltip, label: "H_T",  symmetric: false });
const hmW2T = new MatrixHeatmap(document.getElementById("hm-w2t"),
    { tooltip, label: "W2_T", symmetric: true });

const hmXS  = new VectorHeatmap(document.getElementById("hm-x-s"),
    { tooltip, label: "X" });
const hmW1S = new MatrixHeatmap(document.getElementById("hm-w1s"),
    { tooltip, label: "W1",   symmetric: true });
const hmHS  = new VectorHeatmap(document.getElementById("hm-hs"),
    { tooltip, label: "H",    symmetric: false });
const hmW2S = new MatrixHeatmap(document.getElementById("hm-w2s"),
    { tooltip, label: "W2",   symmetric: true });

const hmYT  = new ClusterStripHeatmap(document.getElementById("hm-yt"),
    { tooltip, label: "Y_T (Teacher)", clusterSize: 5, deadZone: 0.09 });
const hmYS  = new ClusterStripHeatmap(document.getElementById("hm-ys"),
    { tooltip, label: "Y_S (Student)", clusterSize: 5, deadZone: 0.09 });

// Cluster heatmaps under Manual Input — show the actual sigmoid output
// of each network for the user-supplied X.
const hmManualYT = new ClusterStripHeatmap(
    document.getElementById("manual-hm-yt"),
    { tooltip, label: "Manual Y_T", clusterSize: 5, deadZone: 0.09 });
const hmManualYS = new ClusterStripHeatmap(
    document.getElementById("manual-hm-ys"),
    { tooltip, label: "Manual Y_S", clusterSize: 5, deadZone: 0.09 });

const lossChart = new LossChart(document.getElementById("loss-chart"));

// ---------------------------------------------------------------------------
// DOM references.
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const els = {
    statusPill:   $("status-pill"),
    archMeta:     $("arch-meta"),
    runBtn:           $("run-btn"),
    stopBtn:          $("stop-btn"),
    presetFheFastBtn: $("preset-fhe-fast-btn"),
    presetFheBtn:     $("preset-fhe-btn"),
    presetFheRichBtn: $("preset-fhe-rich-btn"),
    runProgress:      $("run-progress"),
    runStatus:        $("run-status"),
    configForm:   $("config-form"),
    epochSlider:  $("epoch-slider"),
    epochDisplay: $("epoch-display"),
    batchSelect:  $("batch-select"),
    showWeights:  $("show-weights"),
    outputDimDisplay: $("output_dim_display"),
    dimW1T: $("dim-w1t"), dimW2T: $("dim-w2t"),
    dimW1S: $("dim-w1s"), dimW2S: $("dim-w2s"),
    dimYT:  $("dim-yt"),  dimYS:  $("dim-ys"),
    dimXT:  $("dim-x-t"), dimXS:  $("dim-x-s"),
    dimHT:  $("dim-ht"),  dimHS:  $("dim-hs"),
    lossMin: $("loss-min"), lossLast: $("loss-last"),
    manualForm:     $("manual-form"),
    manualX:        $("manual-x"),
    manualFill:     $("manual-fill"),
    manualStatus:   $("manual-status"),
    manualSummary:  $("manual-summary"),
    manualResult:   $("manual-result"),
    manualHeatmaps: $("manual-heatmaps"),
    stressForm:   $("stress-form"),
    stressN:      $("stress-n"),
    stressSeed:   $("stress-seed"),
    stressRun:    $("stress-run"),
    stressStatus: $("stress-status"),
    stressResult: $("stress-result"),
    stressHist:   $("stress-histogram"),
    stressHistCap:$("stress-hist-caption"),
};

// ---------------------------------------------------------------------------
// Status pill.
// ---------------------------------------------------------------------------

function setStatus(label, kind) {
    els.statusPill.textContent = label;
    els.statusPill.className = `pill pill-${kind}`;
}

// ---------------------------------------------------------------------------
// Boot.
// ---------------------------------------------------------------------------

async function boot() {
    bindControls();
    bindRangeOutputs();
    syncDerivedValues();

    // First check if a training is currently in progress (e.g. browser was
    // reloaded mid-run by another tab) — re-attach pollers if so. This
    // prevents the pill from showing "Ready" while training is actually live.
    let liveStatus = null;
    try {
        liveStatus = await getTrainingStatus();
    } catch (_e) { /* 404 / connection error → treat as no active run */ }

    if (liveStatus && liveStatus.running) {
        state.runEpochs      = liveStatus.total_epochs;
        state.seenRunning    = true;     // status already says running
        state.autoStressDone = true;     // do NOT auto-stress on a foreign run
        els.runBtn.disabled  = true;
        setStatus("Processing", "processing");
        els.runStatus.classList.remove("error", "ok");
        els.runStatus.textContent =
            `Re-attached to live training (epoch ${liveStatus.epoch}/${liveStatus.total_epochs}).`;
        startPolling({ epochs: liveStatus.total_epochs });
        startStatusPolling({ epochs: liveStatus.total_epochs });
    }

    try {
        const data = await getTrainingData();
        if (data) {
            state.data = data;
            initialiseUIFromData();
        } else {
            // No log AND no in-progress run -> idle state.
            if (!liveStatus || !liveStatus.running) {
                setStatus("No training data", "idle");
                els.runStatus.textContent =
                    "No training_log.json yet. Adjust parameters and click Run Training.";
            }
        }
    } catch (err) {
        setStatus("Load failed", "error");
        els.runStatus.classList.add("error");
        els.runStatus.textContent = `Error loading training data: ${err.message}`;
    }
}

// ---------------------------------------------------------------------------
// Slider <-> output sync, derived value (output_dim), input mirroring.
// ---------------------------------------------------------------------------

function bindRangeOutputs() {
    const ranges = els.configForm.querySelectorAll('input[type="range"]');
    ranges.forEach(r => {
        const out = els.configForm.querySelector(`output[for="${r.id}"]`);
        if (!out) return;
        const fmt = (val) => {
            // Float-stepped sliders need decimal preservation.
            if (r.step.includes(".")) {
                const decimals = r.step.split(".")[1].length;
                return parseFloat(val).toFixed(decimals);
            }
            return String(val);
        };
        const update = () => { out.value = fmt(r.value); };
        r.addEventListener("input", update);
        update();
    });

    // Derived: output_dim = output_clusters * cluster_size.
    const oc = els.configForm.elements.output_clusters;
    const cs = els.configForm.elements.cluster_size;
    [oc, cs].forEach(el => el.addEventListener("input", syncDerivedValues));

    // samples_to_log is logically bounded by batch_size: rows beyond batch
    // get clipped server-side anyway. Mirror that constraint live in the UI.
    const batch   = els.configForm.elements.batch_size;
    const samples = els.configForm.elements.samples_to_log;
    const samplesOut = els.configForm.querySelector(
        'output[for="samples_to_log"]');

    function clampSamplesToBatch() {
        const b = parseInt(batch.value, 10);
        samples.max = String(b);
        if (parseInt(samples.value, 10) > b) {
            samples.value = String(b);
            if (samplesOut) samplesOut.value = String(b);
        }
    }
    batch.addEventListener("input", clampSamplesToBatch);
    clampSamplesToBatch();   // apply on first paint
}

function syncDerivedValues() {
    const oc = parseInt(els.configForm.elements.output_clusters.value, 10);
    const cs = parseInt(els.configForm.elements.cluster_size.value, 10);
    const dim = oc * cs;
    els.outputDimDisplay.value = String(dim);
}

// ---------------------------------------------------------------------------
// Controls.
// ---------------------------------------------------------------------------

function bindControls() {
    els.configForm.addEventListener("submit", onRun);
    els.stopBtn.addEventListener("click", onStop);
    els.presetFheFastBtn.addEventListener("click", applyFhePresetFast);
    els.presetFheBtn.addEventListener("click", applyFhePreset);
    els.presetFheRichBtn.addEventListener("click", applyFhePresetRich);
    els.epochSlider.addEventListener("input", onEpochChange);
    els.batchSelect.addEventListener("change", onBatchChange);
    els.showWeights.addEventListener("change", onShowWeightsChange);
    els.manualForm.addEventListener("submit", onManualTest);
    els.manualFill.addEventListener("click", onFillX);
    els.stressForm.addEventListener("submit", onStressRun);
}

function readConfig() {
    // Build the JSON payload for /api/run_training.  S_input is mirrored
    // from T_input so both networks see the same X, per protocol design.
    const f = els.configForm.elements;
    const T_input = parseInt(f.T_input.value, 10);
    const cluster_size = parseInt(f.cluster_size.value, 10);
    const output_clusters = parseInt(f.output_clusters.value, 10);
    return {
        T_input,
        T_hidden:       parseInt(f.T_hidden.value, 10),
        S_input:        T_input,
        S_hidden:       parseInt(f.S_hidden.value, 10),
        output_dim:     output_clusters * cluster_size,
        cluster_size,
        batch_size:     parseInt(f.batch_size.value, 10),
        epochs:         parseInt(f.epochs.value, 10),
        dz:             parseFloat(f.dz.value),
        lr_max:         parseFloat(f.lr_max.value),
        warmup_frac:    parseFloat(f.warmup_frac.value),
        samples_to_log: parseInt(f.samples_to_log.value, 10),
        snapshot_count: parseInt(f.snapshot_count.value, 10),
        use_fhe:        !!f.use_fhe.checked,
    };
}

async function onRun(e) {
    e.preventDefault();
    const params = readConfig();

    if (params.output_dim % params.cluster_size !== 0) {
        flagError(els.runStatus, "output_dim must be divisible by cluster_size.");
        return;
    }

    els.runBtn.disabled = true;
    setStatus("Processing", "processing");
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent = "Submitting training request...";
    els.runProgress.hidden = true;
    els.runProgress.textContent = "";
    // Reset per-run guards so completion detection is not contaminated
    // by the *previous* run's status file (running:false stays on disk
    // between runs).
    state.autoStressDone  = false;
    state.seenRunning     = false;
    state.runEpochs       = params.epochs;
    state.runSubmittedAt  = Date.now();   // ms epoch — used by status tick
                                          // to spot completed_at > submit
    state.runPid          = null;

    try {
        const resp = await postRunTraining(params);
        state.runPid = resp.pid;
        els.runStatus.textContent =
            `Training subprocess started (pid=${resp.pid}). Polling for snapshots...`;
        // Reveal Stop button + progress line.
        els.stopBtn.hidden    = false;
        els.stopBtn.disabled  = false;
        els.runProgress.hidden = false;
        els.runProgress.textContent = `epoch 0/${params.epochs} — starting…`;
        startPolling(params);
        startStatusPolling(params);
    } catch (err) {
        flagError(els.runStatus, `Run failed: ${err.message}`);
        setStatus("Error", "error");
        els.runBtn.disabled = false;
    }
}

// FHE presets — three variants tuned for different wall-time budgets.
//
// Common shared-secret recipe (verified by tests/test_key_agreement_e2e.cpp):
//   T_in=4, T_h=16  — wider Teacher is free in FHE (Teacher is plaintext
//                      server-side); spreads sigmoid outputs out of dead-zone
//   S_h=16, Y=20    — modest Student capacity, cheap mul_cipher count
//   batch=4         — halves per-step FHE cost vs batch=8 with marginal
//                      yield loss
//   dz=0.10         — verified mismatch-safe at 200+ epochs
//
// Per-preset epochs:
//   fast:    ep=200  → ~5  min, ~1.79 shared/trial (sweep)
//   default: ep=400  → ~10 min, ~1.97 shared/trial (sweep)
//   rich:    ep=800  → ~20 min, ~2.07 shared/trial (sweep)
function applyFhePresetVariant(epochs, label, expectedYield) {
    const preset = {
        T_input:         4,
        T_hidden:        16,
        S_hidden:        16,
        output_clusters: 4,
        cluster_size:    5,
        batch_size:      4,
        epochs:          epochs,
        dz:              0.10,
        lr_max:          0.01,
        warmup_frac:     0.05,
        samples_to_log:  2,
        snapshot_count:  20,
    };
    for (const [name, val] of Object.entries(preset)) {
        const el = els.configForm.elements[name];
        if (!el) continue;
        el.value = String(val);
        el.dispatchEvent(new Event("input", { bubbles: true }));
    }
    const fhe = els.configForm.elements.use_fhe;
    if (fhe) {
        fhe.checked = true;
        fhe.dispatchEvent(new Event("change", { bubbles: true }));
    }
    syncDerivedValues();
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent =
        `${label} applied (T=4/16 S=16 Y=20 B=4 ep=${epochs}, ~${expectedYield} shared bits/trial, 0 mismatches). Press Run Training.`;
}

function applyFhePresetFast()    { applyFhePresetVariant(200, "FHE fast preset",  "1.79"); }
function applyFhePreset()        { applyFhePresetVariant(400, "FHE preset",        "1.97"); }
function applyFhePresetRich()    { applyFhePresetVariant(800, "FHE rich preset",  "2.07"); }

// Stop button handler — POST /api/stop_training with the stored pid.
async function onStop() {
    if (!state.runPid) return;
    els.stopBtn.disabled = true;
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent = `Stopping pid=${state.runPid}...`;
    try {
        await postStopTraining(state.runPid);
        // Don't tear down polling here — the status poll will pick up the
        // running:false marker the server just wrote and finalise the UI
        // through the existing completion path.
        els.runStatus.textContent = `Stop signal sent (pid=${state.runPid}). Waiting for confirmation...`;
    } catch (err) {
        flagError(els.runStatus, `Stop failed: ${err.message}`);
        els.stopBtn.disabled = false;
    }
}

function flagError(node, msg) {
    node.textContent = msg;
    node.classList.add("error");
    node.classList.remove("ok");
}

// ---------------------------------------------------------------------------
// Polling.
// ---------------------------------------------------------------------------

function startPolling(params) {
    if (state.polling) clearInterval(state.polling);
    // The pill state is set by the caller (onRun or boot); startPolling
    // itself is purely a snapshot-data refresher and must not flip the pill.

    let lastEpoch = -1;

    const tick = async () => {
        try {
            const data = await getTrainingData();
            if (!data) return;

            const isNewRun = !state.data
                || (state.data.metadata && data.metadata
                    && state.data.metadata.epochs !== data.metadata.epochs);
            const newCount = data.snapshots.length;
            const lastSnap = data.snapshots[newCount - 1];
            const observedEpoch = lastSnap ? lastSnap.epoch : 0;

            if (isNewRun || observedEpoch !== lastEpoch) {
                state.data = data;
                lastEpoch = observedEpoch;
                initialiseUIFromData();
                els.runStatus.textContent =
                    `Snapshots: ${newCount}. Latest epoch: ${observedEpoch}.`;
            }
            // Completion is detected by startStatusPolling (which watches
            // training_status.running), not here — the training_log.json
            // is rewritten only at the very end of the run, so it is an
            // unreliable progress signal during training.
        } catch (err) {
            console.warn("poll error", err);
        }
    };

    setTimeout(tick, 800);
    state.polling = setInterval(tick, 2500);
}

function stopAllPolling() {
    if (state.polling)       { clearInterval(state.polling);       state.polling = null; }
    if (state.statusPolling) { clearInterval(state.statusPolling); state.statusPolling = null; }
}

// ---------------------------------------------------------------------------
// Status poll — /api/training_status, 500ms cadence.
//
// Drives BOTH (a) completion detection ("Ready" pill, stops snapshot polling,
// re-enables Run button) and (b) auto-trigger of the post-sync stress test.
//
// We cannot trust the very first running:false reading because the previous
// run's status file persists between runs. We require a running:true sighting
// before treating a subsequent running:false as "this run is done".
//
// Safety nets to keep request volume bounded:
//   - HARD CAP: stop after MAX_TICKS regardless (~10 min wall time).
//   - Stop after N consecutive 404s (status file vanished mid-run).
// ---------------------------------------------------------------------------

const STATUS_POLL_MS      = 500;
const STATUS_MAX_404      = 30;            // 30 * 500ms = 15s of consecutive 404s
const STALL_TIMEOUT_MS    = 10 * 60 * 1000; // 10 min of no epoch advance = stuck
const HARD_CEILING_MS     = 4  * 60 * 60 * 1000; // 4 h hard ceiling, no matter what

function startStatusPolling(_params) {
    if (state.statusPolling) clearInterval(state.statusPolling);

    const startedAt        = Date.now();
    let consecutive404     = 0;
    let lastProgressEpoch  = -1;
    let lastProgressAt     = startedAt;

    const giveUp = (reason) => {
        stopAllPolling();
        els.runBtn.disabled = false;
        els.stopBtn.hidden = true;
        els.stopBtn.disabled = false;
        els.runProgress.hidden = true;
        state.runPid = null;
        setStatus("Stalled", "error");
        els.runStatus.classList.remove("ok");
        els.runStatus.classList.add("error");
        els.runStatus.textContent =
            `Polling stopped: ${reason}. Reload the page if you still expect a run.`;
    };

    const tick = async () => {
        const now = Date.now();
        // Hard ceiling — last-resort fuse if everything else fails.
        if (now - startedAt > HARD_CEILING_MS) {
            giveUp(`hard ceiling reached (${(HARD_CEILING_MS/3600000)|0} h)`);
            return;
        }
        // Stall detector — give up if epoch hasn't advanced in STALL_TIMEOUT_MS.
        // This adapts automatically to any wall-clock budget: a slow but
        // *progressing* training is fine; a frozen subprocess is not.
        if (now - lastProgressAt > STALL_TIMEOUT_MS) {
            giveUp(`no epoch advance for ${(STALL_TIMEOUT_MS/60000)|0} min (subprocess likely dead)`);
            return;
        }
        try {
            const s = await getTrainingStatus();
            if (!s) {
                consecutive404 += 1;
                if (consecutive404 > STATUS_MAX_404) {
                    giveUp(`status file missing for ${STATUS_MAX_404} consecutive polls`);
                }
                return;
            }
            consecutive404 = 0;
            if (s.running) state.seenRunning = true;
            // Refresh stall timer whenever we see real forward progress.
            if (typeof s.epoch === "number" && s.epoch > lastProgressEpoch) {
                lastProgressEpoch = s.epoch;
                lastProgressAt    = now;
            }

            // Live progress while training: epoch counter + loss + elapsed.
            // The eta_sec field is best-effort; only render it when present.
            if (s.running) {
                const eta = (s.eta_sec != null && Number.isFinite(s.eta_sec))
                    ? `, ETA ${s.eta_sec.toFixed(0)}s`
                    : "";
                const lossStr = (typeof s.loss === "number" && s.loss > 0)
                    ? `, loss=${s.loss.toExponential(2)}`
                    : "";
                els.runProgress.hidden = false;
                els.runProgress.textContent =
                    `epoch ${s.epoch}/${s.total_epochs}` +
                    lossStr +
                    `, elapsed ${s.elapsed_sec.toFixed(1)}s${eta}`;
            }

            // Treat completed_at > runSubmittedAt as definitive proof that
            // OUR run finished — even if we never observed running:true
            // (e.g. the tab was background-throttled through the entire
            // training window and missed the transition).
            const completedAfterSubmit = !s.running && s.completed_at &&
                state.runSubmittedAt &&
                Date.parse(s.completed_at) >= state.runSubmittedAt - 1000;

            if (!s.running && (state.seenRunning || completedAfterSubmit)) {
                // Training has finished (either naturally or via Stop).
                // Refresh snapshot data once, then wind everything down.
                stopAllPolling();
                try {
                    const data = await getTrainingData();
                    if (data) { state.data = data; initialiseUIFromData(); }
                } catch (e) { /* ignore — UI is already best-effort */ }
                els.runBtn.disabled = false;
                els.stopBtn.hidden = true;
                els.stopBtn.disabled = false;
                els.runProgress.hidden = true;
                state.runPid = null;
                setStatus("Ready", "ready");
                els.runStatus.classList.remove("error");
                els.runStatus.classList.add("ok");
                const stoppedTag = s.stopped ? " (stopped by user)" : "";
                els.runStatus.textContent =
                    `Training complete${stoppedTag}: epoch ${s.epoch}/${s.total_epochs} ` +
                    `(loss=${s.loss.toExponential(2)}, ` +
                    `elapsed=${s.elapsed_sec.toFixed(1)}s).`;
                if (!state.autoStressDone && !s.stopped) {
                    // Skip the auto-stress test on user-initiated stop —
                    // the training was cut short, the snapshot probably
                    // isn't a converged keypair yet.
                    state.autoStressDone = true;
                    autoRunStressTest();
                }
            }
        } catch (err) {
            console.warn("status poll error", err);
        }
    };
    setTimeout(tick, 250);
    state.statusPolling = setInterval(tick, STATUS_POLL_MS);
}

// ---------------------------------------------------------------------------
// Re-initialise after fresh data arrives.
// ---------------------------------------------------------------------------

function initialiseUIFromData() {
    const { metadata, snapshots } = state.data;
    if (!snapshots || snapshots.length === 0) {
        setStatus("Empty log", "idle");
        return;
    }

    // Configure cluster-strip heatmaps with the trained-network's params.
    hmYT.setParams({ clusterSize: metadata.cluster_size, deadZone: metadata.dz });
    hmYS.setParams({ clusterSize: metadata.cluster_size, deadZone: metadata.dz });

    // Epoch slider snaps to recorded snapshot indices.
    els.epochSlider.disabled = false;
    els.epochSlider.min = 0;
    els.epochSlider.max = snapshots.length - 1;
    els.epochSlider.step = 1;
    els.epochSlider.value = snapshots.length - 1;
    state.snapshotIndex = snapshots.length - 1;

    // Batch dropdown: range over the first samples_logged batch rows.
    const samplesLogged = metadata.samples_logged ||
        (snapshots[0].samples ? snapshots[0].samples.length : 1);
    populateBatchSelect(samplesLogged);
    state.batchIndex = Math.min(state.batchIndex, samplesLogged - 1);

    // Loss curve.
    const points = snapshots.map(s => ({ epoch: s.epoch, loss: s.loss }));
    lossChart.setData(points);
    if (points.length) {
        const losses = points.map(p => p.loss);
        els.lossMin.textContent  = Math.min(...losses).toExponential(3);
        els.lossLast.textContent = losses[losses.length - 1].toExponential(3);
    }

    // Architecture readout in the topbar.
    els.archMeta.textContent =
        `T(${metadata.T_input}->${metadata.T_hidden}->${metadata.output_dim}) | ` +
        `S(${metadata.S_input}->${metadata.S_hidden}->${metadata.output_dim}) | ` +
        `clusters=${metadata.output_dim / metadata.cluster_size} x ${metadata.cluster_size} | ` +
        `dz=${metadata.dz}`;

    // Only set "Ready" when no polling is active AND no run is being
    // monitored — otherwise leave the current pill (Processing) alone.
    if (!state.polling && !state.statusPolling) {
        setStatus("Ready", "ready");
    }
    render();
}

function populateBatchSelect(n) {
    els.batchSelect.innerHTML = "";
    for (let i = 0; i < n; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${i}`;
        els.batchSelect.appendChild(opt);
    }
    els.batchSelect.value = String(state.batchIndex);
}

// ---------------------------------------------------------------------------
// Event handlers.
// ---------------------------------------------------------------------------

function onEpochChange() {
    state.snapshotIndex = parseInt(els.epochSlider.value, 10);
    render();
}

function onBatchChange() {
    state.batchIndex = parseInt(els.batchSelect.value, 10);
    render();
}

function onShowWeightsChange() {
    state.showWeights = els.showWeights.checked;
    document.body.classList.toggle("show-weights", state.showWeights);
    if (state.showWeights) renderWeights();
}

// ---------------------------------------------------------------------------
// Render — single source of truth.
// ---------------------------------------------------------------------------

function render() {
    if (!state.data) return;
    const { snapshots, metadata } = state.data;
    const snap = snapshots[state.snapshotIndex];
    if (!snap) return;
    const sample = snap.samples[Math.min(state.batchIndex, snap.samples.length - 1)];

    els.epochDisplay.textContent = `${snap.epoch}`;
    const nClusters = (sample.Y_true && metadata.cluster_size)
        ? Math.floor(sample.Y_true.length / metadata.cluster_size) : 0;
    els.dimYT.textContent = `[${sample.Y_true.length} = ${nClusters} × ${metadata.cluster_size}]`;
    els.dimYS.textContent = `[${sample.Y_pred.length} = ${nClusters} × ${metadata.cluster_size}]`;
    if (els.dimXT) els.dimXT.textContent = sample.X   ? `[${sample.X.length}]`   : "[—]";
    if (els.dimXS) els.dimXS.textContent = sample.X   ? `[${sample.X.length}]`   : "[—]";
    if (els.dimHT) els.dimHT.textContent = sample.H_T ? `[${sample.H_T.length}]` : "[—]";
    if (els.dimHS) els.dimHS.textContent = sample.H_raw ? `[${sample.H_raw.length}]` : "[—]";
    const W = snap.weights || {};
    els.dimW1T.textContent = matShape(W.W1_T);
    els.dimW2T.textContent = matShape(W.W2_T);
    els.dimW1S.textContent = matShape(W.W1);
    els.dimW2S.textContent = matShape(W.W2);

    // Input X (shared) and per-network hidden activations — always rendered.
    if (sample.X) {
        hmXT.setData(sample.X);
        hmXS.setData(sample.X);
    } else {
        hmXT.clear(); hmXS.clear();
    }
    sample.H_T   ? hmHT.setData(sample.H_T)   : hmHT.clear();
    sample.H_raw ? hmHS.setData(sample.H_raw) : hmHS.clear();

    // Cluster-strip outputs (always rendered).
    hmYT.setData(sigmoidArr(sample.Y_true));
    hmYS.setData(sigmoidArr(sample.Y_pred));

    // Weight matrices — only when checkbox is on (heavy at v9 dimensions).
    if (state.showWeights) renderWeights();
}

function renderWeights() {
    if (!state.data) return;
    const snap = state.data.snapshots[state.snapshotIndex];
    if (!snap) return;
    // Older logs (pre-W2_T schema) may lack some keys — fall back to clear()
    // rather than crashing on .setData(undefined).
    const W = snap.weights || {};
    W.W1_T ? hmW1T.setData(W.W1_T) : hmW1T.clear();
    W.W2_T ? hmW2T.setData(W.W2_T) : hmW2T.clear();
    W.W1   ? hmW1S.setData(W.W1)   : hmW1S.clear();
    W.W2   ? hmW2S.setData(W.W2)   : hmW2S.clear();
}

function matShape(mat) {
    if (!mat || !mat.length) return "[—]";
    return `[${mat.length} x ${mat[0].length}]`;
}

// ---------------------------------------------------------------------------
// Manual input — POST /api/manual_test → render bit comparison strings.
// ---------------------------------------------------------------------------

function onFillX() {
    if (!state.data) {
        flagError(els.manualStatus,
            "Train a model first to know the input dimensionality.");
        return;
    }
    const dim = state.data.metadata.S_input;
    const X = new Array(dim);
    for (let i = 0; i < dim; i++) {
        const u1 = Math.max(1e-12, Math.random());
        const u2 = Math.random();
        X[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    els.manualX.value = X.map(v => v.toFixed(4)).join(", ");
    els.manualStatus.classList.remove("error", "ok");
    els.manualStatus.textContent = `Filled X with ${dim} samples from N(0, 1).`;
}

async function onManualTest(e) {
    e.preventDefault();
    if (!state.data) {
        flagError(els.manualStatus, "Train a model first.");
        return;
    }
    const text = els.manualX.value.trim();
    if (!text) {
        flagError(els.manualStatus, "X is empty.");
        return;
    }
    let X;
    try {
        X = parseVector(text);
    } catch (err) {
        flagError(els.manualStatus, `Cannot parse X: ${err.message}`);
        return;
    }

    els.manualStatus.classList.remove("error", "ok");
    els.manualStatus.textContent = "Running...";
    els.manualSummary.classList.remove("visible", "match", "mismatch");
    els.manualSummary.textContent = "";
    try {
        const resp = await postManualTest(X);
        els.manualStatus.textContent = "Done.";
        els.manualStatus.classList.add("ok");
        renderManualResult(resp);
    } catch (err) {
        flagError(els.manualStatus, `Manual test failed: ${err.message}`);
        els.manualResult.innerHTML = "";
        els.manualHeatmaps.hidden = true;
        els.manualSummary.classList.remove("visible", "match", "mismatch");
        els.manualSummary.textContent = "";
    }
}

function parseVector(text) {
    const parts = text.split(/[\s,;]+/).filter(Boolean);
    const out = new Array(parts.length);
    for (let i = 0; i < parts.length; i++) {
        const v = parseFloat(parts[i]);
        if (!Number.isFinite(v)) throw new Error(`token "${parts[i]}" is not a number`);
        out[i] = v;
    }
    return out;
}

/**
 * Render Teacher and Student bit strings on two aligned rows.  Each cluster
 * occupies one column: '1' or '0' if the side is confident there, '·' if
 * inside the dead zone.  Mismatches are coloured red.
 */
function renderManualResult(resp) {
    const setT = new Set(resp.idx_T);
    const setS = new Set(resp.idx_S);
    const mapT = new Map(resp.idx_T.map((c, i) => [c, resp.bits_T[i]]));
    const mapS = new Map(resp.idx_S.map((c, i) => [c, resp.bits_S[i]]));

    const nClusters = Math.max(
        ...resp.idx_T, ...resp.idx_S, -1,
    ) + 1;
    const total = Math.max(nClusters, resp.shared_indices.length);

    // Summary lives next to manual-status in the form column.
    const summaryClass = resp.match ? "match" : "mismatch";
    const summaryLabel = resp.match
        ? "ALL SHARED BITS AGREE"
        : `MISMATCHES: ${resp.mismatches}`;
    els.manualSummary.classList.remove("match", "mismatch");
    els.manualSummary.classList.add("visible", summaryClass);
    els.manualSummary.innerHTML =
        `<div class="label">${summaryLabel}</div>` +
        `<div>shared=${resp.shared_indices.length} · ` +
        `T_conf=${resp.n_confident_T} · ` +
        `S_conf=${resp.n_confident_S} · ` +
        `miss=${resp.mismatches}</div>`;

    // Bit strings — full-width row below heatmaps.
    const rowT = [];
    const rowS = [];
    for (let c = 0; c < total; c++) {
        const inT = setT.has(c), inS = setS.has(c);
        const bT = inT ? mapT.get(c) : null;
        const bS = inS ? mapS.get(c) : null;
        const mismatch = (inT && inS && bT !== bS);
        rowT.push(span(bT, mismatch));
        rowS.push(span(bS, mismatch));
    }
    els.manualResult.innerHTML = `
        <div class="bit-strings">
            <div><span class="row-label">T</span>${rowT.join("")}</div>
            <div><span class="row-label">S</span>${rowS.join("")}</div>
        </div>`;

    // Render the two cluster heatmaps with the trained network's
    // cluster_size / dead-zone so the dead-zone shading is accurate.
    const meta = state.data && state.data.metadata;
    if (meta) {
        hmManualYT.setParams({
            clusterSize: meta.cluster_size, deadZone: meta.dz,
        });
        hmManualYS.setParams({
            clusterSize: meta.cluster_size, deadZone: meta.dz,
        });
    }
    hmManualYT.setData(resp.Y_T);
    hmManualYS.setData(resp.Y_S);
    els.manualHeatmaps.hidden = false;
}

function span(bit, mismatch) {
    if (bit === null) return `<span class="gap">·</span>`;
    const cls = mismatch ? "miss" : (bit ? "b1" : "b0");
    return `<span class="${cls}">${bit}</span>`;
}

// ---------------------------------------------------------------------------
// Stress test — POST /api/stress_test, render aggregated stats + histogram.
// ---------------------------------------------------------------------------

function autoRunStressTest() {
    // Trigger as if the user clicked the button.
    const evt = new Event("submit", { cancelable: true });
    els.stressForm.dispatchEvent(evt);
}

async function onStressRun(e) {
    e.preventDefault();
    if (!state.data) {
        flagError(els.stressStatus, "Train a model first.");
        return;
    }
    const n = parseInt(els.stressN.value, 10);
    const seedRaw = els.stressSeed.value.trim();
    const seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    if (!Number.isFinite(n) || n < 1 || n > 10000) {
        flagError(els.stressStatus, "n_trials must be in [1, 10000].");
        return;
    }
    els.stressRun.disabled = true;
    els.stressStatus.classList.remove("error", "ok");
    els.stressStatus.textContent = `Running ${n} trials...`;
    const t0 = performance.now();
    try {
        const resp = await postStressTest(n, seed);
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        els.stressStatus.classList.add("ok");
        els.stressStatus.textContent =
            `Done in ${dt}s. ` +
            `${resp.perfect_trials}/${resp.n_trials} trials matched ` +
            `(${resp.match_rate_pct.toFixed(2)}%), ` +
            `${resp.total_mismatches} total mismatches across ` +
            `${resp.total_shared_bits} shared bits.`;
        renderStressResult(resp);
    } catch (err) {
        flagError(els.stressStatus, `Stress test failed: ${err.message}`);
        els.stressResult.innerHTML = "";
        els.stressHist.hidden = true;
        els.stressHistCap.hidden = true;
    } finally {
        els.stressRun.disabled = false;
    }
}

function renderStressResult(r) {
    const fmt2 = (x) => Number.isFinite(x) ? x.toFixed(2) : "—";
    const total = r.total_clusters;
    const summary = `
        <div class="stress-summary-grid">
            <div class="metric ${r.match_rate_pct >= 99.999 ? 'match' : 'amber'}">
                <span class="metric-label">Match Rate</span>
                <span class="metric-value mono">${r.match_rate_pct.toFixed(2)}%</span>
            </div>
            <div class="metric ${r.total_mismatches === 0 ? 'match' : 'amber'}">
                <span class="metric-label">Mismatches (sum)</span>
                <span class="metric-value mono">${r.total_mismatches}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Shared Bits (sum)</span>
                <span class="metric-value mono">${r.total_shared_bits}</span>
            </div>
        </div>
    `;
    const tableHTML = `
        <table>
            <thead>
                <tr>
                    <th class="row-name">metric</th>
                    <th>mean</th>
                    <th>std</th>
                    <th>min</th>
                    <th>max</th>
                    <th>median</th>
                    <th>of total</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="row-name">teacher_confident</td>
                    <td>${fmt2(r.teacher_confident.mean)}</td>
                    <td>${fmt2(r.teacher_confident.std)}</td>
                    <td>${r.teacher_confident.min}</td>
                    <td>${r.teacher_confident.max}</td>
                    <td>${fmt2(r.teacher_confident.median)}</td>
                    <td>${fmt2(100 * r.teacher_confident.mean / total)}%</td>
                </tr>
                <tr>
                    <td class="row-name">student_confident</td>
                    <td>${fmt2(r.student_confident.mean)}</td>
                    <td>${fmt2(r.student_confident.std)}</td>
                    <td>${r.student_confident.min}</td>
                    <td>${r.student_confident.max}</td>
                    <td>${fmt2(r.student_confident.median)}</td>
                    <td>${fmt2(100 * r.student_confident.mean / total)}%</td>
                </tr>
                <tr>
                    <td class="row-name">shared</td>
                    <td>${fmt2(r.shared.mean)}</td>
                    <td>${fmt2(r.shared.std)}</td>
                    <td>${r.shared.min}</td>
                    <td>${r.shared.max}</td>
                    <td>${fmt2(r.shared.median)}</td>
                    <td>${fmt2(100 * r.shared.mean / total)}%</td>
                </tr>
                <tr>
                    <td class="row-name">mismatches</td>
                    <td>${fmt2(r.mismatches.mean)}</td>
                    <td>${fmt2(r.mismatches.std)}</td>
                    <td>${r.mismatches.min}</td>
                    <td>${r.mismatches.max}</td>
                    <td>${fmt2(r.mismatches.median)}</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
        <p class="subtle stress-help">
            Per-trial counts over ${r.n_trials} N(0,1) inputs, evaluated at
            epoch ${r.epoch_used}. "of total" shows what fraction of the
            ${total} clusters survived dead-zone (dz=${r.dead_zone}) filtering on average.
        </p>`;
    els.stressResult.innerHTML = summary + tableHTML;
    drawHistogram(r.histogram_shared, r.shared.mean, total);
    els.stressHist.hidden = false;
    els.stressHistCap.hidden = false;
}

function drawHistogram(hist, mean, total) {
    const canvas = els.stressHist;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 36, padR = 12, padT = 12, padB = 26;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const counts = hist.counts;
    const bins = hist.bins;
    const nBins = counts.length;
    const maxCount = Math.max(...counts, 1);

    // Background grid (4 horizontal lines).
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * i / 4;
        ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
    }
    ctx.stroke();

    // Bars.
    const barW = plotW / nBins;
    ctx.fillStyle = "#1f6feb";
    for (let i = 0; i < nBins; i++) {
        const h = (counts[i] / maxCount) * plotH;
        const x = padL + i * barW;
        const y = padT + plotH - h;
        ctx.fillRect(x + 1, y, Math.max(1, barW - 2), h);
    }

    // Mean line.
    if (Number.isFinite(mean)) {
        const lo = bins[0], hi = bins[bins.length - 1];
        const x = padL + plotW * (mean - lo) / Math.max(1e-9, hi - lo);
        ctx.strokeStyle = "#d29922";
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
        ctx.setLineDash([]);
    }

    // Axes.
    ctx.strokeStyle = "#30363d";
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
    ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
    ctx.stroke();

    // Labels.
    ctx.fillStyle = "#9da7b3";
    ctx.font = "10px monospace";
    ctx.fillText(String(maxCount), 4, padT + 8);
    ctx.fillText("0", 4, H - padB + 4);
    ctx.fillText(`0`, padL, H - 6);
    ctx.fillText(`${total}`, W - padR - 18, H - 6);
    ctx.fillStyle = "#d29922";
    ctx.fillText(`mean ${mean.toFixed(1)}`, W - padR - 96, padT + 12);
}

// ---------------------------------------------------------------------------
// Go.
// ---------------------------------------------------------------------------

boot();
