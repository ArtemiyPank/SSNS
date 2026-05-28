// ssns visual ide main orchestrator v9 layout range slider config
//
// state
//   data            full training_log.json null until first load
//   snapshotIndex   index into state.data.snapshots
//   batchIndex      index into snapshot.samples
//   showWeights     bool render weights or not
//   polling         interval id during run_training or null
import {
    getTrainingData, getTrainingStatus,
    postRunTraining, postStopTraining, postManualTest, postStressTest,
} from "/ui/js/api.js";
import { sigmoidArr } from "/ui/js/extract.js";
import { MatrixHeatmap, ClusterStripHeatmap, VectorHeatmap } from "/ui/js/heatmap.js";
import { LossChart } from "/ui/js/lossChart.js";

const tooltip = document.getElementById("tooltip");

// state

const state = {
    data:            null,
    snapshotIndex:   0,
    batchIndex:      0,
    showWeights:     false,
    polling:         null,   // training_data interval id snapshots
    statusPolling:   null,   // training_status interval id progress
    autoStressDone:  false,  // auto stress already ran
    seenRunning:     false,  // status reported running:true at least once
    runSubmittedAt:  0,      // wall clock ms of run_training POST
};

// heatmaps 3 weight matrices plus 2 cluster strip outputs

// top down per col X W1 H W2 Y
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

// cluster heatmaps under manual input sigmoid output for user X
const hmManualYT = new ClusterStripHeatmap(
    document.getElementById("manual-hm-yt"),
    { tooltip, label: "Manual Y_T", clusterSize: 5, deadZone: 0.09 });
const hmManualYS = new ClusterStripHeatmap(
    document.getElementById("manual-hm-ys"),
    { tooltip, label: "Manual Y_S", clusterSize: 5, deadZone: 0.09 });

const lossChart = new LossChart(document.getElementById("loss-chart"));

// dom refs

const $ = (id) => document.getElementById(id);
const els = {
    statusPill:   $("status-pill"),
    archMeta:     $("arch-meta"),
    runBtn:           $("run-btn"),
    stopBtn:          $("stop-btn"),
    presetFheConservativeBtn: $("preset-fhe-conservative-btn"),
    presetFheBalancedBtn:     $("preset-fhe-balanced-btn"),
    randomizeSeedsBtn:        $("randomize-seeds-btn"),
    teacherSeedInput:         $("teacher_seed"),
    bfaSeedInput:             $("bfa_seed"),
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

// status pill

function setStatus(label, kind) {
    els.statusPill.textContent = label;
    els.statusPill.className = `pill pill-${kind}`;
}

// boot

async function boot() {
    bindControls();
    bindRangeOutputs();
    syncDerivedValues();

    // check if run is in progress like reload mid run
    // if so reattach pollers so pill is correct
    let liveStatus = null;
    try {
        liveStatus = await getTrainingStatus();
    } catch (_e) { /* 404 or conn err treat as no active run */ }

    if (liveStatus && liveStatus.running) {
        state.seenRunning    = true;     // status already says running
        state.autoStressDone = true;     // do not auto stress a foreign run
        els.runBtn.disabled  = true;
        setStatus("Processing", "processing");
        els.runStatus.classList.remove("error", "ok");
        els.runStatus.textContent =
            `Re-attached to live training (epoch ${liveStatus.epoch}/${liveStatus.total_epochs}).`;
        startPolling();
        startStatusPolling({ epochs: liveStatus.total_epochs });
    }

    try {
        const data = await getTrainingData();
        if (data) {
            state.data = data;
            initialiseUIFromData();
        } else {
            // no log no run go idle
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

// slider output sync derived value output_dim input mirror

function bindRangeOutputs() {
    const ranges = els.configForm.querySelectorAll('input[type="range"]');
    ranges.forEach(r => {
        const fmt = (val) => {
            // float stepped sliders need decimals
            if (r.step.includes(".")) {
                const decimals = r.step.split(".")[1].length;
                return parseFloat(val).toFixed(decimals);
            }
            return String(val);
        };

        // two way sync slider <-> num input id = `${range_id}_num`
        const num = document.getElementById(`${r.id}_num`);
        if (num) {
            // skip slider to num write back while user is typing in num
            // otherwise typing 1 lands on slider as 2 due to clamp and overwrites the num field
            let numEditing = false;

            r.addEventListener("input", () => {
                if (numEditing) return;
                num.value = fmt(r.value);
            });
            // num to slider drives slider visually fires input so other listeners run like syncDerivedValues
            // do NOT clamp num.value here clamping runs on blur so user can finish typing
            num.addEventListener("input", () => {
                numEditing = true;
                const v = parseFloat(num.value);
                if (Number.isFinite(v)) {
                    r.value = String(v);
                    r.dispatchEvent(new Event("input", { bubbles: true }));
                }
                numEditing = false;
            });
            // on blur restore valid value if empty or NaN
            // never silently overwrite a parseable number even if outside slider range
            num.addEventListener("blur", () => {
                const v = parseFloat(num.value);
                if (!Number.isFinite(v)) num.value = fmt(r.value);
            });
            // initial sync
            num.value = fmt(r.value);
        }

        // legacy any output for=ID also gets the value
        const out = els.configForm.querySelector(`output[for="${r.id}"]`);
        if (out) {
            const update = () => { out.value = fmt(r.value); };
            r.addEventListener("input", update);
            update();
        }
    });

    // derived output_dim = output_clusters * cluster_size
    const oc = els.configForm.elements.output_clusters;
    const cs = els.configForm.elements.cluster_size;
    [oc, cs].forEach(el => el.addEventListener("input", syncDerivedValues));

    // samples_to_log bounded by batch_size rows beyond batch get clipped server side anyway
    // mirror that constraint but only AFTER user finishes editing
    // never clamp while still typing
    const batch      = els.configForm.elements.batch_size;
    const batchNum   = document.getElementById("batch_size_num");
    const samples    = els.configForm.elements.samples_to_log;
    const samplesNum = document.getElementById("samples_to_log_num");
    const samplesOut = els.configForm.querySelector(
        'output[for="samples_to_log"]');

    function clampSamplesToBatch() {
        const b = parseInt(paramVal("batch_size"), 10);
        if (!Number.isFinite(b) || b < 1) return;
        samples.max = String(b);
        if (samplesNum) samplesNum.max = String(b);
        const currentSamples = parseInt(paramVal("samples_to_log"), 10);
        if (Number.isFinite(currentSamples) && currentSamples > b) {
            samples.value = String(b);
            if (samplesNum) samplesNum.value = String(b);
            if (samplesOut) samplesOut.value = String(b);
        }
    }
    // slider drag commits on change number commits on blur thats the natural done typing signal
    batch.addEventListener("change", clampSamplesToBatch);
    if (batchNum) batchNum.addEventListener("blur", clampSamplesToBatch);
    clampSamplesToBatch();   // first paint
}

function syncDerivedValues() {
    const oc = parseInt(els.configForm.elements.output_clusters.value, 10);
    const cs = parseInt(els.configForm.elements.cluster_size.value, 10);
    const dim = oc * cs;
    els.outputDimDisplay.value = String(dim);
}

// controls

function bindControls() {
    els.configForm.addEventListener("submit", onRun);
    els.stopBtn.addEventListener("click", onStop);
    els.presetFheConservativeBtn.addEventListener("click", applyFhePresetConservative);
    els.presetFheBalancedBtn.addEventListener("click", applyFhePresetBalanced);
    els.randomizeSeedsBtn.addEventListener("click", onRandomizeSeeds);
    els.epochSlider.addEventListener("input", onEpochChange);
    els.batchSelect.addEventListener("change", onBatchChange);
    els.showWeights.addEventListener("change", onShowWeightsChange);
    els.manualForm.addEventListener("submit", onManualTest);
    els.manualFill.addEventListener("click", onFillX);
    els.stressForm.addEventListener("submit", onStressRun);

    // 3 way encryption radio plaintext sim real ckks
    // mirrors to two hidden fields server reads use_fhe and simulate_fhe_noise
    const radios = els.configForm.querySelectorAll('input[name="encryption_mode"]');
    radios.forEach(r => r.addEventListener("change", onEncryptionModeChange));
    // init hidden mirrors from default checked radio
    onEncryptionModeChange();

    // live topbar meta update reflects current form values
    els.configForm.addEventListener("input",  updateTopbarMetaFromForm);
    els.configForm.addEventListener("change", updateTopbarMetaFromForm);
    updateTopbarMetaFromForm();
}

// render archMeta as compact line of live form values
//   T(in->h->OD) S(in->h) 16x5 b=4 ep=80 dz=0.08 a=0.7 FHE
function updateTopbarMetaFromForm() {
    const tin  = paramVal("T_input");
    const th   = paramVal("T_hidden");
    const sin  = tin;                                  // mirrored
    const sh   = paramVal("S_hidden");
    const oc   = parseInt(paramVal("output_clusters"), 10) || 0;
    const cs   = parseInt(paramVal("cluster_size"),    10) || 0;
    const od   = oc * cs;
    const b    = paramVal("batch_size");
    const ep   = paramVal("epochs");
    const dz   = paramVal("dz");
    const a    = paramVal("bimodality_alpha");
    const checked = els.configForm.querySelector('input[name="encryption_mode"]:checked');
    const mode = checked ? checked.value : "real_fhe";
    const conf = !!els.configForm.elements.key_confirmation?.checked;

    const sep = '<span class="meta-sep">·</span>';
    const modeTxt = ({
        real_fhe:  '<span class="meta-val fhe-on">FHE</span>',
        simulated: '<span class="meta-val fhe-on">FHE-sim</span>',
        plaintext: '<span class="meta-val fhe-off">plaintext</span>',
    })[mode] || mode;
    const confTxt = conf
        ? '<span class="meta-val">+hash</span>'
        : '<span class="meta-val fhe-off">no-hash</span>';

    els.archMeta.innerHTML =
        `T(${tin}→${th}→${od})${sep}` +
        `S(${sin}→${sh}→${od})${sep}` +
        `<span class="meta-key">clusters</span> ${oc}×${cs}${sep}` +
        `<span class="meta-key">b</span>=${b} <span class="meta-key">ep</span>=${ep}${sep}` +
        `<span class="meta-key">dz</span>=${dz} <span class="meta-key">α</span>=${a}${sep}` +
        `${modeTxt}${sep}${confTxt}`;
}

function onEncryptionModeChange() {
    const checked = els.configForm.querySelector('input[name="encryption_mode"]:checked');
    const mode = checked ? checked.value : "real_fhe";
    const useFhe = els.configForm.elements.use_fhe;
    const noise  = els.configForm.elements.simulate_fhe_noise;
    if (mode === "real_fhe") {
        useFhe.value = "true";
        noise.value  = "0";
    } else if (mode === "simulated") {
        useFhe.value = "false";
        noise.value  = "0.001";  // calibrated to real ckks noise
    } else {  // plaintext
        useFhe.value = "false";
        noise.value  = "0";
    }
}

function setEncryptionMode(mode) {
    const radio = els.configForm.querySelector(`input[name="encryption_mode"][value="${mode}"]`);
    if (radio) {
        radio.checked = true;
        radio.dispatchEvent(new Event("change", { bubbles: true }));
    }
}

// prefer typed num input if present can hold values outside slider range like epochs > 30000
// else fall back to slider
function paramVal(id) {
    const num = document.getElementById(`${id}_num`);
    if (num && num.value !== "") return num.value;
    return els.configForm.elements[id].value;
}

function readConfig() {
    // build json payload for run_training
    // S_input mirrors T_input so both nets see same X
    const T_input         = parseInt  (paramVal("T_input"),         10);
    const cluster_size    = parseInt  (paramVal("cluster_size"),    10);
    const output_clusters = parseInt  (paramVal("output_clusters"), 10);
    return {
        T_input,
        T_hidden:       parseInt  (paramVal("T_hidden"),       10),
        S_input:        T_input,
        S_hidden:       parseInt  (paramVal("S_hidden"),       10),
        output_dim:     output_clusters * cluster_size,
        cluster_size,
        batch_size:     parseInt  (paramVal("batch_size"),     10),
        epochs:         parseInt  (paramVal("epochs"),         10),
        dz:             parseFloat(paramVal("dz")),
        lr_max:         parseFloat(paramVal("lr_max")),
        warmup_frac:    parseFloat(paramVal("warmup_frac")),
        samples_to_log: parseInt  (paramVal("samples_to_log"), 10),
        snapshot_count: parseInt  (paramVal("snapshot_count"), 10),
        use_fhe:             els.configForm.elements.use_fhe.value === "true",
        bimodality_alpha:    parseFloat(paramVal("bimodality_alpha")) || 0,
        simulate_fhe_noise:  parseFloat(els.configForm.elements.simulate_fhe_noise.value) || 0,
        key_confirmation:    !!els.configForm.elements.key_confirmation?.checked,
        teacher_seed:        readSeed("teacher_seed", 42),
        bfa_seed:            readSeed("bfa_seed",     43),
    };
}

// parse seed input as non-negative integer, default to fallback if blank or invalid
// js Number is 53-bit safe so accept values up to 2^53-1 plenty for uint64 seeds in practice
function readSeed(id, fallback) {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const raw = (el.value ?? "").trim();
    if (raw === "") return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) return fallback;
    return Math.floor(n);
}

// pick fresh random pair pair of seeds and write them into the form fields
// keep values within 2^53-1 so json roundtrip stays exact
function onRandomizeSeeds() {
    const MAX = Number.MAX_SAFE_INTEGER;
    const pick = () => Math.floor(Math.random() * MAX);
    if (els.teacherSeedInput) els.teacherSeedInput.value = String(pick());
    if (els.bfaSeedInput)     els.bfaSeedInput.value     = String(pick());
}

async function onRun(e) {
    e.preventDefault();
    const params = readConfig();

    if (params.output_dim % params.cluster_size !== 0) {
        flagError(els.runStatus, "output_dim must be divisible by cluster_size");
        return;
    }

    els.runBtn.disabled = true;
    setStatus("Processing", "processing");
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent = "Submitting training request...";
    els.runProgress.hidden = true;
    els.runProgress.textContent = "";
    // reset per run guards so completion detection is clean of last runs status
    state.autoStressDone  = false;
    state.seenRunning     = false;
    state.runSubmittedAt  = Date.now();   // ms epoch used by status tick
    state.runPid          = null;

    try {
        const resp = await postRunTraining(params);
        state.runPid = resp.pid;
        els.runStatus.textContent =
            `Training subprocess started (pid=${resp.pid}). Polling for snapshots...`;
        // show stop btn and progress line
        els.stopBtn.hidden    = false;
        els.stopBtn.disabled  = false;
        els.runProgress.hidden = false;
        els.runProgress.textContent = `epoch 0/${params.epochs} - starting…`;
        startPolling();
        startStatusPolling(params);
    } catch (err) {
        flagError(els.runStatus, `Run failed: ${err.message}`);
        setStatus("Error", "error");
        els.runBtn.disabled = false;
    }
}

// fhe conservative preset truly converged ~2 bits per trial
//
// recipe validated 2026-05-05 on 1500 random seeds 50K stress trials each
//   T_h=16 S_h=48 OD=30 cs=5 b=4 ep=230 dz=0.08 alpha=0.3
//   final_loss ~0.11 converged ~18.8 min fhe
//
// measured 100% TRUE FULL on 1500 random seeds mm_rate=0
//          bits >= 2 on 100% of sessions
//          best when reliability matters more than yield
function applyFhePresetConservative() {
    const preset = {
        T_input:           4,
        T_hidden:          16,
        S_hidden:          48,
        output_clusters:   6,
        cluster_size:      5,
        batch_size:        4,
        epochs:            230,
        dz:                0.08,
        lr_max:            0.01,
        warmup_frac:       0.05,
        bimodality_alpha:  0.3,
        simulate_fhe_noise: 0,
        samples_to_log:    2,
        snapshot_count:    20,
    };
    for (const [name, val] of Object.entries(preset)) {
        const el = els.configForm.elements[name];
        if (!el) continue;
        el.value = String(val);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        // mirror to num input so values outside slider range still apply like epochs=230
        const num = document.getElementById(`${name}_num`);
        if (num) {
            num.value = String(val);
            num.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }
    setEncryptionMode("real_fhe");
    const conf = els.configForm.elements.key_confirmation;
    if (conf) {
        conf.checked = true;
        conf.dispatchEvent(new Event("change", { bubbles: true }));
    }
    syncDerivedValues();
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent =
        "FHE conservative preset applied (T=4/16 S=48 OD=30 cs=5 B=4 ep=230 dz=0.08 alpha=0.3, ~18.8 min FHE, ~2.34 shared bits/trial, truly converged loss=0.11, 100% TRUE FULL on 1500 random seeds). Press Run Training.";
}

// fhe balanced preset bimodality driven more bits per session
//
// recipe validated 2026-05-06 on 4 real fhe seeds 1M+ stress trials each
//   T_h=16 S_h=48 OD=80 cs=5 b=4 ep=80 dz=0.08 alpha=0.7
//   final_loss ~1.09 not converged in usual sense ~17.3 min fhe
//
// real fhe measured seeds 1 7 100 538
//   mean_bits per trial 5.39 avg target >= 4
//   success_first_try 72.85% avg range 63 to 79%
//   silent_fail 0 of 200000 sessions key_confirmation REQUIRED
//   after 10 X_key resamples >= 99.99979% success
//
// trade off bits per session vs convergence quality
// relies on bimodality plus cs=5 voting plus sha256 hash confirm
function applyFhePresetBalanced() {
    const preset = {
        T_input:           4,
        T_hidden:          16,
        S_hidden:          48,
        output_clusters:   16,
        cluster_size:      5,
        batch_size:        4,
        epochs:            80,
        dz:                0.08,
        lr_max:            0.01,
        warmup_frac:       0.05,
        bimodality_alpha:  0.7,
        simulate_fhe_noise: 0,
        samples_to_log:    2,
        snapshot_count:    20,
    };
    for (const [name, val] of Object.entries(preset)) {
        const el = els.configForm.elements[name];
        if (!el) continue;
        el.value = String(val);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        const num = document.getElementById(`${name}_num`);
        if (num) {
            num.value = String(val);
            num.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }
    setEncryptionMode("real_fhe");
    const conf = els.configForm.elements.key_confirmation;
    if (conf) {
        conf.checked = true;
        conf.dispatchEvent(new Event("change", { bubbles: true }));
    }
    syncDerivedValues();
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent =
        "FHE balanced preset applied (T=4/16 S=48 OD=80 cs=5 B=4 ep=80 dz=0.08 alpha=0.7, ~17.3 min FHE, mean 5.39 bits/trial across 4 random seeds; 73% first-try + 99.99979% after 10 retries via key confirmation; loss ~1.09 - bimodality-driven, NOT traditionally converged). Press Run Training.";
}

// stop btn POST stop_training with stored pid
async function onStop() {
    if (!state.runPid) return;
    els.stopBtn.disabled = true;
    els.runStatus.classList.remove("error", "ok");
    els.runStatus.textContent = `Stopping pid=${state.runPid}...`;
    try {
        await postStopTraining(state.runPid);
        // do not tear down polling here status poll picks up running:false marker and finalises ui
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

// polling

function startPolling() {
    if (state.polling) clearInterval(state.polling);
    // pill state set by caller onRun or boot
    // startPolling just refreshes snapshot data does not flip pill

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
            // completion detection lives in startStatusPolling not here
            // training_log.json only rewritten at very end so not reliable progress signal
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

// status poll training_status at 500ms
//
// drives both completion detection ready pill stops snapshot polling re enables run btn
// and auto trigger of post sync stress test
//
// cant trust first running:false reading because last runs status file persists between runs
// require running:true sighting first then later running:false means this run is done
//
// safety nets to bound request volume
//   HARD CAP stop after HARD_CEILING_MS no matter what
//   stop after N consecutive 404s status file vanished mid run

const STATUS_POLL_MS      = 500;
const STATUS_MAX_404      = 30;            // 30 * 500ms = 15s of 404s
const STALL_TIMEOUT_MS    = 10 * 60 * 1000; // 10 min no epoch advance = stuck
const HARD_CEILING_MS     = 4  * 60 * 60 * 1000; // 4h hard ceiling

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
        // hard ceiling last resort fuse
        if (now - startedAt > HARD_CEILING_MS) {
            giveUp(`hard ceiling reached (${(HARD_CEILING_MS/3600000)|0} h)`);
            return;
        }
        // stall detector give up if epoch hasnt advanced
        // adapts to any wall clock budget slow but progressing is fine frozen subprocess is not
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
            // reset stall timer on forward progress
            if (typeof s.epoch === "number" && s.epoch > lastProgressEpoch) {
                lastProgressEpoch = s.epoch;
                lastProgressAt    = now;
            }

            // live progress epoch loss elapsed
            // eta_sec is best effort render only when present
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

            // treat completed_at > runSubmittedAt as proof OUR run finished
            // tab might be background throttled and miss the running:true window
            const completedAfterSubmit = !s.running && s.completed_at &&
                state.runSubmittedAt &&
                Date.parse(s.completed_at) >= state.runSubmittedAt - 1000;

            if (!s.running && (state.seenRunning || completedAfterSubmit)) {
                // training finished naturally or via stop
                // refresh snapshots once then wind down
                stopAllPolling();
                try {
                    const data = await getTrainingData();
                    if (data) { state.data = data; initialiseUIFromData(); }
                } catch (e) { /* ignore ui best effort */ }
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
                    // skip auto stress on user stop training was cut short
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

// init after fresh data arrives

function initialiseUIFromData() {
    const { metadata, snapshots } = state.data;
    if (!snapshots || snapshots.length === 0) {
        setStatus("Empty log", "idle");
        return;
    }

    // configure cluster strip heatmaps with trained params
    hmYT.setParams({ clusterSize: metadata.cluster_size, deadZone: metadata.dz });
    hmYS.setParams({ clusterSize: metadata.cluster_size, deadZone: metadata.dz });

    // epoch slider snaps to recorded snapshot indices
    els.epochSlider.disabled = false;
    els.epochSlider.min = 0;
    els.epochSlider.max = snapshots.length - 1;
    els.epochSlider.step = 1;
    els.epochSlider.value = snapshots.length - 1;
    state.snapshotIndex = snapshots.length - 1;

    // batch dropdown range over first samples_logged batch rows
    const samplesLogged = metadata.samples_logged ||
        (snapshots[0].samples ? snapshots[0].samples.length : 1);
    populateBatchSelect(samplesLogged);
    state.batchIndex = Math.min(state.batchIndex, samplesLogged - 1);

    // loss curve
    const points = snapshots.map(s => ({ epoch: s.epoch, loss: s.loss }));
    lossChart.setData(points);
    if (points.length) {
        const losses = points.map(p => p.loss);
        els.lossMin.textContent  = Math.min(...losses).toExponential(3);
        els.lossLast.textContent = losses[losses.length - 1].toExponential(3);
    }

    // refresh topbar after loading metadata so it matches viewed log
    // updateTopbarMetaFromForm picks up new values via input events fired by setting value
    updateTopbarMetaFromForm();

    // only set Ready when no polling and no run being monitored else leave current pill alone
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

// event handlers

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

// render single source of truth

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
    if (els.dimXT) els.dimXT.textContent = sample.X   ? `[${sample.X.length}]`   : "[-]";
    if (els.dimXS) els.dimXS.textContent = sample.X   ? `[${sample.X.length}]`   : "[-]";
    if (els.dimHT) els.dimHT.textContent = sample.H_T ? `[${sample.H_T.length}]` : "[-]";
    if (els.dimHS) els.dimHS.textContent = sample.H_raw ? `[${sample.H_raw.length}]` : "[-]";
    const W = snap.weights || {};
    els.dimW1T.textContent = matShape(W.W1_T);
    els.dimW2T.textContent = matShape(W.W2_T);
    els.dimW1S.textContent = matShape(W.W1);
    els.dimW2S.textContent = matShape(W.W2);

    // input X shared and per net hidden activations always rendered
    if (sample.X) {
        hmXT.setData(sample.X);
        hmXS.setData(sample.X);
    } else {
        hmXT.clear(); hmXS.clear();
    }
    sample.H_T   ? hmHT.setData(sample.H_T)   : hmHT.clear();
    sample.H_raw ? hmHS.setData(sample.H_raw) : hmHS.clear();

    // cluster strip outputs always rendered
    hmYT.setData(sigmoidArr(sample.Y_true));
    hmYS.setData(sigmoidArr(sample.Y_pred));

    // weight matrices only when checkbox is on heavy at v9 dims
    if (state.showWeights) renderWeights();
}

function renderWeights() {
    if (!state.data) return;
    const snap = state.data.snapshots[state.snapshotIndex];
    if (!snap) return;
    // older logs may lack some keys fall back to clear instead of crashing on setData(undefined)
    const W = snap.weights || {};
    W.W1_T ? hmW1T.setData(W.W1_T) : hmW1T.clear();
    W.W2_T ? hmW2T.setData(W.W2_T) : hmW2T.clear();
    W.W1   ? hmW1S.setData(W.W1)   : hmW1S.clear();
    W.W2   ? hmW2S.setData(W.W2)   : hmW2S.clear();
}

function matShape(mat) {
    if (!mat || !mat.length) return "[-]";
    return `[${mat.length} x ${mat[0].length}]`;
}

// manual input POST manual_test then render bit comparison strings

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

// render teacher and student bit strings on two aligned rows
// each cluster is one col 1 or 0 if confident . if dead zone mismatches red
function renderManualResult(resp) {
    const setT = new Set(resp.idx_T);
    const setS = new Set(resp.idx_S);
    const mapT = new Map(resp.idx_T.map((c, i) => [c, resp.bits_T[i]]));
    const mapS = new Map(resp.idx_S.map((c, i) => [c, resp.bits_S[i]]));

    const nClusters = Math.max(
        ...resp.idx_T, ...resp.idx_S, -1,
    ) + 1;
    const total = Math.max(nClusters, resp.shared_indices.length);

    // summary lives next to manual status in form col
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

    // bit strings full width row below heatmaps
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

    // render two cluster heatmaps with trained cluster_size and dz so dead zone shading is right
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

// stress test POST stress_test render aggregated stats and histogram

function autoRunStressTest() {
    // fire as if user clicked the btn
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
    if (!Number.isFinite(n) || n < 1 || n > 200000) {
        flagError(els.stressStatus, "n_trials out of range [1, 200000]");
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
        const pmr = postConfirmMatchRate(resp);
        els.stressStatus.textContent =
            `Done in ${dt}s. ` +
            `Post-confirm match rate ${pmr.toFixed(2)}%: ` +
            `${resp.hash_accepted}/${resp.n_trials} trials accepted, ` +
            `${resp.hash_dropped} filtered by hash. ` +
            `Raw bit-level: ${resp.total_mismatches} mismatches across ` +
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

// match rate AFTER sha-256 confirmation: among accepted trials how many are truly equal
// silent_fail should be 0 in practice sha-256 collision is 2^-128
// returns 100 when no trials accepted edge case empty input
function postConfirmMatchRate(r) {
    if (r.hash_accepted <= 0) return 100;
    return 100 * (r.hash_accepted - r.hash_silent_fail) / r.hash_accepted;
}

function renderStressResult(r) {
    const fmt2 = (x) => Number.isFinite(x) ? x.toFixed(2) : "-";
    const total = r.total_clusters;
    const dropPct = 100 * r.hash_dropped / r.n_trials;
    const pmr = postConfirmMatchRate(r);
    const summary = `
        <div class="stress-summary-grid">
            <div class="metric ${pmr >= 99.999 ? 'match' : 'amber'}">
                <span class="metric-label">Match Rate (post-confirm)</span>
                <span class="metric-value mono">${pmr.toFixed(2)}%</span>
            </div>
            <div class="metric ${r.total_mismatches === 0 ? 'match' : 'amber'}">
                <span class="metric-label">Raw bit mismatches (sum)</span>
                <span class="metric-value mono">${r.total_mismatches}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Shared Bits (sum)</span>
                <span class="metric-value mono">${r.total_shared_bits}</span>
            </div>
        </div>
        <p class="stress-hash-line mono">
            <strong>SHA-256 confirmation:</strong>
            ${r.hash_dropped}/${r.n_trials} trials filtered by hash exchange
            (${dropPct.toFixed(2)}%),
            ${r.hash_silent_fail} silent failures.
        </p>
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
                    <td class="row-name">raw_mismatches</td>
                    <td>${fmt2(r.mismatches.mean)}</td>
                    <td>${fmt2(r.mismatches.std)}</td>
                    <td>${r.mismatches.min}</td>
                    <td>${r.mismatches.max}</td>
                    <td>${fmt2(r.mismatches.median)}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
        <p class="subtle stress-help">
            Per-trial counts over ${r.n_trials} N(0,1) inputs, evaluated at
            epoch ${r.epoch_used}. "of total" shows what fraction of the
            ${total} clusters survived dead-zone (dz=${r.dead_zone}) filtering on average.
            Match Rate is computed AFTER SHA-256 confirmation (trials filtered by hash
            are excluded from the denominator); raw_mismatches are pre-confirm bit-level.
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

    // resolve theme colors at draw time so palette swaps in styles.css just work
    const css = getComputedStyle(document.documentElement);
    const COL_GRID = css.getPropertyValue("--border-soft").trim() || "#1f1f1f";
    const COL_BAR  = css.getPropertyValue("--accent").trim()      || "#4a9eff";
    const COL_MEAN = css.getPropertyValue("--student").trim()     || "#f04747";
    const COL_AX   = css.getPropertyValue("--border-strong").trim() || "#3a3a3a";
    const COL_LBL  = css.getPropertyValue("--fg-dim").trim()      || "#a8a8a8";

    // bg grid 4 horizontal lines
    ctx.strokeStyle = COL_GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * i / 4;
        ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
    }
    ctx.stroke();

    // bars blue accent
    const barW = plotW / nBins;
    ctx.fillStyle = COL_BAR;
    for (let i = 0; i < nBins; i++) {
        const h = (counts[i] / maxCount) * plotH;
        const x = padL + i * barW;
        const y = padT + plotH - h;
        ctx.fillRect(x + 1, y, Math.max(1, barW - 2), h);
    }

    // mean line red distinct from blue bars
    if (Number.isFinite(mean)) {
        const lo = bins[0], hi = bins[bins.length - 1];
        const x = padL + plotW * (mean - lo) / Math.max(1e-9, hi - lo);
        ctx.strokeStyle = COL_MEAN;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
        ctx.setLineDash([]);
    }

    // axes
    ctx.strokeStyle = COL_AX;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
    ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
    ctx.stroke();

    // labels
    ctx.fillStyle = COL_LBL;
    ctx.font = "10px monospace";
    ctx.fillText(String(maxCount), 4, padT + 8);
    ctx.fillText("0", 4, H - padB + 4);
    ctx.fillText(`0`, padL, H - 6);
    ctx.fillText(`${total}`, W - padR - 18, H - 6);
    ctx.fillStyle = COL_MEAN;
    ctx.fillText(`mean ${mean.toFixed(1)}`, W - padR - 96, padT + 12);
}

// go

boot();
