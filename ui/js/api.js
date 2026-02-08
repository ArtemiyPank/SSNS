/**
 * Backend API wrappers — thin fetch helpers.
 */

const ENDPOINT = {
    trainingData:   "/api/training_data",
    trainingStatus: "/api/training_status",
    runTraining:    "/api/run_training",
    manualTest:     "/api/manual_test",
    stressTest:     "/api/stress_test",
};

export async function getTrainingData() {
    const r = await fetch(ENDPOINT.trainingData, { cache: "no-store" });
    if (r.status === 404) return null;          // no log yet
    if (!r.ok) throw new Error(`GET training_data → ${r.status}`);
    return await r.json();
}

export async function getTrainingStatus() {
    const r = await fetch(ENDPOINT.trainingStatus, { cache: "no-store" });
    if (r.status === 404) return null;          // no status file yet
    if (!r.ok) throw new Error(`GET training_status → ${r.status}`);
    return await r.json();
}

export async function postStressTest(n_trials, seed) {
    const body = { n_trials };
    if (seed !== undefined && seed !== null) body.seed = seed;
    const r = await fetch(ENDPOINT.stressTest, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
        const detail = data && data.detail
            ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
            : `HTTP ${r.status}`;
        throw new Error(detail);
    }
    return data;
}

export async function postRunTraining(params) {
    const r = await fetch(ENDPOINT.runTraining, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(params),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
        const detail = body && body.detail
            ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail))
            : `HTTP ${r.status}`;
        throw new Error(detail);
    }
    return body;
}

export async function postManualTest(X) {
    const r = await fetch(ENDPOINT.manualTest, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ X }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
        const detail = body && body.detail
            ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail))
            : `HTTP ${r.status}`;
        throw new Error(detail);
    }
    return body;
}
