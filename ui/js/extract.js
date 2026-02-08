/**
 * Pure utilities — sigmoid, cluster-mean repetition-code bit extraction.
 *
 * Mirrors src/ssns_clean/bit_extract.py.extract_with_indices so the
 * frontend can derive Teacher/Student bits from the latest snapshot's
 * raw Y_true / Y_pred without an extra round-trip to the server.
 */

export function sigmoid(x) {
    // Numerically stable: avoid Math.exp overflow on large negative x.
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1.0 / (1.0 + z);
    }
    const z = Math.exp(x);
    return z / (1.0 + z);
}

export function sigmoidArr(arr) {
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = sigmoid(arr[i]);
    return out;
}

/**
 * Repetition-code + dead-zone bit extraction.
 * @param {number[]} values  flat 1D array of post-sigmoid values
 * @param {number}   clusterSize  neurons per bit
 * @param {number}   deadZone     half-width of discard band around 0.5
 * @returns {{bits: number[], indices: number[]}}
 */
export function extractWithIndices(values, clusterSize, deadZone) {
    const n = Math.floor(values.length / clusterSize);
    const lo = 0.5 - deadZone;
    const hi = 0.5 + deadZone;
    const bits = [];
    const indices = [];
    for (let c = 0; c < n; c++) {
        let sum = 0;
        const base = c * clusterSize;
        for (let k = 0; k < clusterSize; k++) sum += values[base + k];
        const mean = sum / clusterSize;
        if (mean >= hi) {
            bits.push(1);
            indices.push(c);
        } else if (mean <= lo) {
            bits.push(0);
            indices.push(c);
        }
    }
    return { bits, indices };
}

/**
 * Compute cluster-level metrics for a snapshot sample.
 * @returns {{
 *   teacher: {bits: number[], indices: number[]},
 *   student: {bits: number[], indices: number[]},
 *   shared: number[],
 *   mismatches: number,
 * }}
 */
export function computeMetrics(yTrueRaw, yPredRaw, clusterSize, deadZone) {
    const yT = sigmoidArr(yTrueRaw);
    const yS = sigmoidArr(yPredRaw);
    const teacher = extractWithIndices(yT, clusterSize, deadZone);
    const student = extractWithIndices(yS, clusterSize, deadZone);
    const setT = new Set(teacher.indices);
    const setS = new Set(student.indices);
    const shared = [];
    for (const idx of teacher.indices) if (setS.has(idx)) shared.push(idx);
    shared.sort((a, b) => a - b);
    const mapT = new Map(teacher.indices.map((c, i) => [c, teacher.bits[i]]));
    const mapS = new Map(student.indices.map((c, i) => [c, student.bits[i]]));
    let mismatches = 0;
    for (const c of shared) if (mapT.get(c) !== mapS.get(c)) mismatches++;
    return { teacher, student, shared, mismatches };
}
