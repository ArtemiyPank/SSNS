// stable sigmoid plus array variant
// matches nn::sigmoid in cpp side used when rendering output panels

function sigmoid(x) {
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
