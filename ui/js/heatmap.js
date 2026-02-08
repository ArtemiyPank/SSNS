/**
 * Canvas-based heatmaps with hover tooltips.
 *
 * One ImageData blit per re-render — handles 768×1280 ≈ 1M cells without
 * stutter on a modern laptop.  Tooltip resolution stays at the data level:
 * regardless of canvas pixel-density, mousemove maps the cursor back to
 * (row, col) and reports the exact float at that data cell.
 *
 * Two flavours are exposed:
 *   - MatrixHeatmap:  a generic (rows × cols) heatmap with a configurable
 *     colormap; used for the four weight matrices.
 *   - ClusterStripHeatmap:  a 1-D vector of post-sigmoid outputs visualised
 *     as a 2-D grid of cluster strips (cluster_size cells per strip),
 *     coloured by a sigmoid colormap that highlights confident bits and
 *     dims the dead-zone.
 */

const TOOLTIP_OFFSET_X = 14;
const TOOLTIP_OFFSET_Y = 14;

// ---------------------------------------------------------------------------
// Colormaps.
// ---------------------------------------------------------------------------

/** Diverging cool→warm (blue → near-black → red) for symmetric ranges. */
export function colormapDiverging(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.5) {
        const u = (0.5 - t) * 2;            // 0..1, 1 at the most-negative end
        const r = Math.round(40 + (90 - 40)   * (1 - u));
        const g = Math.round(70 + (130 - 70)  * (1 - u));
        const b = Math.round(160 + (220 - 160) * u);
        return [r, g, b];
    } else {
        const u = (t - 0.5) * 2;
        const r = Math.round(90 + (235 - 90)  * u);
        const g = Math.round(70 + (90 - 70)   * (1 - u));
        const b = Math.round(70 + (90 - 70)   * (1 - u));
        return [r, g, b];
    }
}

/**
 * Sigmoid-output colormap. v ∈ [0, 1].
 * Dead zone (|v - 0.5| < deadZone) renders dim grey to flag low confidence.
 * Below 0.5 → cool teal (bit 0 likely). Above 0.5 → warm orange (bit 1 likely).
 */
export function makeSigmoidColormap(deadZone) {
    return function (v) {
        v = Math.max(0, Math.min(1, v));
        if (Math.abs(v - 0.5) < deadZone) {
            // Dim grey ramp inside the dead zone.
            const u = Math.abs(v - 0.5) / deadZone;        // 0..1
            const lum = Math.round(54 + 18 * u);
            return [lum, lum, lum + 4];
        }
        if (v < 0.5) {
            const u = (0.5 - deadZone - v) / (0.5 - deadZone);  // 0..1
            return [
                Math.round(36 + (24 - 36) * u),
                Math.round(110 + (170 - 110) * u),
                Math.round(140 + (210 - 140) * u),
            ];
        }
        const u = (v - 0.5 - deadZone) / (0.5 - deadZone);
        return [
            Math.round(200 + (240 - 200) * u),
            Math.round(110 + (90 - 110) * u),
            Math.round(60 + (40 - 60) * u),
        ];
    };
}

// ---------------------------------------------------------------------------
// Generic matrix heatmap.
// ---------------------------------------------------------------------------

export class MatrixHeatmap {
    constructor(canvas, opts = {}) {
        this.canvas    = canvas;
        this.ctx       = canvas.getContext("2d");
        this.tooltip   = opts.tooltip || null;
        this.label     = opts.label || "";
        this.colormap  = opts.colormap || colormapDiverging;
        this.symmetric = opts.symmetric !== false;   // default true for weights
        this.formatter = opts.formatter || ((v) => v.toFixed(4));
        this.data      = null;
        this.rows      = 0;
        this.cols      = 0;
        this.min       = -1;
        this.max       = 1;
        this._bindHover();
    }

    setData(data2d, opts = {}) {
        this.data = data2d;
        this.rows = data2d.length;
        this.cols = this.rows ? data2d[0].length : 0;

        if (opts.min !== undefined && opts.max !== undefined) {
            this.min = opts.min;
            this.max = opts.max;
        } else {
            let lo = Infinity, hi = -Infinity;
            for (let r = 0; r < this.rows; r++) {
                const row = data2d[r];
                for (let c = 0; c < this.cols; c++) {
                    const v = row[c];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            if (this.symmetric) {
                const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
                this.min = -m; this.max = m;
            } else {
                this.min = lo; this.max = hi || 1;
            }
        }
        this._render();
    }

    clear() {
        this.data = null;
        this.rows = 0;
        this.cols = 0;
        const W = this.canvas.width, H = this.canvas.height;
        this.ctx.fillStyle = "#0e1117";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.data || !this.rows || !this.cols) {
            this.ctx.fillStyle = "#0e1117";
            this.ctx.fillRect(0, 0, W, H);
            return;
        }
        const img = this.ctx.createImageData(W, H);
        const buf = img.data;
        const range = (this.max - this.min) || 1;
        const rows = this.rows, cols = this.cols;
        const data = this.data;
        const cmap = this.colormap;

        for (let py = 0; py < H; py++) {
            const r = Math.min(rows - 1, (py * rows / H) | 0);
            const row = data[r];
            const baseY = py * W * 4;
            for (let px = 0; px < W; px++) {
                const c = Math.min(cols - 1, (px * cols / W) | 0);
                const v = row[c];
                const t = (v - this.min) / range;
                const rgb = cmap(t);
                const i = baseY + px * 4;
                buf[i]     = rgb[0];
                buf[i + 1] = rgb[1];
                buf[i + 2] = rgb[2];
                buf[i + 3] = 255;
            }
        }
        this.ctx.putImageData(img, 0, 0);
    }

    _bindHover() {
        this.canvas.addEventListener("mousemove", (e) => this._onMove(e));
        this.canvas.addEventListener("mouseleave", () => this._hideTooltip());
    }

    _onMove(e) {
        if (!this.data) return;
        const rect = this.canvas.getBoundingClientRect();
        const px = (e.clientX - rect.left) * this.canvas.width  / rect.width;
        const py = (e.clientY - rect.top)  * this.canvas.height / rect.height;
        const c  = Math.min(this.cols - 1, Math.max(0, (px * this.cols / this.canvas.width)  | 0));
        const r  = Math.min(this.rows - 1, Math.max(0, (py * this.rows / this.canvas.height) | 0));
        const v  = this.data[r][c];
        const text = `${this.label}[${r}, ${c}] = ${this.formatter(v)}`;
        this._showTooltip(e.clientX, e.clientY, text);
    }

    _showTooltip(x, y, text) {
        if (!this.tooltip) return;
        const t = this.tooltip;
        t.textContent = text;
        t.style.display = "block";
        t.style.left = (x + TOOLTIP_OFFSET_X) + "px";
        t.style.top  = (y + TOOLTIP_OFFSET_Y) + "px";
    }

    _hideTooltip() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }
}

// ---------------------------------------------------------------------------
// Vector heatmap — single-row strip for 1D activations (Input X, Hidden H).
// Tooltip reports `label[i] = value` with full precision (4dp).
// ---------------------------------------------------------------------------

export class VectorHeatmap {
    constructor(canvas, opts = {}) {
        this.canvas    = canvas;
        this.ctx       = canvas.getContext("2d");
        this.tooltip   = opts.tooltip || null;
        this.label     = opts.label || "";
        this.colormap  = opts.colormap || colormapDiverging;
        this.symmetric = opts.symmetric !== false;
        this.values    = null;
        this.n         = 0;
        this.min       = -1;
        this.max       = 1;
        this._bindHover();
    }

    setData(values1d, opts = {}) {
        this.values = values1d;
        this.n = values1d.length;

        if (opts.min !== undefined && opts.max !== undefined) {
            this.min = opts.min; this.max = opts.max;
        } else {
            let lo = Infinity, hi = -Infinity;
            for (let i = 0; i < this.n; i++) {
                const v = values1d[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (this.symmetric) {
                const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
                this.min = -m; this.max = m;
            } else {
                this.min = lo; this.max = hi || 1;
            }
        }
        this._render();
    }

    clear() {
        this.values = null;
        this.n = 0;
        const W = this.canvas.width, H = this.canvas.height;
        this.ctx.fillStyle = "#0e1117";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.values || !this.n) {
            this.ctx.fillStyle = "#0e1117";
            this.ctx.fillRect(0, 0, W, H);
            return;
        }
        const img = this.ctx.createImageData(W, H);
        const buf = img.data;
        const range = (this.max - this.min) || 1;
        const cmap = this.colormap;
        const n = this.n;
        const data = this.values;

        // Single horizontal strip — every pixel column maps to a value.
        for (let px = 0; px < W; px++) {
            const i = Math.min(n - 1, (px * n / W) | 0);
            const v = data[i];
            const t = (v - this.min) / range;
            const rgb = cmap(t);
            for (let py = 0; py < H; py++) {
                const k = (py * W + px) * 4;
                buf[k]     = rgb[0];
                buf[k + 1] = rgb[1];
                buf[k + 2] = rgb[2];
                buf[k + 3] = 255;
            }
        }
        this.ctx.putImageData(img, 0, 0);
    }

    _bindHover() {
        this.canvas.addEventListener("mousemove", (e) => this._onMove(e));
        this.canvas.addEventListener("mouseleave", () => this._hideTooltip());
    }

    _onMove(e) {
        if (!this.values) return;
        const rect = this.canvas.getBoundingClientRect();
        const px = (e.clientX - rect.left) * this.canvas.width / rect.width;
        const i = Math.min(this.n - 1, Math.max(0, (px * this.n / this.canvas.width) | 0));
        const v = this.values[i];
        this._showTooltip(e.clientX, e.clientY,
            `${this.label}[${i}] = ${v.toFixed(4)}`);
    }

    _showTooltip(x, y, text) {
        if (!this.tooltip) return;
        const t = this.tooltip;
        t.textContent = text;
        t.style.display = "block";
        t.style.left = (x + TOOLTIP_OFFSET_X) + "px";
        t.style.top  = (y + TOOLTIP_OFFSET_Y) + "px";
    }

    _hideTooltip() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }
}


// ---------------------------------------------------------------------------
// Cluster-strip heatmap (Y_true / Y_pred output viewer).
// ---------------------------------------------------------------------------

/**
 * Lay a 1-D vector of `output_dim = clusterSize × nClusters` post-sigmoid
 * values out as a near-square grid of cluster strips. Each strip is one
 * row of `clusterSize` cells; the strip itself is annotated by its cluster
 * index in the tooltip. Tooltip also reports the cluster mean and the
 * derived bit (or "—" if inside the dead zone).
 */
export class ClusterStripHeatmap {
    constructor(canvas, opts = {}) {
        this.canvas      = canvas;
        this.ctx         = canvas.getContext("2d");
        this.tooltip     = opts.tooltip || null;
        this.label       = opts.label || "Y";
        this.clusterSize = opts.clusterSize || 5;
        this.deadZone    = opts.deadZone || 0.10;
        this.colormap    = makeSigmoidColormap(this.deadZone);
        this.values      = null;     // 1D
        this.gridCols    = 0;        // clusters across
        this.gridRows    = 0;        // clusters down
        this.nClusters   = 0;
        this._bindHover();
    }

    setParams({ clusterSize, deadZone }) {
        if (clusterSize !== undefined) this.clusterSize = clusterSize;
        if (deadZone !== undefined) {
            this.deadZone = deadZone;
            this.colormap = makeSigmoidColormap(this.deadZone);
        }
    }

    setData(values1d) {
        this.values = values1d;
        this.nClusters = Math.floor(values1d.length / this.clusterSize);
        // Choose grid shape that approximates the canvas aspect ratio.
        const aspect = this.canvas.width / this.canvas.height;
        // We want gridCols * clusterSize / gridRows ≈ aspect.
        // Solve: gridCols ≈ sqrt(nClusters * aspect / clusterSize),
        //        gridRows = ceil(nClusters / gridCols)
        const target = Math.max(1, Math.round(Math.sqrt(this.nClusters * aspect / this.clusterSize)));
        this.gridCols = Math.max(1, target);
        this.gridRows = Math.ceil(this.nClusters / this.gridCols);
        this._render();
    }

    clear() {
        this.values = null;
        this.nClusters = 0;
        const W = this.canvas.width, H = this.canvas.height;
        this.ctx.fillStyle = "#0e1117";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.values || !this.nClusters) {
            this.ctx.fillStyle = "#0e1117";
            this.ctx.fillRect(0, 0, W, H);
            return;
        }
        const img = this.ctx.createImageData(W, H);
        const buf = img.data;
        const totalCols = this.gridCols * this.clusterSize;
        const totalRows = this.gridRows;
        const cmap = this.colormap;

        for (let py = 0; py < H; py++) {
            const gridR = Math.min(totalRows - 1, (py * totalRows / H) | 0);
            const baseY = py * W * 4;
            for (let px = 0; px < W; px++) {
                const cellCol = Math.min(totalCols - 1, (px * totalCols / W) | 0);
                const gridC   = (cellCol / this.clusterSize) | 0;
                const k       = cellCol - gridC * this.clusterSize;
                const cIdx    = gridR * this.gridCols + gridC;
                const i       = baseY + px * 4;

                if (cIdx >= this.nClusters) {
                    // Empty padding cell.
                    buf[i] = 18; buf[i + 1] = 22; buf[i + 2] = 30; buf[i + 3] = 255;
                    continue;
                }
                const v = this.values[cIdx * this.clusterSize + k];
                const rgb = cmap(v);
                buf[i] = rgb[0]; buf[i + 1] = rgb[1]; buf[i + 2] = rgb[2]; buf[i + 3] = 255;
            }
        }
        this.ctx.putImageData(img, 0, 0);

        // Faint vertical separators between cluster strips (rendered on top).
        if (this.gridCols > 1) {
            this.ctx.strokeStyle = "rgba(14, 17, 23, 0.55)";
            this.ctx.lineWidth = 1;
            for (let g = 1; g < this.gridCols; g++) {
                const x = Math.round((g * this.clusterSize) * (W / totalCols));
                this.ctx.beginPath();
                this.ctx.moveTo(x + 0.5, 0);
                this.ctx.lineTo(x + 0.5, H);
                this.ctx.stroke();
            }
        }
    }

    _bindHover() {
        this.canvas.addEventListener("mousemove", (e) => this._onMove(e));
        this.canvas.addEventListener("mouseleave", () => this._hideTooltip());
    }

    _onMove(e) {
        if (!this.values || !this.nClusters) return;
        const rect = this.canvas.getBoundingClientRect();
        const px = (e.clientX - rect.left) * this.canvas.width  / rect.width;
        const py = (e.clientY - rect.top)  * this.canvas.height / rect.height;
        const totalCols = this.gridCols * this.clusterSize;
        const totalRows = this.gridRows;
        const cellCol = Math.min(totalCols - 1, Math.max(0, (px * totalCols / this.canvas.width)  | 0));
        const gridR   = Math.min(totalRows - 1, Math.max(0, (py * totalRows / this.canvas.height) | 0));
        const gridC   = (cellCol / this.clusterSize) | 0;
        const k       = cellCol - gridC * this.clusterSize;
        const cIdx    = gridR * this.gridCols + gridC;

        if (cIdx >= this.nClusters) { this._hideTooltip(); return; }

        const v = this.values[cIdx * this.clusterSize + k];

        // Cluster mean + bit decision.
        let sum = 0;
        for (let kk = 0; kk < this.clusterSize; kk++) {
            sum += this.values[cIdx * this.clusterSize + kk];
        }
        const mean = sum / this.clusterSize;
        const lo = 0.5 - this.deadZone;
        const hi = 0.5 + this.deadZone;
        let bit;
        if (mean >= hi)      bit = "1 (confident)";
        else if (mean <= lo) bit = "0 (confident)";
        else                 bit = "—  (dead zone)";

        const text =
            `${this.label} cluster=${cIdx} pos=${k} | v=${v.toFixed(4)} | ` +
            `mean=${mean.toFixed(4)} | bit=${bit}`;
        this._showTooltip(e.clientX, e.clientY, text);
    }

    _showTooltip(x, y, text) {
        if (!this.tooltip) return;
        const t = this.tooltip;
        t.textContent = text;
        t.style.display = "block";
        t.style.left = (x + TOOLTIP_OFFSET_X) + "px";
        t.style.top  = (y + TOOLTIP_OFFSET_Y) + "px";
    }

    _hideTooltip() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }
}
