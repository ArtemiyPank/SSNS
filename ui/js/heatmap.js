// canvas heatmaps with hover tooltips
//
// one ImageData blit per render handles big sizes fine
// tooltip stays at data resolution mousemove maps back to row col
//
// exports
//   MatrixHeatmap        rows x cols heatmap with colormap used for weight matrices
//   ClusterStripHeatmap  1d post sigmoid drawn as 2d grid of cluster strips with frame color per bit decision

const TOOLTIP_OFFSET_X = 14;
const TOOLTIP_OFFSET_Y = 14;

// colormaps

// diverging blue red for weight matrices
// t in 0..1 mid is gray below is blue above is red
function colormapDiverging(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.5) {
        // negative side gray to blue
        const u = (0.5 - t) * 2;            // 0..1
        return [
            Math.round( 70 + ( 30 -  70) * u),  // R 70 to 30
            Math.round( 90 + ( 90 -  90) * u),  // G flat
            Math.round(120 + (255 - 120) * u),  // B 120 to 255
        ];
    } else {
        // positive side gray to red
        const u = (t - 0.5) * 2;
        return [
            Math.round(130 + (255 - 130) * u),  // R 130 to 255
            Math.round( 70 + ( 60 -  70) * u),  // G 70 to 60
            Math.round( 70 + ( 60 -  70) * u),  // B 70 to 60
        ];
    }
}

// per neuron sigmoid colormap v in 0..1
// blue to red gradient with brightness by abs v 0.5
//   v near 0   bright blue   confident 0
//   v near 0.5 dim mid       uncertain
//   v near 1   bright red    confident 1
//
// dead zone is a cluster level filter not per neuron
// so this colormap ignores deadZone cluster status shown by frame color
//
// deadZone arg kept for api compat but unused
// eslint-disable-next-line no-unused-vars
function makeSigmoidColormap(_deadZone) {
    return function (v) {
        v = Math.max(0, Math.min(1, v));
        // confidence 0 at v=0.5 1 at v=0 or v=1
        const conf = Math.min(1, Math.abs(v - 0.5) * 2);
        if (v < 0.5) {
            // blue side
            return [
                Math.round( 40 + ( 30 -  40) * conf),
                Math.round( 80 + (110 -  80) * conf),
                Math.round(140 + (255 - 140) * conf),
            ];
        }
        // red side
        return [
            Math.round(140 + (255 - 140) * conf),
            Math.round( 60 + ( 60 -  60) * conf),
            Math.round( 60 + ( 60 -  60) * conf),
        ];
    };
}

// generic matrix heatmap

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
        this.ctx.fillStyle = "#0a0a0a";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.data || !this.rows || !this.cols) {
            this.ctx.fillStyle = "#0a0a0a";
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

// vector heatmap single row strip for 1d activations like X or H
// tooltip shows label[i] = value with 4 decimals

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
        this.ctx.fillStyle = "#0a0a0a";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.values || !this.n) {
            this.ctx.fillStyle = "#0a0a0a";
            this.ctx.fillRect(0, 0, W, H);
            return;
        }
        const img = this.ctx.createImageData(W, H);
        const buf = img.data;
        const range = (this.max - this.min) || 1;
        const cmap = this.colormap;
        const n = this.n;
        const data = this.values;

        // single horizontal strip each col maps to a value
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

// cluster strip heatmap for Y_true Y_pred

// lays out 1d output_dim values as a near square grid of cluster strips
// each strip is one row of clusterSize cells tooltip shows idx mean and bit
export class ClusterStripHeatmap {
    constructor(canvas, opts = {}) {
        this.canvas      = canvas;
        this.ctx         = canvas.getContext("2d");
        this.tooltip     = opts.tooltip || null;
        this.label       = opts.label || "Y";
        this.clusterSize = opts.clusterSize || 5;
        this.deadZone    = opts.deadZone || 0.10;
        this.colormap    = makeSigmoidColormap(this.deadZone);
        this.values      = null;     // 1d
        this.gridCols    = 0;        // num clusters across
        this.gridRows    = 0;        // num cluster rows down
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
        // pick grid that roughly matches canvas aspect
        const aspect = this.canvas.width / this.canvas.height;
        // want gridCols * clusterSize / gridRows ~ aspect
        const target = Math.max(1, Math.round(Math.sqrt(this.nClusters * aspect / this.clusterSize)));
        this.gridCols = Math.max(1, target);
        this.gridRows = Math.ceil(this.nClusters / this.gridCols);
        this._render();
    }

    clear() {
        this.values = null;
        this.nClusters = 0;
        const W = this.canvas.width, H = this.canvas.height;
        this.ctx.fillStyle = "#0a0a0a";
        this.ctx.fillRect(0, 0, W, H);
    }

    _render() {
        const W = this.canvas.width, H = this.canvas.height;
        if (!this.values || !this.nClusters) {
            this.ctx.fillStyle = "#0a0a0a";
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
                    // empty padding cell
                    buf[i] = 18; buf[i + 1] = 22; buf[i + 2] = 30; buf[i + 3] = 255;
                    continue;
                }
                const v = this.values[cIdx * this.clusterSize + k];
                const rgb = cmap(v);
                buf[i] = rgb[0]; buf[i + 1] = rgb[1]; buf[i + 2] = rgb[2]; buf[i + 3] = 255;
            }
        }
        this.ctx.putImageData(img, 0, 0);

        // cluster status overlay each cluster gets a frame
        //   bit 1 mean >= 0.5+dz red frame
        //   bit 0 mean <= 0.5-dz blue frame
        //   dead zone gray frame plus dark overlay
        const css = getComputedStyle(document.documentElement);
        const COL_BG       = (css.getPropertyValue("--bg").trim()           || "#0a0a0a");
        const COL_BORDER   = (css.getPropertyValue("--border-strong").trim() || "#3a3a3a");
        const COL_TICK     = (css.getPropertyValue("--border-soft").trim()  || "#1f1f1f");
        const COL_LBL      = (css.getPropertyValue("--fg-dim").trim()       || "#a8a8a8");
        const COL_LBL_DEAD = (css.getPropertyValue("--fg-subtle").trim()    || "#6e6e6e");
        const COL_BIT0     = (css.getPropertyValue("--teacher").trim()      || "#4a9eff");
        const COL_BIT1     = (css.getPropertyValue("--student").trim()      || "#f04747");
        const cellW = W / totalCols;
        const cellH = H / totalRows;
        const stripW = this.clusterSize * cellW;

        const lo = 0.5 - this.deadZone;
        const hi = 0.5 + this.deadZone;

        // status for cluster cIdx
        const status = (cIdx) => {
            let sum = 0;
            for (let k = 0; k < this.clusterSize; k++) {
                sum += this.values[cIdx * this.clusterSize + k];
            }
            const mean = sum / this.clusterSize;
            if (mean >= hi) return { kind: "bit1", mean };
            if (mean <= lo) return { kind: "bit0", mean };
            return { kind: "dead", mean };
        };

        // pass 1 dim dead zone cells with dark overlay
        this.ctx.save();
        this.ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
        for (let r = 0; r < this.gridRows; r++) {
            for (let g = 0; g < this.gridCols; g++) {
                const cIdx = r * this.gridCols + g;
                if (cIdx >= this.nClusters) continue;
                const st = status(cIdx);
                if (st.kind !== "dead") continue;
                const x = Math.round(g * stripW);
                const y = Math.round(r * cellH);
                const w = Math.round(stripW);
                const h = Math.round(cellH);
                this.ctx.fillRect(x, y, w, h);
            }
        }
        this.ctx.restore();

        // pass 2 inner neuron ticks
        if (this.clusterSize > 1) {
            this.ctx.strokeStyle = COL_TICK;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            for (let c = 0; c < totalCols; c++) {
                if (c === 0) continue;
                if (c % this.clusterSize === 0) continue;
                const x = Math.round(c * cellW) + 0.5;
                this.ctx.moveTo(x, 0); this.ctx.lineTo(x, H);
            }
            this.ctx.stroke();
        }

        // pass 3 colored cluster frames red bit1 blue bit0 gray dead 2px inset
        const FRAME_W = 2;
        for (let r = 0; r < this.gridRows; r++) {
            for (let g = 0; g < this.gridCols; g++) {
                const cIdx = r * this.gridCols + g;
                if (cIdx >= this.nClusters) {
                    // padding cell faint outline only
                    const x = Math.round(g * stripW);
                    const y = Math.round(r * cellH);
                    const w = Math.round(stripW);
                    const h = Math.round(cellH);
                    this.ctx.strokeStyle = COL_BORDER;
                    this.ctx.lineWidth = 1;
                    this.ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
                    continue;
                }
                const st = status(cIdx);
                const colour = st.kind === "bit1" ? COL_BIT1
                              : st.kind === "bit0" ? COL_BIT0
                              : COL_BORDER;
                const x = Math.round(g * stripW);
                const y = Math.round(r * cellH);
                const w = Math.round(stripW);
                const h = Math.round(cellH);
                this.ctx.lineWidth = FRAME_W;
                this.ctx.strokeStyle = colour;
                this.ctx.strokeRect(
                    x + FRAME_W / 2,
                    y + FRAME_W / 2,
                    w - FRAME_W,
                    h - FRAME_W
                );
            }
        }

        // pass 4 cluster idx label in corner
        if (cellW * this.clusterSize >= 26 && cellH >= 16) {
            const fontPx = Math.min(11, Math.floor(cellH * 0.35));
            this.ctx.font = `${fontPx}px ${getComputedStyle(this.canvas).fontFamily || "monospace"}`;
            this.ctx.textBaseline = "top";
            for (let r = 0; r < this.gridRows; r++) {
                for (let g = 0; g < this.gridCols; g++) {
                    const cIdx = r * this.gridCols + g;
                    if (cIdx >= this.nClusters) continue;
                    const st = status(cIdx);
                    const x = Math.round(g * stripW) + 4;
                    const y = Math.round(r * cellH) + 3;
                    // label dim for dead normal otherwise
                    this.ctx.fillStyle = st.kind === "dead" ? COL_LBL_DEAD : COL_LBL;
                    this.ctx.fillText(String(cIdx), x, y);
                }
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

        // cluster mean and bit
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
        else                 bit = "-  (dead zone)";

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
