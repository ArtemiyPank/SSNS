/**
 * Vanilla canvas line chart for the loss curve. No external library — keeps
 * the IDE dependency-free and bundles fast under offline use.
 *
 * Uses log10(loss) on the Y axis when the loss range spans more than two
 * decades; otherwise linear.
 */
export class LossChart {
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.points = [];
        this.gridColor   = opts.gridColor   || "#21262d";
        this.lineColor   = opts.lineColor   || "#58a6ff";
        this.fillColor   = opts.fillColor   || "rgba(88, 166, 255, 0.12)";
        this.axisColor   = opts.axisColor   || "#30363d";
        this.labelColor  = opts.labelColor  || "#9da7b3";
    }

    /** points: array of {epoch: number, loss: number}. */
    setData(points) {
        this.points = (points || []).filter(p => Number.isFinite(p.loss) && p.loss > 0);
        this._render();
    }

    clear() {
        this.points = [];
        const W = this.canvas.width, H = this.canvas.height;
        this.ctx.clearRect(0, 0, W, H);
    }

    _render() {
        const ctx = this.ctx;
        const W = this.canvas.width, H = this.canvas.height;
        ctx.clearRect(0, 0, W, H);

        const padL = 48, padR = 12, padT = 12, padB = 26;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        if (!this.points.length) {
            ctx.fillStyle = this.labelColor;
            ctx.font = "11px var(--mono, monospace)";
            ctx.fillText("(no data)", padL + 4, padT + 14);
            return;
        }

        const xMin = this.points[0].epoch;
        const xMax = this.points[this.points.length - 1].epoch || (xMin + 1);
        const losses = this.points.map(p => p.loss);
        const lossMin = Math.min(...losses);
        const lossMax = Math.max(...losses);
        const useLog = (lossMax / Math.max(lossMin, 1e-30)) > 100;
        const tx = (e) => padL + ((e - xMin) / Math.max(1, xMax - xMin)) * plotW;
        const ty = (l) => {
            if (useLog) {
                const lmin = Math.log10(Math.max(lossMin, 1e-30));
                const lmax = Math.log10(Math.max(lossMax, lmin + 1e-9));
                const u = (Math.log10(Math.max(l, 1e-30)) - lmin) / (lmax - lmin);
                return padT + plotH * (1 - u);
            }
            const u = (l - lossMin) / Math.max(1e-30, lossMax - lossMin);
            return padT + plotH * (1 - u);
        };

        // Grid (4 horizontal lines).
        ctx.strokeStyle = this.gridColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= 4; i++) {
            const y = padT + (plotH * i / 4);
            ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
        }
        ctx.stroke();

        // Axes.
        ctx.strokeStyle = this.axisColor;
        ctx.beginPath();
        ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
        ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
        ctx.stroke();

        // Filled area below curve.
        ctx.beginPath();
        ctx.moveTo(tx(this.points[0].epoch), H - padB);
        for (const p of this.points) ctx.lineTo(tx(p.epoch), ty(p.loss));
        ctx.lineTo(tx(this.points[this.points.length - 1].epoch), H - padB);
        ctx.closePath();
        ctx.fillStyle = this.fillColor;
        ctx.fill();

        // Curve.
        ctx.beginPath();
        this.points.forEach((p, i) => {
            const x = tx(p.epoch), y = ty(p.loss);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = this.lineColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Last-point dot.
        const last = this.points[this.points.length - 1];
        ctx.fillStyle = this.lineColor;
        ctx.beginPath();
        ctx.arc(tx(last.epoch), ty(last.loss), 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Labels.
        ctx.fillStyle = this.labelColor;
        ctx.font = "10px monospace";
        const fmt = (v) => useLog ? v.toExponential(2) : v.toFixed(4);
        ctx.fillText(fmt(lossMax),     6, padT + 8);
        ctx.fillText(fmt(lossMin),     6, H - padB + 4);
        ctx.fillText(`epoch ${xMin}`,  padL,                 H - 6);
        ctx.fillText(`epoch ${xMax}`,  W - padR - 64,        H - 6);
        if (useLog) ctx.fillText("(log)", W - padR - 32, padT + 8);
    }
}
