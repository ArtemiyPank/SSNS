// simple canvas line chart for loss no external lib
// log y when range spans more than two decades else linear
export class LossChart {
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.points = [];
        // read css vars so palette swaps just work
        const css = getComputedStyle(document.documentElement);
        const cssVar = (name, fallback) =>
            (css.getPropertyValue(name).trim() || fallback);
        this.gridColor   = opts.gridColor   || cssVar("--border-soft",   "#1f1f1f");
        this.lineColor   = opts.lineColor   || cssVar("--accent",        "#4a9eff");
        this.fillColor   = opts.fillColor   || "rgba(74, 158, 255, 0.12)";
        this.axisColor   = opts.axisColor   || cssVar("--border-strong", "#3a3a3a");
        this.labelColor  = opts.labelColor  || cssVar("--fg-dim",        "#a8a8a8");
    }

    // points like {epoch loss}
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

        // grid 5 horizontal lines also used for y label anchors
        const Y_TICKS = 5;
        ctx.strokeStyle = this.gridColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < Y_TICKS; i++) {
            const y = padT + (plotH * i / (Y_TICKS - 1));
            ctx.moveTo(padL, y); ctx.lineTo(W - padR, y);
        }
        ctx.stroke();

        // vertical grid lines at x ticks
        const X_TICKS = 5;
        ctx.strokeStyle = this.gridColor;
        ctx.beginPath();
        for (let i = 0; i < X_TICKS; i++) {
            const x = padL + (plotW * i / (X_TICKS - 1));
            ctx.moveTo(x, padT); ctx.lineTo(x, H - padB);
        }
        ctx.stroke();

        // axes
        ctx.strokeStyle = this.axisColor;
        ctx.beginPath();
        ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
        ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
        ctx.stroke();

        // fill below curve
        ctx.beginPath();
        ctx.moveTo(tx(this.points[0].epoch), H - padB);
        for (const p of this.points) ctx.lineTo(tx(p.epoch), ty(p.loss));
        ctx.lineTo(tx(this.points[this.points.length - 1].epoch), H - padB);
        ctx.closePath();
        ctx.fillStyle = this.fillColor;
        ctx.fill();

        // curve
        ctx.beginPath();
        this.points.forEach((p, i) => {
            const x = tx(p.epoch), y = ty(p.loss);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = this.lineColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // dot at last point
        const last = this.points[this.points.length - 1];
        ctx.fillStyle = this.lineColor;
        ctx.beginPath();
        ctx.arc(tx(last.epoch), ty(last.loss), 2.5, 0, Math.PI * 2);
        ctx.fill();

        // tick labels
        ctx.fillStyle = this.labelColor;
        ctx.font = "10px var(--mono, monospace)";
        const fmtLoss = (v) =>
            useLog ? v.toExponential(2)
                    : (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0)
                        ? v.toExponential(2)
                        : v.toFixed(3));

        // y axis one label per gridline top to bottom
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let i = 0; i < Y_TICKS; i++) {
            const t = i / (Y_TICKS - 1);
            const y = padT + plotH * t;
            const v = useLog
                ? Math.pow(10,
                    Math.log10(Math.max(lossMax, 1e-30)) * (1 - t)
                  + Math.log10(Math.max(lossMin, 1e-30)) * t)
                : (lossMax * (1 - t) + lossMin * t);
            ctx.fillText(fmtLoss(v), padL - 4, y);
        }

        // x axis 5 epoch ticks
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (let i = 0; i < X_TICKS; i++) {
            const t = i / (X_TICKS - 1);
            const x = padL + plotW * t;
            const epoch = Math.round(xMin + (xMax - xMin) * t);
            ctx.fillText(String(epoch), x, H - padB + 6);
        }

        // reset align and corner annotations
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        if (useLog) ctx.fillText("(log y)", W - padR - 38, padT + 10);
    }
}
