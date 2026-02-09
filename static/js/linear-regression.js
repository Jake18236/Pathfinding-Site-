/**
 * Linear Regression Visualizer
 *
 * Interactive visualization of linear regression with normal equation
 * and gradient descent methods. Supports point placement, residuals,
 * loss curve plotting, and loss surface contour visualization.
 */
(function() {
    'use strict';

    const CANVAS_W = 560, CANVAS_H = 480;
    const POINT_R = 6;
    const AXIS_PAD = 40;
    const PLOT_L = AXIS_PAD, PLOT_T = 10;
    const PLOT_W = CANVAS_W - AXIS_PAD - 10;
    const PLOT_H = CANVAS_H - AXIS_PAD - 10;

    // Data range in data-space
    const DATA_MIN = 0, DATA_MAX = 10;

    // State
    let points = [];
    let fitLine = null;       // {m, b}
    let gdLine = null;        // {m, b} during GD animation
    let gdHistory = [];       // [{m, b, loss}]
    let gdTimer = null;
    let gdIteration = 0;
    let editMode = 'add';
    let showResiduals = false;
    let showEquation = true;
    let method = 'normal';

    // Canvas
    let canvas, ctx, dpr;
    let lossCanvas, lossCtx, lossDpr;
    let lossCurveCanvas, lossCurveCtx, lossCurveDpr;

    // ============================================
    // Coordinate transforms
    // ============================================
    function dataToCanvas(dx, dy) {
        const cx = PLOT_L + ((dx - DATA_MIN) / (DATA_MAX - DATA_MIN)) * PLOT_W;
        const cy = PLOT_T + PLOT_H - ((dy - DATA_MIN) / (DATA_MAX - DATA_MIN)) * PLOT_H;
        return { x: cx, y: cy };
    }

    function canvasToData(cx, cy) {
        const dx = DATA_MIN + ((cx - PLOT_L) / PLOT_W) * (DATA_MAX - DATA_MIN);
        const dy = DATA_MIN + ((PLOT_T + PLOT_H - cy) / PLOT_H) * (DATA_MAX - DATA_MIN);
        return { x: dx, y: dy };
    }

    // ============================================
    // Dataset generators
    // ============================================
    function generateDataset(type) {
        points = [];
        const n = 30;
        switch (type) {
            case 'linear':
                for (let i = 0; i < n; i++) {
                    const x = 1 + Math.random() * 8;
                    const y = 0.8 * x + 1.5 + (Math.random() - 0.5) * 2.5;
                    points.push({ x, y: Math.max(0.2, Math.min(9.8, y)) });
                }
                break;
            case 'quadratic':
                for (let i = 0; i < n; i++) {
                    const x = 0.5 + Math.random() * 9;
                    const y = 0.15 * (x - 5) * (x - 5) + 2 + (Math.random() - 0.5) * 1.5;
                    points.push({ x, y: Math.max(0.2, Math.min(9.8, y)) });
                }
                break;
            case 'sine':
                for (let i = 0; i < n; i++) {
                    const x = 0.5 + Math.random() * 9;
                    const y = 5 + 2.5 * Math.sin(x * 0.8) + (Math.random() - 0.5) * 1.5;
                    points.push({ x, y: Math.max(0.2, Math.min(9.8, y)) });
                }
                break;
            case 'cluster':
                const centers = [{ x: 2, y: 3 }, { x: 5, y: 5 }, { x: 8, y: 7 }];
                centers.forEach(c => {
                    for (let i = 0; i < 10; i++) {
                        points.push({
                            x: c.x + (Math.random() - 0.5) * 2,
                            y: c.y + (Math.random() - 0.5) * 2
                        });
                    }
                });
                break;
            case 'outliers':
                for (let i = 0; i < n - 3; i++) {
                    const x = 1 + Math.random() * 8;
                    const y = 0.7 * x + 2 + (Math.random() - 0.5) * 1.2;
                    points.push({ x, y: Math.max(0.2, Math.min(9.8, y)) });
                }
                points.push({ x: 2, y: 9 });
                points.push({ x: 8, y: 1 });
                points.push({ x: 5, y: 9.5 });
                break;
        }
    }

    // ============================================
    // Regression math
    // ============================================
    function normalEquation(pts) {
        if (pts.length < 2) return null;
        const n = pts.length;
        let sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (const p of pts) {
            sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y;
        }
        const denom = n * sxx - sx * sx;
        if (Math.abs(denom) < 1e-10) return null;
        const m = (n * sxy - sx * sy) / denom;
        const b = (sy - m * sx) / n;
        return { m, b };
    }

    function computeMSE(pts, m, b) {
        if (pts.length === 0) return 0;
        let sum = 0;
        for (const p of pts) {
            const err = p.y - (m * p.x + b);
            sum += err * err;
        }
        return sum / pts.length;
    }

    function computeR2(pts, m, b) {
        if (pts.length < 2) return 0;
        const mean = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        let ssTot = 0, ssRes = 0;
        for (const p of pts) {
            ssTot += (p.y - mean) ** 2;
            ssRes += (p.y - (m * p.x + b)) ** 2;
        }
        if (ssTot === 0) return 1;
        return 1 - ssRes / ssTot;
    }

    function gradientStep(pts, m, b, lr) {
        const n = pts.length;
        let dm = 0, db = 0;
        for (const p of pts) {
            const err = p.y - (m * p.x + b);
            dm += -2 * p.x * err / n;
            db += -2 * err / n;
        }
        return { m: m - lr * dm, b: b - lr * db };
    }

    // ============================================
    // Drawing
    // ============================================
    function getThemeColors() {
        const style = getComputedStyle(document.documentElement);
        return {
            point: style.getPropertyValue('--lr-point-color').trim() || '#377eb8',
            pointStroke: style.getPropertyValue('--lr-point-stroke').trim() || '#2a5f8f',
            fitLine: style.getPropertyValue('--lr-fit-color').trim() || '#e41a1c',
            residual: style.getPropertyValue('--lr-residual-color').trim() || 'rgba(76,175,80,0.5)',
            gdLine: style.getPropertyValue('--lr-gd-color').trim() || '#ff9800',
            text: style.getPropertyValue('--viz-text').trim() || '#333',
            muted: style.getPropertyValue('--viz-text-muted').trim() || '#6c757d',
            border: style.getPropertyValue('--viz-border').trim() || '#dee2e6',
            bg: style.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa',
        };
    }

    function render() {
        const c = getThemeColors();
        const CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(ctx, dpr);
        CU.clearCanvas(ctx, CANVAS_W, CANVAS_H, c.bg);

        drawAxes(c);
        drawGridLines(c);

        // Residuals
        const activeLine = gdLine || fitLine;
        if (showResiduals && activeLine && points.length > 0) {
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = c.residual;
            ctx.lineWidth = 1.5;
            for (const p of points) {
                const predicted = activeLine.m * p.x + activeLine.b;
                const from = dataToCanvas(p.x, p.y);
                const to = dataToCanvas(p.x, predicted);
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.stroke();
            }
            ctx.setLineDash([]);
        }

        // Fit line
        if (fitLine) {
            drawRegressionLine(fitLine, c.fitLine, 2.5);
        }

        // GD line (animated)
        if (gdLine && gdLine !== fitLine) {
            ctx.setLineDash([6, 4]);
            drawRegressionLine(gdLine, c.gdLine, 2);
            ctx.setLineDash([]);
        }

        // Points
        for (const p of points) {
            const cp = dataToCanvas(p.x, p.y);
            ctx.fillStyle = c.point;
            ctx.strokeStyle = c.pointStroke;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, POINT_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Equation label
        if (showEquation && activeLine) {
            const mStr = activeLine.m.toFixed(2);
            const bStr = activeLine.b >= 0 ? `+ ${activeLine.b.toFixed(2)}` : `− ${Math.abs(activeLine.b).toFixed(2)}`;
            const eq = `y = ${mStr}x ${bStr}`;
            ctx.font = '13px ' + getComputedStyle(document.documentElement).getPropertyValue('--viz-mono-font');
            ctx.fillStyle = c.fitLine;
            ctx.textAlign = 'left';
            ctx.fillText(eq, PLOT_L + 8, PLOT_T + 20);
        }

        // Overlay
        const overlay = document.getElementById('click-overlay');
        if (overlay) overlay.classList.toggle('hidden', points.length > 0);
    }

    function drawAxes(c) {
        ctx.strokeStyle = c.border;
        ctx.lineWidth = 1;
        // X axis
        ctx.beginPath();
        ctx.moveTo(PLOT_L, PLOT_T + PLOT_H);
        ctx.lineTo(PLOT_L + PLOT_W, PLOT_T + PLOT_H);
        ctx.stroke();
        // Y axis
        ctx.beginPath();
        ctx.moveTo(PLOT_L, PLOT_T);
        ctx.lineTo(PLOT_L, PLOT_T + PLOT_H);
        ctx.stroke();

        // Tick labels
        ctx.fillStyle = c.muted;
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        for (let v = 0; v <= 10; v += 2) {
            const pos = dataToCanvas(v, 0);
            ctx.fillText(v, pos.x, PLOT_T + PLOT_H + 16);
        }
        ctx.textAlign = 'right';
        for (let v = 0; v <= 10; v += 2) {
            const pos = dataToCanvas(0, v);
            ctx.fillText(v, PLOT_L - 6, pos.y + 4);
        }
    }

    function drawGridLines(c) {
        ctx.strokeStyle = c.border;
        ctx.lineWidth = 0.3;
        ctx.setLineDash([2, 4]);
        for (let v = 2; v <= 10; v += 2) {
            const h = dataToCanvas(0, v);
            ctx.beginPath(); ctx.moveTo(PLOT_L, h.y); ctx.lineTo(PLOT_L + PLOT_W, h.y); ctx.stroke();
            const vv = dataToCanvas(v, 0);
            ctx.beginPath(); ctx.moveTo(vv.x, PLOT_T); ctx.lineTo(vv.x, PLOT_T + PLOT_H); ctx.stroke();
        }
        ctx.setLineDash([]);
    }

    function drawRegressionLine(line, color, width) {
        const y0 = line.m * DATA_MIN + line.b;
        const y1 = line.m * DATA_MAX + line.b;
        const from = dataToCanvas(DATA_MIN, y0);
        const to = dataToCanvas(DATA_MAX, y1);

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();

        // Clip to plot area
        ctx.save();
        ctx.beginPath();
        ctx.rect(PLOT_L, PLOT_T, PLOT_W, PLOT_H);
        ctx.clip();

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        ctx.restore();
    }

    // ============================================
    // Loss curve
    // ============================================
    function renderLossCurve() {
        if (!lossCurveCanvas || gdHistory.length < 2) return;
        const CU = window.VizLib.CanvasUtils;
        const c = getThemeColors();
        const W = 300, H = 180;
        const pad = { l: 40, r: 10, t: 10, b: 25 };

        CU.resetCanvasTransform(lossCurveCtx, lossCurveDpr);
        CU.clearCanvas(lossCurveCtx, W, H, c.bg);

        const maxLoss = Math.max(...gdHistory.map(h => h.loss)) * 1.1;
        const pw = W - pad.l - pad.r;
        const ph = H - pad.t - pad.b;

        // Axes
        lossCurveCtx.strokeStyle = c.border;
        lossCurveCtx.lineWidth = 1;
        lossCurveCtx.beginPath();
        lossCurveCtx.moveTo(pad.l, pad.t);
        lossCurveCtx.lineTo(pad.l, pad.t + ph);
        lossCurveCtx.lineTo(pad.l + pw, pad.t + ph);
        lossCurveCtx.stroke();

        // Labels
        lossCurveCtx.fillStyle = c.muted;
        lossCurveCtx.font = '9px sans-serif';
        lossCurveCtx.textAlign = 'center';
        lossCurveCtx.fillText('Iteration', pad.l + pw / 2, H - 2);
        lossCurveCtx.save();
        lossCurveCtx.translate(10, pad.t + ph / 2);
        lossCurveCtx.rotate(-Math.PI / 2);
        lossCurveCtx.fillText('MSE', 0, 0);
        lossCurveCtx.restore();

        // Line
        lossCurveCtx.strokeStyle = c.fitLine;
        lossCurveCtx.lineWidth = 2;
        lossCurveCtx.beginPath();
        for (let i = 0; i < gdHistory.length; i++) {
            const x = pad.l + (i / (gdHistory.length - 1)) * pw;
            const y = pad.t + ph - (gdHistory[i].loss / maxLoss) * ph;
            if (i === 0) lossCurveCtx.moveTo(x, y);
            else lossCurveCtx.lineTo(x, y);
        }
        lossCurveCtx.stroke();
    }

    // ============================================
    // Loss surface (contour)
    // ============================================
    function renderLossSurface() {
        if (!lossCanvas || points.length < 2) return;
        const CU = window.VizLib.CanvasUtils;
        const c = getThemeColors();
        const W = 560, H = 280;
        const pad = { l: 50, r: 20, t: 15, b: 30 };
        const pw = W - pad.l - pad.r;
        const ph = H - pad.t - pad.b;

        CU.resetCanvasTransform(lossCtx, lossDpr);
        CU.clearCanvas(lossCtx, W, H, c.bg);

        // Determine parameter ranges centered on optimal
        const opt = normalEquation(points);
        if (!opt) return;

        const mRange = Math.max(2, Math.abs(opt.m) * 2);
        const bRange = Math.max(4, Math.abs(opt.b) * 2);
        const mMin = opt.m - mRange, mMax = opt.m + mRange;
        const bMin = opt.b - bRange, bMax = opt.b + bRange;

        // Compute loss grid
        const res = 60;
        const grid = [];
        let minLoss = Infinity, maxLoss = 0;
        for (let i = 0; i < res; i++) {
            grid[i] = [];
            for (let j = 0; j < res; j++) {
                const m = mMin + (mMax - mMin) * i / (res - 1);
                const b = bMin + (bMax - bMin) * j / (res - 1);
                const loss = computeMSE(points, m, b);
                grid[i][j] = loss;
                if (loss < minLoss) minLoss = loss;
                if (loss > maxLoss) maxLoss = loss;
            }
        }

        // Draw heatmap
        const cellW = pw / res;
        const cellH = ph / res;
        for (let i = 0; i < res; i++) {
            for (let j = 0; j < res; j++) {
                const t = Math.log(grid[i][j] - minLoss + 1) / Math.log(maxLoss - minLoss + 1);
                const r = Math.round(33 + t * 200);
                const g = Math.round(150 - t * 100);
                const b2 = Math.round(243 - t * 180);
                lossCtx.fillStyle = `rgb(${r},${g},${b2})`;
                lossCtx.fillRect(pad.l + i * cellW, pad.t + (res - 1 - j) * cellH, cellW + 1, cellH + 1);
            }
        }

        // Optimal point
        const optX = pad.l + ((opt.m - mMin) / (mMax - mMin)) * pw;
        const optY = pad.t + ph - ((opt.b - bMin) / (bMax - bMin)) * ph;
        lossCtx.fillStyle = '#fff';
        lossCtx.strokeStyle = c.fitLine;
        lossCtx.lineWidth = 2;
        lossCtx.beginPath();
        lossCtx.arc(optX, optY, 5, 0, Math.PI * 2);
        lossCtx.fill();
        lossCtx.stroke();

        // GD path
        if (gdHistory.length > 1) {
            lossCtx.strokeStyle = c.gdLine;
            lossCtx.lineWidth = 2;
            lossCtx.beginPath();
            for (let i = 0; i < gdHistory.length; i++) {
                const px = pad.l + ((gdHistory[i].m - mMin) / (mMax - mMin)) * pw;
                const py = pad.t + ph - ((gdHistory[i].b - bMin) / (bMax - bMin)) * ph;
                if (i === 0) lossCtx.moveTo(px, py);
                else lossCtx.lineTo(px, py);
            }
            lossCtx.stroke();

            // Current point
            const last = gdHistory[gdHistory.length - 1];
            const lx = pad.l + ((last.m - mMin) / (mMax - mMin)) * pw;
            const ly = pad.t + ph - ((last.b - bMin) / (bMax - bMin)) * ph;
            lossCtx.fillStyle = c.gdLine;
            lossCtx.beginPath();
            lossCtx.arc(lx, ly, 4, 0, Math.PI * 2);
            lossCtx.fill();
        }

        // Axes labels
        lossCtx.fillStyle = c.muted;
        lossCtx.font = '10px sans-serif';
        lossCtx.textAlign = 'center';
        lossCtx.fillText('slope (m)', pad.l + pw / 2, H - 4);
        lossCtx.save();
        lossCtx.translate(12, pad.t + ph / 2);
        lossCtx.rotate(-Math.PI / 2);
        lossCtx.fillText('intercept (b)', 0, 0);
        lossCtx.restore();

        // Tick labels
        lossCtx.fillStyle = c.muted;
        lossCtx.font = '9px sans-serif';
        for (let i = 0; i <= 4; i++) {
            const val = mMin + (mMax - mMin) * i / 4;
            const x = pad.l + (i / 4) * pw;
            lossCtx.textAlign = 'center';
            lossCtx.fillText(val.toFixed(1), x, H - 16);
        }
        lossCtx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = bMin + (bMax - bMin) * i / 4;
            const y = pad.t + ph - (i / 4) * ph;
            lossCtx.fillText(val.toFixed(1), pad.l - 5, y + 3);
        }
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        const setEl = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setEl('metric-points', points.length);

        const line = gdLine || fitLine;
        if (line) {
            setEl('metric-slope', line.m.toFixed(4));
            setEl('metric-intercept', line.b.toFixed(4));
            setEl('metric-mse', computeMSE(points, line.m, line.b).toFixed(4));
            setEl('metric-r2', computeR2(points, line.m, line.b).toFixed(4));
        } else {
            setEl('metric-slope', '-');
            setEl('metric-intercept', '-');
            setEl('metric-mse', '-');
            setEl('metric-r2', '-');
        }

        setEl('metric-iteration', gdIteration > 0 ? gdIteration : '-');
    }

    function setStatus(msg) {
        const el = document.getElementById('metric-status');
        if (el) el.textContent = msg;
        const ps = document.getElementById('playback-step');
        if (ps) ps.textContent = msg;
    }

    // ============================================
    // Actions
    // ============================================
    function doFit() {
        if (points.length < 2) {
            setStatus('Need at least 2 points');
            return;
        }

        if (method === 'normal') {
            fitLine = normalEquation(points);
            gdLine = null;
            gdHistory = [];
            gdIteration = 0;
            if (fitLine) {
                setStatus('Fit complete (normal equation)');
            }
            updateMetrics();
            render();
        } else {
            startGradientDescent();
        }
    }

    function startGradientDescent() {
        stopGD();
        const lr = parseFloat(document.getElementById('lr-slider').value);
        const maxIter = parseInt(document.getElementById('iter-value').value);

        // Initialize randomly
        gdLine = { m: (Math.random() - 0.5) * 4, b: (Math.random() - 0.5) * 4 + 5 };
        gdHistory = [{ m: gdLine.m, b: gdLine.b, loss: computeMSE(points, gdLine.m, gdLine.b) }];
        gdIteration = 0;
        fitLine = null;

        // Show loss panels
        document.getElementById('loss-panel').style.display = '';
        document.getElementById('loss-curve-panel').style.display = '';

        setStatus('Gradient descent running...');
        enableButtons(false);

        const speed = parseInt(document.getElementById('speed-slider').value);
        const delay = 1050 - speed * 100;

        function step() {
            if (gdIteration >= maxIter) {
                fitLine = { ...gdLine };
                setStatus(`Converged after ${gdIteration} iterations`);
                enableButtons(true);
                updateMetrics();
                render();
                renderLossCurve();
                renderLossSurface();
                return;
            }

            gdLine = gradientStep(points, gdLine.m, gdLine.b, lr);
            gdIteration++;
            const loss = computeMSE(points, gdLine.m, gdLine.b);
            gdHistory.push({ m: gdLine.m, b: gdLine.b, loss });

            updateMetrics();
            render();
            renderLossCurve();
            if (gdIteration % 5 === 0 || gdIteration <= 3) {
                renderLossSurface();
            }

            gdTimer = setTimeout(step, delay);
        }

        step();
    }

    function stopGD() {
        if (gdTimer) {
            clearTimeout(gdTimer);
            gdTimer = null;
        }
        enableButtons(true);
    }

    function doReset() {
        stopGD();
        fitLine = null;
        gdLine = null;
        gdHistory = [];
        gdIteration = 0;
        document.getElementById('loss-panel').style.display = 'none';
        document.getElementById('loss-curve-panel').style.display = 'none';
        setStatus('Ready');
        updateMetrics();
        render();
    }

    function doStep() {
        if (points.length < 2) return;
        const lr = parseFloat(document.getElementById('lr-slider').value);

        if (!gdLine) {
            gdLine = { m: (Math.random() - 0.5) * 4, b: (Math.random() - 0.5) * 4 + 5 };
            gdHistory = [{ m: gdLine.m, b: gdLine.b, loss: computeMSE(points, gdLine.m, gdLine.b) }];
            gdIteration = 0;
            fitLine = null;
            document.getElementById('loss-panel').style.display = '';
            document.getElementById('loss-curve-panel').style.display = '';
        }

        gdLine = gradientStep(points, gdLine.m, gdLine.b, lr);
        gdIteration++;
        gdHistory.push({ m: gdLine.m, b: gdLine.b, loss: computeMSE(points, gdLine.m, gdLine.b) });

        setStatus(`Step ${gdIteration}`);
        updateMetrics();
        render();
        renderLossCurve();
        renderLossSurface();
    }

    function enableButtons(enabled) {
        document.getElementById('btn-fit').disabled = !enabled;
        document.getElementById('btn-step').disabled = !enabled || method !== 'gradient';
        document.getElementById('btn-reset').disabled = !enabled;
    }

    // ============================================
    // Event handlers
    // ============================================
    function onCanvasClick(e) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const d = canvasToData(cx, cy);

        if (d.x < DATA_MIN || d.x > DATA_MAX || d.y < DATA_MIN || d.y > DATA_MAX) return;

        if (editMode === 'add') {
            points.push({ x: d.x, y: d.y });
        } else if (editMode === 'delete') {
            let closest = -1, minDist = 15;
            for (let i = 0; i < points.length; i++) {
                const cp = dataToCanvas(points[i].x, points[i].y);
                const dist = Math.hypot(cp.x - cx, cp.y - cy);
                if (dist < minDist) { minDist = dist; closest = i; }
            }
            if (closest >= 0) points.splice(closest, 1);
        }

        updateMetrics();
        render();
    }

    function init() {
        // Main canvas
        canvas = document.getElementById('regression-canvas');
        const setup = window.VizLib.CanvasUtils.setupHiDPICanvas(canvas);
        ctx = setup.ctx; dpr = setup.dpr;

        // Loss surface canvas
        lossCanvas = document.getElementById('loss-canvas');
        if (lossCanvas) {
            const ls = window.VizLib.CanvasUtils.setupHiDPICanvas(lossCanvas);
            lossCtx = ls.ctx; lossDpr = ls.dpr;
        }

        // Loss curve canvas
        lossCurveCanvas = document.getElementById('loss-curve-canvas');
        if (lossCurveCanvas) {
            const lc = window.VizLib.CanvasUtils.setupHiDPICanvas(lossCurveCanvas);
            lossCurveCtx = lc.ctx; lossCurveDpr = lc.dpr;
        }

        // Canvas click
        canvas.addEventListener('click', onCanvasClick);

        // Edit mode buttons
        document.querySelectorAll('.edit-mode-buttons .btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.edit-mode-buttons .btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                editMode = this.dataset.mode;
                canvas.style.cursor = editMode === 'delete' ? 'pointer' : 'crosshair';
            });
        });

        // Clear
        document.getElementById('btn-clear-points').addEventListener('click', () => {
            points = [];
            doReset();
        });

        // Dataset select
        document.getElementById('dataset-select').addEventListener('change', function() {
            if (this.value !== 'custom') {
                generateDataset(this.value);
                doReset();
            }
        });

        // Method select
        document.getElementById('method-select').addEventListener('change', function() {
            method = this.value;
            document.getElementById('gd-options').style.display = method === 'gradient' ? '' : 'none';
            document.getElementById('btn-step').disabled = method !== 'gradient';
            doReset();
        });

        // LR slider
        document.getElementById('lr-slider').addEventListener('input', function() {
            document.getElementById('lr-value').textContent = parseFloat(this.value).toFixed(3);
        });

        // Iteration stepper
        document.getElementById('iter-minus').addEventListener('click', () => {
            const el = document.getElementById('iter-value');
            el.value = Math.max(10, parseInt(el.value) - 50);
        });
        document.getElementById('iter-plus').addEventListener('click', () => {
            const el = document.getElementById('iter-value');
            el.value = Math.min(2000, parseInt(el.value) + 50);
        });

        // Checkboxes
        document.getElementById('show-residuals').addEventListener('change', function() {
            showResiduals = this.checked;
            render();
        });
        document.getElementById('show-equation').addEventListener('change', function() {
            showEquation = this.checked;
            render();
        });

        // Buttons
        document.getElementById('btn-fit').addEventListener('click', doFit);
        document.getElementById('btn-step').addEventListener('click', doStep);
        document.getElementById('btn-reset').addEventListener('click', doReset);

        // Info tabs
        document.querySelectorAll('.info-panel-tabs .btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const tabId = this.dataset.tab;
                document.querySelectorAll('.info-panel-tabs .btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                document.querySelectorAll('.info-tab-content').forEach(c => c.classList.remove('active'));
                const content = document.getElementById('tab-' + tabId);
                if (content) content.classList.add('active');
            });
        });

        // Theme change
        document.addEventListener('themechange', () => {
            render();
            if (gdHistory.length > 1) {
                renderLossCurve();
                renderLossSurface();
            }
        });

        render();
        updateMetrics();
    }

    // Wait for VizLib
    window.addEventListener('vizlib-ready', init);
})();
