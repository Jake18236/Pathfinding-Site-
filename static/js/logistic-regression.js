/**
 * Logistic Regression Visualizer
 *
 * Interactive visualization with sigmoid function, animated gradient descent,
 * decision boundary, and probability gradient display.
 */
(function() {
    'use strict';

    const CANVAS_W = 560, CANVAS_H = 400;
    const SIG_W = 560, SIG_H = 160;
    const PAD = 10;
    const POINT_R = 6;
    const BOUNDARY_RES = 60;

    // State
    let points = [];
    let weights = [0, 0]; // w1, w2
    let bias = 0;
    let trainedState = false;
    let lossHistory = [];
    let iteration = 0;
    let trainTimer = null;
    let editMode = 'add';
    let selectedClass = 0;
    let showProbability = true, showBoundaryLine = true;

    let canvas, ctx, dpr;
    let sigCanvas, sigCtx, sigDpr;
    let lossCurveCanvas, lossCurveCtx, lossCurveDpr;

    // ============================================
    // Math
    // ============================================
    function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z)))); }

    function predict(x, y) {
        return sigmoid(weights[0] * x + weights[1] * y + bias);
    }

    function computeLoss() {
        if (points.length === 0) return 0;
        let loss = 0;
        for (const p of points) {
            const prob = predict(p.x, p.y);
            const clipped = Math.max(1e-7, Math.min(1 - 1e-7, prob));
            loss -= p.classLabel * Math.log(clipped) + (1 - p.classLabel) * Math.log(1 - clipped);
        }
        return loss / points.length;
    }

    function gradientStep(lr) {
        const n = points.length;
        let dw0 = 0, dw1 = 0, db = 0;
        for (const p of points) {
            const prob = predict(p.x, p.y);
            const err = prob - p.classLabel;
            dw0 += err * p.x;
            dw1 += err * p.y;
            db += err;
        }
        weights[0] -= lr * dw0 / n;
        weights[1] -= lr * dw1 / n;
        bias -= lr * db / n;
    }

    function computeAccuracy() {
        if (points.length === 0) return 0;
        let correct = 0;
        for (const p of points) {
            const pred = predict(p.x, p.y) >= 0.5 ? 1 : 0;
            if (pred === p.classLabel) correct++;
        }
        return correct / points.length;
    }

    // ============================================
    // Coordinate transforms (data in [0,1])
    // ============================================
    function d2c(dx, dy) {
        return {
            x: PAD + dx * (CANVAS_W - 2 * PAD),
            y: PAD + (1 - dy) * (CANVAS_H - 2 * PAD)
        };
    }
    function c2d(cx, cy) {
        return {
            x: (cx - PAD) / (CANVAS_W - 2 * PAD),
            y: 1 - (cy - PAD) / (CANVAS_H - 2 * PAD)
        };
    }

    // ============================================
    // Dataset generators
    // ============================================
    function generateDataset(type) {
        points = [];
        const DG = window.VizLib && window.VizLib.DatasetGenerators;
        let raw = [];
        switch (type) {
            case 'linear': raw = DG ? DG.linear(80) : []; break;
            case 'overlap':
                for (let i = 0; i < 80; i++) {
                    const x = Math.random(), y = Math.random();
                    const cls = y > x + (Math.random() - 0.5) * 0.5 ? 1 : 0;
                    raw.push({ x, y, classLabel: cls });
                }
                break;
            case 'moons': raw = DG ? DG.moons(80, 0.12) : []; break;
            case 'xor': raw = DG ? DG.xor(80, 0.06) : []; break;
        }
        points = raw.map(p => ({ x: p.x, y: p.y, classLabel: p.classLabel }));
    }

    // ============================================
    // Drawing
    // ============================================
    function getColors() {
        const s = getComputedStyle(document.documentElement);
        return {
            class0: s.getPropertyValue('--viz-class-0').trim() || '#e41a1c',
            class1: s.getPropertyValue('--viz-class-1').trim() || '#377eb8',
            boundary: s.getPropertyValue('--logreg-boundary-color').trim() || '#333',
            bg: s.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa',
            muted: s.getPropertyValue('--viz-text-muted').trim() || '#6c757d',
            sigColor: s.getPropertyValue('--logreg-sigmoid-color').trim() || '#e41a1c',
            sigMarker: s.getPropertyValue('--logreg-sigmoid-marker').trim() || '#ff9800',
            border: s.getPropertyValue('--viz-border').trim() || '#dee2e6',
        };
    }

    function render() {
        const c = getColors();
        const CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(ctx, dpr);
        CU.clearCanvas(ctx, CANVAS_W, CANVAS_H, c.bg);

        // Probability gradient background
        if (showProbability && trainedState) {
            const cellW = (CANVAS_W - 2 * PAD) / BOUNDARY_RES;
            const cellH = (CANVAS_H - 2 * PAD) / BOUNDARY_RES;
            for (let i = 0; i < BOUNDARY_RES; i++) {
                for (let j = 0; j < BOUNDARY_RES; j++) {
                    const dx = i / (BOUNDARY_RES - 1);
                    const dy = j / (BOUNDARY_RES - 1);
                    const prob = predict(dx, dy);
                    // Color: class0 color at prob=0, class1 color at prob=1
                    const r0 = parseInt(c.class0.slice(1, 3), 16);
                    const g0 = parseInt(c.class0.slice(3, 5), 16);
                    const b0 = parseInt(c.class0.slice(5, 7), 16);
                    const r1 = parseInt(c.class1.slice(1, 3), 16);
                    const g1 = parseInt(c.class1.slice(3, 5), 16);
                    const b1 = parseInt(c.class1.slice(5, 7), 16);
                    const r = Math.round(r0 + (r1 - r0) * prob);
                    const g = Math.round(g0 + (g1 - g0) * prob);
                    const b = Math.round(b0 + (b1 - b0) * prob);
                    ctx.fillStyle = `rgba(${r},${g},${b},0.15)`;
                    ctx.fillRect(
                        PAD + i * cellW,
                        PAD + (BOUNDARY_RES - 1 - j) * cellH,
                        cellW + 1, cellH + 1
                    );
                }
            }
        }

        // Decision boundary line (where w·x + b = 0)
        if (showBoundaryLine && trainedState) {
            const wMag = Math.sqrt(weights[0] ** 2 + weights[1] ** 2);
            if (wMag > 1e-6) {
                ctx.strokeStyle = c.boundary;
                ctx.lineWidth = 2.5;
                ctx.setLineDash([]);

                // Find two endpoints on the canvas boundary
                const linePoints = [];
                // y = -(w0*x + b) / w1
                if (Math.abs(weights[1]) > 1e-6) {
                    for (const dx of [0, 1]) {
                        const dy = -(weights[0] * dx + bias) / weights[1];
                        if (dy >= -0.1 && dy <= 1.1) linePoints.push(d2c(dx, dy));
                    }
                }
                // x = -(w1*y + b) / w0
                if (Math.abs(weights[0]) > 1e-6) {
                    for (const dy of [0, 1]) {
                        const dx = -(weights[1] * dy + bias) / weights[0];
                        if (dx >= -0.1 && dx <= 1.1) linePoints.push(d2c(dx, dy));
                    }
                }

                if (linePoints.length >= 2) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(PAD, PAD, CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD);
                    ctx.clip();
                    ctx.beginPath();
                    ctx.moveTo(linePoints[0].x, linePoints[0].y);
                    ctx.lineTo(linePoints[1].x, linePoints[1].y);
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }

        // Points
        for (const p of points) {
            const cp = d2c(p.x, p.y);
            ctx.fillStyle = p.classLabel === 0 ? c.class0 : c.class1;
            ctx.strokeStyle = p.classLabel === 0 ? c.class0 : c.class1;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, POINT_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 0.4;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        const overlay = document.getElementById('click-overlay');
        if (overlay) overlay.classList.toggle('hidden', points.length > 0);
    }

    // ============================================
    // Sigmoid display
    // ============================================
    function renderSigmoid() {
        if (!sigCanvas) return;
        const c = getColors();
        const CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(sigCtx, sigDpr);
        CU.clearCanvas(sigCtx, SIG_W, SIG_H, c.bg);

        const pad = { l: 50, r: 20, t: 15, b: 25 };
        const pw = SIG_W - pad.l - pad.r;
        const ph = SIG_H - pad.t - pad.b;
        const zMin = -6, zMax = 6;

        // Axes
        sigCtx.strokeStyle = c.border;
        sigCtx.lineWidth = 1;
        sigCtx.beginPath();
        sigCtx.moveTo(pad.l, pad.t + ph);
        sigCtx.lineTo(pad.l + pw, pad.t + ph);
        sigCtx.stroke();
        sigCtx.beginPath();
        sigCtx.moveTo(pad.l, pad.t);
        sigCtx.lineTo(pad.l, pad.t + ph);
        sigCtx.stroke();

        // Grid lines
        sigCtx.strokeStyle = c.border;
        sigCtx.lineWidth = 0.3;
        sigCtx.setLineDash([2, 4]);
        // Horizontal at 0.5
        const halfY = pad.t + ph * 0.5;
        sigCtx.beginPath(); sigCtx.moveTo(pad.l, halfY); sigCtx.lineTo(pad.l + pw, halfY); sigCtx.stroke();
        // Vertical at 0
        const zeroX = pad.l + pw * 0.5;
        sigCtx.beginPath(); sigCtx.moveTo(zeroX, pad.t); sigCtx.lineTo(zeroX, pad.t + ph); sigCtx.stroke();
        sigCtx.setLineDash([]);

        // Labels
        sigCtx.fillStyle = c.muted;
        sigCtx.font = '10px sans-serif';
        sigCtx.textAlign = 'center';
        for (let z = -6; z <= 6; z += 2) {
            const x = pad.l + ((z - zMin) / (zMax - zMin)) * pw;
            sigCtx.fillText(z, x, pad.t + ph + 15);
        }
        sigCtx.textAlign = 'right';
        sigCtx.fillText('0', pad.l - 5, pad.t + ph + 3);
        sigCtx.fillText('0.5', pad.l - 5, halfY + 3);
        sigCtx.fillText('1', pad.l - 5, pad.t + 10);

        sigCtx.textAlign = 'center';
        sigCtx.fillText('z = w·x + b', pad.l + pw / 2, SIG_H - 2);

        // Sigmoid curve
        sigCtx.strokeStyle = c.sigColor;
        sigCtx.lineWidth = 2.5;
        sigCtx.beginPath();
        for (let i = 0; i <= pw; i++) {
            const z = zMin + (i / pw) * (zMax - zMin);
            const sv = sigmoid(z);
            const x = pad.l + i;
            const y = pad.t + ph - sv * ph;
            if (i === 0) sigCtx.moveTo(x, y);
            else sigCtx.lineTo(x, y);
        }
        sigCtx.stroke();

        // Mark threshold at 0.5
        sigCtx.fillStyle = c.sigMarker;
        sigCtx.beginPath();
        sigCtx.arc(zeroX, halfY, 5, 0, Math.PI * 2);
        sigCtx.fill();
    }

    // ============================================
    // Loss curve
    // ============================================
    function renderLossCurve() {
        if (!lossCurveCanvas || lossHistory.length < 2) return;
        const c = getColors();
        const CU = window.VizLib.CanvasUtils;
        const W = 300, H = 160;
        const pad = { l: 40, r: 10, t: 10, b: 25 };

        CU.resetCanvasTransform(lossCurveCtx, lossCurveDpr);
        CU.clearCanvas(lossCurveCtx, W, H, c.bg);

        const maxLoss = Math.max(...lossHistory) * 1.1;
        const pw = W - pad.l - pad.r;
        const ph = H - pad.t - pad.b;

        lossCurveCtx.strokeStyle = c.border;
        lossCurveCtx.lineWidth = 1;
        lossCurveCtx.beginPath();
        lossCurveCtx.moveTo(pad.l, pad.t);
        lossCurveCtx.lineTo(pad.l, pad.t + ph);
        lossCurveCtx.lineTo(pad.l + pw, pad.t + ph);
        lossCurveCtx.stroke();

        lossCurveCtx.fillStyle = c.muted;
        lossCurveCtx.font = '9px sans-serif';
        lossCurveCtx.textAlign = 'center';
        lossCurveCtx.fillText('Iteration', pad.l + pw / 2, H - 2);

        lossCurveCtx.strokeStyle = c.sigColor;
        lossCurveCtx.lineWidth = 2;
        lossCurveCtx.beginPath();
        for (let i = 0; i < lossHistory.length; i++) {
            const x = pad.l + (i / (lossHistory.length - 1)) * pw;
            const y = pad.t + ph - (lossHistory[i] / maxLoss) * ph;
            if (i === 0) lossCurveCtx.moveTo(x, y);
            else lossCurveCtx.lineTo(x, y);
        }
        lossCurveCtx.stroke();
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set('metric-points', points.length);
        set('metric-iteration', iteration > 0 ? iteration : '-');
        set('metric-loss', trainedState ? computeLoss().toFixed(4) : '-');
        set('metric-accuracy', trainedState ? (computeAccuracy() * 100).toFixed(1) + '%' : '-');
        set('metric-weights', trainedState ? `[${weights[0].toFixed(3)}, ${weights[1].toFixed(3)}]` : '-');
        set('metric-bias', trainedState ? bias.toFixed(4) : '-');
    }

    function setStatus(msg) {
        const el = document.getElementById('metric-status'); if (el) el.textContent = msg;
        const ps = document.getElementById('playback-step'); if (ps) ps.textContent = msg;
    }

    // ============================================
    // Actions
    // ============================================
    function doTrain() {
        if (points.length < 2) { setStatus('Need at least 2 points'); return; }
        stopTrain();

        weights = [(Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5];
        bias = 0;
        iteration = 0;
        lossHistory = [];
        trainedState = true;

        document.getElementById('loss-curve-panel').style.display = '';
        setStatus('Training...');

        const lr = parseFloat(document.getElementById('lr-slider').value);
        const maxIter = parseInt(document.getElementById('iter-value').value);
        const speed = parseInt(document.getElementById('speed-slider').value);
        const delay = Math.max(5, 1050 - speed * 100);

        function step() {
            if (iteration >= maxIter) {
                setStatus(`Done: ${iteration} iterations`);
                updateMetrics(); render(); renderLossCurve();
                return;
            }

            gradientStep(lr);
            iteration++;
            lossHistory.push(computeLoss());

            if (iteration % 5 === 0 || iteration <= 5 || iteration === maxIter) {
                updateMetrics(); render(); renderLossCurve();
            }

            trainTimer = setTimeout(step, delay);
        }
        step();
    }

    function doStep() {
        if (points.length < 2) return;
        if (!trainedState) {
            weights = [(Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5];
            bias = 0; iteration = 0; lossHistory = []; trainedState = true;
            document.getElementById('loss-curve-panel').style.display = '';
        }
        const lr = parseFloat(document.getElementById('lr-slider').value);
        gradientStep(lr);
        iteration++;
        lossHistory.push(computeLoss());
        setStatus(`Step ${iteration}`);
        updateMetrics(); render(); renderLossCurve();
    }

    function stopTrain() {
        if (trainTimer) { clearTimeout(trainTimer); trainTimer = null; }
    }

    function doReset() {
        stopTrain();
        weights = [0, 0]; bias = 0; trainedState = false;
        iteration = 0; lossHistory = [];
        document.getElementById('loss-curve-panel').style.display = 'none';
        setStatus('Ready');
        updateMetrics(); render();
    }

    // ============================================
    // Event handlers
    // ============================================
    function onCanvasClick(e) {
        const rect = canvas.getBoundingClientRect();
        const d = c2d(e.clientX - rect.left, e.clientY - rect.top);
        if (d.x < 0 || d.x > 1 || d.y < 0 || d.y > 1) return;

        if (editMode === 'add') {
            points.push({ x: d.x, y: d.y, classLabel: selectedClass });
        } else {
            let closest = -1, minDist = 20;
            for (let i = 0; i < points.length; i++) {
                const cp = d2c(points[i].x, points[i].y);
                const dist = Math.hypot(cp.x - (e.clientX - rect.left), cp.y - (e.clientY - rect.top));
                if (dist < minDist) { minDist = dist; closest = i; }
            }
            if (closest >= 0) points.splice(closest, 1);
        }
        if (trainedState) doReset(); else { updateMetrics(); render(); }
    }

    function init() {
        canvas = document.getElementById('logistic-canvas');
        const setup = window.VizLib.CanvasUtils.setupHiDPICanvas(canvas);
        ctx = setup.ctx; dpr = setup.dpr;

        sigCanvas = document.getElementById('sigmoid-canvas');
        if (sigCanvas) { const s = window.VizLib.CanvasUtils.setupHiDPICanvas(sigCanvas); sigCtx = s.ctx; sigDpr = s.dpr; }

        lossCurveCanvas = document.getElementById('loss-curve-canvas');
        if (lossCurveCanvas) { const l = window.VizLib.CanvasUtils.setupHiDPICanvas(lossCurveCanvas); lossCurveCtx = l.ctx; lossCurveDpr = l.dpr; }

        canvas.addEventListener('click', onCanvasClick);

        // Edit mode
        document.querySelectorAll('.edit-mode-buttons .btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.edit-mode-buttons .btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                editMode = this.dataset.mode;
                canvas.style.cursor = editMode === 'delete' ? 'pointer' : 'crosshair';
            });
        });

        // Class selector
        document.querySelectorAll('#class-selector .btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('#class-selector .btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                selectedClass = parseInt(this.dataset.class);
            });
        });

        document.getElementById('btn-clear-points').addEventListener('click', () => { points = []; doReset(); });

        document.getElementById('dataset-select').addEventListener('change', function() {
            if (this.value !== 'custom') { generateDataset(this.value); doReset(); }
        });

        document.getElementById('lr-slider').addEventListener('input', function() {
            document.getElementById('lr-value').textContent = parseFloat(this.value).toFixed(2);
        });

        document.getElementById('iter-minus').addEventListener('click', () => {
            const el = document.getElementById('iter-value');
            el.value = Math.max(10, parseInt(el.value) - 50);
        });
        document.getElementById('iter-plus').addEventListener('click', () => {
            const el = document.getElementById('iter-value');
            el.value = Math.min(5000, parseInt(el.value) + 50);
        });

        document.getElementById('show-probability').addEventListener('change', function() { showProbability = this.checked; render(); });
        document.getElementById('show-boundary').addEventListener('change', function() { showBoundaryLine = this.checked; render(); });

        document.getElementById('btn-train').addEventListener('click', doTrain);
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

        document.addEventListener('themechange', () => { render(); renderSigmoid(); renderLossCurve(); });

        render();
        renderSigmoid();
        updateMetrics();
    }

    window.addEventListener('vizlib-ready', init);
})();
