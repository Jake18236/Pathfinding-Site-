/**
 * Support Vector Machine Visualizer
 *
 * Interactive visualization of SVM classification with linear, RBF, and
 * polynomial kernels. Shows decision boundary, margin, and support vectors.
 * Uses simplified SMO-style training for real-time interaction.
 */
(function() {
    'use strict';

    const CANVAS_W = 560, CANVAS_H = 480;
    const POINT_R = 6;
    const SV_R = 9;
    const PAD = 10;
    const BOUNDARY_RES = 80;

    // State
    let points = [];          // {x, y, label: -1 or 1, classLabel: 0 or 1}
    let alphas = [];          // Lagrange multipliers
    let bias = 0;
    let supportVectors = [];  // indices of SVs
    let trained = false;
    let boundaryCache = null; // precomputed decision values
    let editMode = 'add';
    let selectedClass = 0;
    let showMargin = true, showBoundary = true, showSV = true;
    let kernel = 'linear';
    let C = 10, gamma = 1, degree = 3;

    let canvas, ctx, dpr;

    // ============================================
    // Coordinate transforms (data in [0,1])
    // ============================================
    function dataToCanvas(dx, dy) {
        return {
            x: PAD + dx * (CANVAS_W - 2 * PAD),
            y: PAD + (1 - dy) * (CANVAS_H - 2 * PAD)
        };
    }

    function canvasToData(cx, cy) {
        return {
            x: (cx - PAD) / (CANVAS_W - 2 * PAD),
            y: 1 - (cy - PAD) / (CANVAS_H - 2 * PAD)
        };
    }

    // ============================================
    // Kernel functions
    // ============================================
    function kernelFn(a, b) {
        switch (kernel) {
            case 'linear':
                return a.x * b.x + a.y * b.y;
            case 'rbf':
                return Math.exp(-gamma * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2));
            case 'polynomial':
                return Math.pow(a.x * b.x + a.y * b.y + 1, degree);
            default:
                return a.x * b.x + a.y * b.y;
        }
    }

    // ============================================
    // Simplified SMO Training
    // ============================================
    function train() {
        if (points.length < 2) return false;

        const n = points.length;
        alphas = new Array(n).fill(0);
        bias = 0;

        // Check if we have both classes
        const labels = new Set(points.map(p => p.label));
        if (labels.size < 2) {
            setStatus('Need both classes');
            return false;
        }

        // Precompute kernel matrix
        const K = [];
        for (let i = 0; i < n; i++) {
            K[i] = [];
            for (let j = 0; j < n; j++) {
                K[i][j] = kernelFn(points[i], points[j]);
            }
        }

        // Simplified SMO
        const tol = 1e-3;
        const maxPasses = 50;
        let passes = 0;

        while (passes < maxPasses) {
            let numChanged = 0;

            for (let i = 0; i < n; i++) {
                const Ei = decisionFnKernel(K, i) - points[i].label;

                if ((points[i].label * Ei < -tol && alphas[i] < C) ||
                    (points[i].label * Ei > tol && alphas[i] > 0)) {

                    // Pick random j != i
                    let j = i;
                    while (j === i) j = Math.floor(Math.random() * n);

                    const Ej = decisionFnKernel(K, j) - points[j].label;

                    const aiOld = alphas[i], ajOld = alphas[j];

                    // Compute bounds
                    let L, H;
                    if (points[i].label !== points[j].label) {
                        L = Math.max(0, alphas[j] - alphas[i]);
                        H = Math.min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = Math.max(0, alphas[i] + alphas[j] - C);
                        H = Math.min(C, alphas[i] + alphas[j]);
                    }

                    if (Math.abs(L - H) < 1e-10) continue;

                    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    alphas[j] -= points[j].label * (Ei - Ej) / eta;
                    alphas[j] = Math.max(L, Math.min(H, alphas[j]));

                    if (Math.abs(alphas[j] - ajOld) < 1e-5) continue;

                    alphas[i] += points[i].label * points[j].label * (ajOld - alphas[j]);

                    const b1 = bias - Ei - points[i].label * (alphas[i] - aiOld) * K[i][i]
                                           - points[j].label * (alphas[j] - ajOld) * K[i][j];
                    const b2 = bias - Ej - points[i].label * (alphas[i] - aiOld) * K[i][j]
                                           - points[j].label * (alphas[j] - ajOld) * K[j][j];

                    if (alphas[i] > 0 && alphas[i] < C) bias = b1;
                    else if (alphas[j] > 0 && alphas[j] < C) bias = b2;
                    else bias = (b1 + b2) / 2;

                    numChanged++;
                }
            }

            passes = numChanged === 0 ? passes + 1 : 0;
        }

        // Identify support vectors
        supportVectors = [];
        for (let i = 0; i < n; i++) {
            if (alphas[i] > 1e-5) supportVectors.push(i);
        }

        trained = true;
        return true;
    }

    function decisionFnKernel(K, idx) {
        let sum = bias;
        for (let i = 0; i < points.length; i++) {
            if (alphas[i] > 1e-8) {
                sum += alphas[i] * points[i].label * K[i][idx];
            }
        }
        return sum;
    }

    function decisionValue(px, py) {
        let sum = bias;
        const pt = { x: px, y: py };
        for (let i = 0; i < points.length; i++) {
            if (alphas[i] > 1e-8) {
                sum += alphas[i] * points[i].label * kernelFn(points[i], pt);
            }
        }
        return sum;
    }

    // ============================================
    // Boundary computation
    // ============================================
    function computeBoundary() {
        if (!trained) { boundaryCache = null; return; }

        const cache = [];
        for (let i = 0; i < BOUNDARY_RES; i++) {
            cache[i] = [];
            for (let j = 0; j < BOUNDARY_RES; j++) {
                const dx = i / (BOUNDARY_RES - 1);
                const dy = j / (BOUNDARY_RES - 1);
                cache[i][j] = decisionValue(dx, dy);
            }
        }
        boundaryCache = cache;
    }

    // ============================================
    // Dataset generators
    // ============================================
    function generateDataset(type) {
        points = [];
        const DG = window.VizLib && window.VizLib.DatasetGenerators;

        switch (type) {
            case 'linear': {
                const raw = DG ? DG.linear(80) : [];
                if (raw.length) {
                    points = raw.map(p => ({ x: p.x, y: p.y, classLabel: p.classLabel, label: p.classLabel === 0 ? -1 : 1 }));
                } else {
                    for (let i = 0; i < 40; i++) {
                        const x = Math.random(); const y = Math.random();
                        if (y > x + 0.1) points.push({ x, y, classLabel: 1, label: 1 });
                        else if (y < x - 0.1) points.push({ x, y, classLabel: 0, label: -1 });
                        else i--;
                    }
                }
                break;
            }
            case 'overlap': {
                for (let i = 0; i < 40; i++) {
                    const x = Math.random(), y = Math.random();
                    const cls = y > x + (Math.random() - 0.5) * 0.4 ? 1 : 0;
                    points.push({ x, y, classLabel: cls, label: cls === 0 ? -1 : 1 });
                }
                break;
            }
            case 'moons': {
                const raw = DG ? DG.moons(80, 0.1) : [];
                points = raw.map(p => ({ x: p.x, y: p.y, classLabel: p.classLabel, label: p.classLabel === 0 ? -1 : 1 }));
                break;
            }
            case 'circles': {
                const raw = DG ? DG.circles(80, 0.05) : [];
                points = raw.map(p => ({ x: p.x, y: p.y, classLabel: p.classLabel, label: p.classLabel === 0 ? -1 : 1 }));
                break;
            }
            case 'xor': {
                const raw = DG ? DG.xor(80, 0.05) : [];
                points = raw.map(p => ({ x: p.x, y: p.y, classLabel: p.classLabel, label: p.classLabel === 0 ? -1 : 1 }));
                break;
            }
        }
    }

    // ============================================
    // Drawing
    // ============================================
    function getColors() {
        const s = getComputedStyle(document.documentElement);
        return {
            class0: s.getPropertyValue('--viz-class-0').trim() || '#e41a1c',
            class1: s.getPropertyValue('--viz-class-1').trim() || '#377eb8',
            boundary: s.getPropertyValue('--svm-boundary-color').trim() || '#333',
            margin: s.getPropertyValue('--svm-margin-color').trim() || 'rgba(100,100,100,0.4)',
            svStroke: s.getPropertyValue('--svm-sv-stroke').trim() || '#ffd700',
            bg: s.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa',
            region0: s.getPropertyValue('--svm-boundary-region-0').trim() || 'rgba(228,26,28,0.08)',
            region1: s.getPropertyValue('--svm-boundary-region-1').trim() || 'rgba(55,126,184,0.08)',
        };
    }

    function render() {
        const c = getColors();
        const CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(ctx, dpr);
        CU.clearCanvas(ctx, CANVAS_W, CANVAS_H, c.bg);

        // Decision boundary regions
        if (showBoundary && boundaryCache) {
            const cellW = (CANVAS_W - 2 * PAD) / BOUNDARY_RES;
            const cellH = (CANVAS_H - 2 * PAD) / BOUNDARY_RES;

            for (let i = 0; i < BOUNDARY_RES; i++) {
                for (let j = 0; j < BOUNDARY_RES; j++) {
                    const val = boundaryCache[i][j];
                    if (val > 0) {
                        ctx.fillStyle = c.region1;
                    } else {
                        ctx.fillStyle = c.region0;
                    }
                    ctx.fillRect(
                        PAD + i * cellW,
                        PAD + (BOUNDARY_RES - 1 - j) * cellH,
                        cellW + 1, cellH + 1
                    );
                }
            }

            // Draw contour lines (boundary at 0, margin at ±1)
            drawContour(0, c.boundary, 2.5);
            if (showMargin) {
                drawContour(1, c.margin, 1.5);
                drawContour(-1, c.margin, 1.5);
            }
        }

        // Points
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const cp = dataToCanvas(p.x, p.y);
            const isSV = showSV && trained && supportVectors.includes(i);

            // Support vector highlight
            if (isSV) {
                ctx.strokeStyle = c.svStroke;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(cp.x, cp.y, SV_R, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Point
            ctx.fillStyle = p.classLabel === 0 ? c.class0 : c.class1;
            ctx.strokeStyle = p.classLabel === 0 ? c.class0 : c.class1;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, POINT_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // Overlay
        const overlay = document.getElementById('click-overlay');
        if (overlay) overlay.classList.toggle('hidden', points.length > 0);
    }

    function drawContour(level, color, width) {
        if (!boundaryCache) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (level !== 0) ctx.setLineDash([6, 4]);

        // Marching squares for contour extraction
        const cellW = (CANVAS_W - 2 * PAD) / (BOUNDARY_RES - 1);
        const cellH = (CANVAS_H - 2 * PAD) / (BOUNDARY_RES - 1);

        ctx.beginPath();
        for (let i = 0; i < BOUNDARY_RES - 1; i++) {
            for (let j = 0; j < BOUNDARY_RES - 1; j++) {
                const v00 = boundaryCache[i][j] - level;
                const v10 = boundaryCache[i + 1][j] - level;
                const v01 = boundaryCache[i][j + 1] - level;
                const v11 = boundaryCache[i + 1][j + 1] - level;

                const cx = PAD + i * cellW;
                const cy = PAD + (BOUNDARY_RES - 1 - j) * cellH;

                // Simple marching: draw segment where sign changes
                const segments = marchingSquareSegments(v00, v10, v01, v11, cx, cy, cellW, cellH);
                for (const seg of segments) {
                    ctx.moveTo(seg.x1, seg.y1);
                    ctx.lineTo(seg.x2, seg.y2);
                }
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function marchingSquareSegments(v00, v10, v01, v11, cx, cy, cw, ch) {
        const segs = [];
        const config = (v00 > 0 ? 8 : 0) | (v10 > 0 ? 4 : 0) | (v11 > 0 ? 2 : 0) | (v01 > 0 ? 1 : 0);

        function interp(va, vb, pa, pb) {
            if (Math.abs(va - vb) < 1e-10) return (pa + pb) / 2;
            const t = va / (va - vb);
            return pa + t * (pb - pa);
        }

        const left   = { x: cx,      y: interp(v00, v01, cy, cy - ch) };
        const right  = { x: cx + cw, y: interp(v10, v11, cy, cy - ch) };
        const top    = { x: interp(v01, v11, cx, cx + cw), y: cy - ch };
        const bottom = { x: interp(v00, v10, cx, cx + cw), y: cy };

        const cases = {
            1: [[left, top]], 2: [[top, right]], 3: [[left, right]],
            4: [[bottom, right]], 5: [[left, bottom], [top, right]],
            6: [[bottom, top]], 7: [[left, bottom]], 8: [[left, bottom]],
            9: [[bottom, top]], 10: [[left, top], [bottom, right]],
            11: [[bottom, right]], 12: [[left, right]], 13: [[left, top]],
            14: [[top, right]]
        };

        const c = cases[config];
        if (c) {
            for (const [a, b] of c) {
                segs.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
            }
        }
        return segs;
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

        set('metric-points', points.length);
        set('metric-kernel', kernel.charAt(0).toUpperCase() + kernel.slice(1));

        if (trained) {
            set('metric-sv', supportVectors.length);

            // Compute accuracy
            let correct = 0;
            for (const p of points) {
                const pred = decisionValue(p.x, p.y) >= 0 ? 1 : -1;
                if (pred === p.label) correct++;
            }
            set('metric-accuracy', (100 * correct / points.length).toFixed(1) + '%');

            // Margin width (for linear kernel)
            if (kernel === 'linear') {
                let w = [0, 0];
                for (let i = 0; i < points.length; i++) {
                    if (alphas[i] > 1e-8) {
                        w[0] += alphas[i] * points[i].label * points[i].x;
                        w[1] += alphas[i] * points[i].label * points[i].y;
                    }
                }
                const wNorm = Math.sqrt(w[0] ** 2 + w[1] ** 2);
                set('metric-margin', wNorm > 0 ? (2 / wNorm).toFixed(4) : '-');
            } else {
                set('metric-margin', 'N/A (non-linear)');
            }
        } else {
            set('metric-sv', '-');
            set('metric-accuracy', '-');
            set('metric-margin', '-');
        }
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
    function doTrain() {
        if (points.length < 4) {
            setStatus('Need at least 4 points');
            return;
        }

        setStatus('Training...');
        // Use setTimeout to allow UI update
        setTimeout(() => {
            const success = train();
            if (success) {
                computeBoundary();
                setStatus(`Trained: ${supportVectors.length} SVs`);
            } else {
                setStatus('Training failed — need both classes');
            }
            updateMetrics();
            render();
        }, 10);
    }

    function doReset() {
        trained = false;
        alphas = [];
        supportVectors = [];
        boundaryCache = null;
        bias = 0;
        setStatus('Ready');
        updateMetrics();
        render();
    }

    // ============================================
    // Event handlers
    // ============================================
    function onCanvasClick(e) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const d = canvasToData(cx, cy);

        if (d.x < 0 || d.x > 1 || d.y < 0 || d.y > 1) return;

        if (editMode === 'add') {
            const label = selectedClass === 0 ? -1 : 1;
            points.push({ x: d.x, y: d.y, classLabel: selectedClass, label });
        } else if (editMode === 'delete') {
            let closest = -1, minDist = 20;
            for (let i = 0; i < points.length; i++) {
                const cp = dataToCanvas(points[i].x, points[i].y);
                const dist = Math.hypot(cp.x - cx, cp.y - cy);
                if (dist < minDist) { minDist = dist; closest = i; }
            }
            if (closest >= 0) points.splice(closest, 1);
        }

        if (trained) doReset();
        else {
            updateMetrics();
            render();
        }
    }

    function init() {
        canvas = document.getElementById('svm-canvas');
        const setup = window.VizLib.CanvasUtils.setupHiDPICanvas(canvas);
        ctx = setup.ctx; dpr = setup.dpr;

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

        // Clear
        document.getElementById('btn-clear-points').addEventListener('click', () => {
            points = [];
            doReset();
        });

        // Dataset
        document.getElementById('dataset-select').addEventListener('change', function() {
            if (this.value !== 'custom') {
                generateDataset(this.value);
                doReset();
            }
        });

        // Kernel
        document.getElementById('kernel-select').addEventListener('change', function() {
            kernel = this.value;
            document.getElementById('gamma-row').style.display = kernel === 'rbf' ? '' : (kernel === 'polynomial' ? '' : 'none');
            document.getElementById('degree-row').style.display = kernel === 'polynomial' ? '' : 'none';
            if (kernel === 'linear') {
                document.getElementById('gamma-row').style.display = 'none';
            }
            doReset();
        });

        // C slider (log scale)
        document.getElementById('c-slider').addEventListener('input', function() {
            C = Math.pow(10, parseFloat(this.value));
            document.getElementById('c-value').textContent = C.toFixed(1);
        });

        // Gamma slider (log scale)
        document.getElementById('gamma-slider').addEventListener('input', function() {
            gamma = Math.pow(10, parseFloat(this.value));
            document.getElementById('gamma-value').textContent = gamma.toFixed(2);
        });

        // Degree stepper
        document.getElementById('degree-minus').addEventListener('click', () => {
            degree = Math.max(2, degree - 1);
            document.getElementById('degree-value').value = degree;
        });
        document.getElementById('degree-plus').addEventListener('click', () => {
            degree = Math.min(6, degree + 1);
            document.getElementById('degree-value').value = degree;
        });

        // Checkboxes
        document.getElementById('show-margin').addEventListener('change', function() { showMargin = this.checked; render(); });
        document.getElementById('show-boundary').addEventListener('change', function() { showBoundary = this.checked; render(); });
        document.getElementById('show-sv').addEventListener('change', function() { showSV = this.checked; render(); });

        // Buttons
        document.getElementById('btn-train').addEventListener('click', doTrain);
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

        // Theme
        document.addEventListener('themechange', () => {
            render();
        });

        // Initial gamma row visibility
        document.getElementById('gamma-row').style.display = 'none';

        render();
        updateMetrics();
    }

    window.addEventListener('vizlib-ready', init);
})();
