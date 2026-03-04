/**
 * Support Vector Machine Visualizer
 *
 * Interactive visualization of SVM classification with linear, RBF, and
 * polynomial kernels. Supports multiclass via One-vs-Rest (OvR).
 * Uses simplified SMO-style training for real-time interaction.
 */
(function() {
    'use strict';

    const POINT_R = 6;
    const SV_R = 9;
    const PAD = 10;
    const BOUNDARY_RES = 150;
    const DRAG_BOUNDARY_RES = 60;

    // Dynamic canvas dimensions (set in setupCanvas)
    let canvasW = 560, canvasH = 480;

    // State
    let points = [];          // {x, y, classLabel: 0,1,2,...}
    let numClasses = 2;
    let classifiers = [];     // [{alphas, bias, classIdx, svIndices}] — one per class (OvR)
    let supportVectors = [];  // union of all classifiers' SV indices
    let trained = false;
    let boundaryCache = null; // precomputed class assignments (grid of class indices)
    let boundaryCacheRes = 0;
    let selectedClass = 0;
    let showMargin = true, showBoundary = true, showSV = true;
    let kernel = 'linear';
    let C = 10, gamma = 10, degree = 3;
    let hoveredPointIdx = -1;
    let dragIdx = -1;
    let dragStartPos = null;
    let dragLastPos = null;
    let dragMoved = false;
    let dragTrainTimer = null;

    let canvas, ctx, dpr;
    let tooltip;
    let resizeTimer = null;

    // ============================================
    // Canvas setup (dynamic sizing)
    // ============================================
    function setupCanvas() {
        canvas.style.width = '';
        canvas.style.height = '';
        var setup = window.VizLib.CanvasUtils.setupHiDPICanvas(canvas);
        ctx = setup.ctx;
        dpr = setup.dpr;
        canvasW = setup.logicalWidth;
        canvasH = setup.logicalHeight;
    }

    // ============================================
    // Coordinate transforms (data in [0,1])
    // ============================================
    function dataToCanvas(dx, dy) {
        return {
            x: PAD + dx * (canvasW - 2 * PAD),
            y: PAD + (1 - dy) * (canvasH - 2 * PAD)
        };
    }

    function canvasToData(cx, cy) {
        return {
            x: (cx - PAD) / (canvasW - 2 * PAD),
            y: 1 - (cy - PAD) / (canvasH - 2 * PAD)
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
    // Binary SMO Training (returns {alphas, bias, svIndices})
    // ============================================
    function trainBinary(labels, maxPasses, warmAlphas, warmBias) {
        var n = points.length;
        var alph, b;

        if (warmAlphas && warmAlphas.length === n) {
            alph = warmAlphas.slice();
            b = warmBias || 0;
        } else {
            alph = new Array(n).fill(0);
            b = 0;
        }

        // Precompute kernel matrix
        var K = [];
        for (var i = 0; i < n; i++) {
            K[i] = [];
            for (var j = 0; j < n; j++) {
                K[i][j] = kernelFn(points[i], points[j]);
            }
        }

        function decisionFn(idx) {
            var sum = b;
            for (var k = 0; k < n; k++) {
                if (alph[k] > 1e-8) sum += alph[k] * labels[k] * K[k][idx];
            }
            return sum;
        }

        var errors = new Array(n);
        for (var i = 0; i < n; i++) {
            errors[i] = decisionFn(i) - labels[i];
        }

        var tol = 1e-3;
        var passes = 0;
        var totalPasses = 0;
        var maxTotal = maxPasses * 10;

        while (passes < maxPasses && totalPasses < maxTotal) {
            totalPasses++;
            var numChanged = 0;

            for (var i = 0; i < n; i++) {
                errors[i] = decisionFn(i) - labels[i];
                var Ei = errors[i];

                if ((labels[i] * Ei < -tol && alph[i] < C) ||
                    (labels[i] * Ei > tol && alph[i] > 0)) {

                    var j = i;
                    while (j === i) j = Math.floor(Math.random() * n);

                    var Ej = errors[j];
                    var aiOld = alph[i], ajOld = alph[j];

                    var L, H;
                    if (labels[i] !== labels[j]) {
                        L = Math.max(0, alph[j] - alph[i]);
                        H = Math.min(C, C + alph[j] - alph[i]);
                    } else {
                        L = Math.max(0, alph[i] + alph[j] - C);
                        H = Math.min(C, alph[i] + alph[j]);
                    }

                    if (Math.abs(L - H) < 1e-10) continue;

                    var eta = 2 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    alph[j] -= labels[j] * (Ei - Ej) / eta;
                    alph[j] = Math.max(L, Math.min(H, alph[j]));

                    if (Math.abs(alph[j] - ajOld) < 1e-5) continue;

                    alph[i] += labels[i] * labels[j] * (ajOld - alph[j]);

                    var b1 = b - Ei - labels[i] * (alph[i] - aiOld) * K[i][i]
                                    - labels[j] * (alph[j] - ajOld) * K[i][j];
                    var b2 = b - Ej - labels[i] * (alph[i] - aiOld) * K[i][j]
                                    - labels[j] * (alph[j] - ajOld) * K[j][j];

                    if (alph[i] > 0 && alph[i] < C) b = b1;
                    else if (alph[j] > 0 && alph[j] < C) b = b2;
                    else b = (b1 + b2) / 2;

                    errors[i] = decisionFn(i) - labels[i];
                    errors[j] = decisionFn(j) - labels[j];

                    numChanged++;
                }
            }

            passes = numChanged === 0 ? passes + 1 : 0;
        }

        var svIndices = [];
        for (var i = 0; i < n; i++) {
            if (alph[i] > 1e-5) svIndices.push(i);
        }

        return { alphas: alph, bias: b, svIndices: svIndices };
    }

    // ============================================
    // OvR Multiclass Training
    // ============================================
    function train(maxPasses, warmStart) {
        maxPasses = maxPasses || 100;
        if (points.length < 2) return false;

        // Determine unique classes
        var classSet = new Set(points.map(function(p) { return p.classLabel; }));
        var classes = Array.from(classSet).sort(function(a, b) { return a - b; });
        numClasses = classes.length;

        if (numClasses < 2) {
            setStatus('Need at least 2 classes');
            return false;
        }

        var oldClassifiers = warmStart ? classifiers : [];
        classifiers = [];
        supportVectors = [];
        var svSet = {};

        for (var c = 0; c < classes.length; c++) {
            var cls = classes[c];
            // OvR labels: +1 for this class, -1 for all others
            var labels = points.map(function(p) { return p.classLabel === cls ? 1 : -1; });

            // Find warm-start data for this class if available
            var warmAlphas = null, warmBias = 0;
            if (warmStart && oldClassifiers.length === classes.length) {
                var old = oldClassifiers[c];
                if (old && old.classIdx === cls) {
                    warmAlphas = old.alphas;
                    warmBias = old.bias;
                }
            }

            var result = trainBinary(labels, maxPasses, warmAlphas, warmBias);
            result.classIdx = cls;
            classifiers.push(result);

            for (var s = 0; s < result.svIndices.length; s++) {
                svSet[result.svIndices[s]] = true;
            }
        }

        supportVectors = Object.keys(svSet).map(Number);
        trained = true;
        return true;
    }

    // ============================================
    // Decision functions
    // ============================================

    // Raw OvR score for a specific class classifier at a point
    function classifierScore(clf, px, py) {
        var sum = clf.bias;
        var pt = { x: px, y: py };
        for (var i = 0; i < points.length; i++) {
            if (clf.alphas[i] > 1e-8) {
                var label = points[i].classLabel === clf.classIdx ? 1 : -1;
                sum += clf.alphas[i] * label * kernelFn(points[i], pt);
            }
        }
        return sum;
    }

    // Predict class at (px, py) — returns {classIdx, score}
    function predict(px, py) {
        var bestClass = 0, bestScore = -Infinity;
        for (var c = 0; c < classifiers.length; c++) {
            var score = classifierScore(classifiers[c], px, py);
            if (score > bestScore) {
                bestScore = score;
                bestClass = classifiers[c].classIdx;
            }
        }
        return { classIdx: bestClass, score: bestScore };
    }

    // Binary-compatible decision value (for 2-class margin drawing)
    function decisionValue(px, py) {
        if (classifiers.length === 0) return 0;
        return classifierScore(classifiers[0], px, py);
    }

    // ============================================
    // Boundary computation
    // ============================================
    // boundaryCache stores per-classifier decision values for smooth contours
    // boundaryClassCache stores winning class index for region coloring
    var boundaryClassCache = null;

    function computeBoundary(res) {
        if (!trained) { boundaryCache = null; boundaryClassCache = null; boundaryCacheRes = 0; return; }
        res = res || BOUNDARY_RES;

        var nClf = classifiers.length;
        var cache = [];      // cache[clf][i][j] = decision value
        var classCache = []; // classCache[i][j] = winning class index

        for (var c = 0; c < nClf; c++) {
            cache[c] = [];
            for (var i = 0; i < res; i++) {
                cache[c][i] = new Float32Array(res);
            }
        }

        for (var i = 0; i < res; i++) {
            classCache[i] = [];
            for (var j = 0; j < res; j++) {
                var dx = i / (res - 1);
                var dy = j / (res - 1);
                var bestClass = 0, bestScore = -Infinity;
                for (var c = 0; c < nClf; c++) {
                    var score = classifierScore(classifiers[c], dx, dy);
                    cache[c][i][j] = score;
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = classifiers[c].classIdx;
                    }
                }
                classCache[i][j] = bestClass;
            }
        }

        boundaryCache = cache;
        boundaryClassCache = classCache;
        boundaryCacheRes = res;
    }

    // ============================================
    // Dataset generators
    // ============================================
    function generateDataset(type) {
        points = [];
        var DG = window.VizLib && window.VizLib.DatasetGenerators;

        function mapBinary(raw) {
            return raw.map(function(p) { return { x: p.x, y: p.y, classLabel: p.classLabel }; });
        }

        switch (type) {
            case 'linear': {
                var raw = DG ? DG.linear(80) : [];
                if (raw.length) {
                    points = mapBinary(raw);
                } else {
                    for (var i = 0; i < 40; i++) {
                        var x = Math.random(); var y = Math.random();
                        if (y > x + 0.1) points.push({ x: x, y: y, classLabel: 1 });
                        else if (y < x - 0.1) points.push({ x: x, y: y, classLabel: 0 });
                        else i--;
                    }
                }
                break;
            }
            case 'overlap': {
                for (var i = 0; i < 40; i++) {
                    var x = Math.random(), y = Math.random();
                    var cls = y > x + (Math.random() - 0.5) * 0.4 ? 1 : 0;
                    points.push({ x: x, y: y, classLabel: cls });
                }
                break;
            }
            case 'moons': {
                var raw = DG ? DG.moons(80, 0.1) : [];
                points = mapBinary(raw);
                break;
            }
            case 'circles': {
                var raw = DG ? DG.circles(80, 0.05) : [];
                points = mapBinary(raw);
                break;
            }
            case 'xor': {
                var raw = DG ? DG.xor(80, 0.05) : [];
                points = mapBinary(raw);
                break;
            }
            case 'blobs3': {
                var raw = DG ? DG.blobs(90, 3) : [];
                points = mapBinary(raw);
                break;
            }
            case 'blobs5': {
                var raw = DG ? DG.blobs(100, 5) : [];
                points = mapBinary(raw);
                break;
            }
        }

        // Update numClasses
        var classSet = new Set(points.map(function(p) { return p.classLabel; }));
        numClasses = classSet.size;
    }

    // ============================================
    // Drawing
    // ============================================
    function parseColor(str) {
        var m = str.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+))?\s*\)/);
        if (m) return [+m[1], +m[2], +m[3], Math.round((m[4] !== undefined ? +m[4] : 1) * 255)];
        var hex = str.replace('#', '');
        if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
        var n = parseInt(hex, 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 255];
    }

    function getColors() {
        var s = getComputedStyle(document.documentElement);
        var colors = {
            boundary: s.getPropertyValue('--svm-boundary-color').trim() || '#333',
            margin: s.getPropertyValue('--svm-margin-color').trim() || 'rgba(100,100,100,0.4)',
            svStroke: s.getPropertyValue('--svm-sv-stroke').trim() || '#8b5cf6',
            bg: s.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa',
            classColors: [],
            regionColors: []
        };

        // Read up to 10 class colors
        for (var i = 0; i < 10; i++) {
            var cc = s.getPropertyValue('--viz-class-' + i).trim();
            colors.classColors.push(cc || '#999');
        }

        // Region colors: class color at low opacity
        var regionOpacity = 0.08;
        for (var i = 0; i < colors.classColors.length; i++) {
            var rgb = parseColor(colors.classColors[i]);
            colors.regionColors.push('rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + regionOpacity + ')');
        }

        return colors;
    }

    function render() {
        var c = getColors();
        var CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(ctx, dpr);
        CU.clearCanvas(ctx, canvasW, canvasH, c.bg);

        // Decision boundary regions
        if (showBoundary && boundaryCache && boundaryClassCache) {
            var res = boundaryCacheRes;
            var bgRGBA = parseColor(c.bg);

            function blend(fg, bg) {
                var a = fg[3] / 255;
                return [
                    Math.round(bg[0] * (1 - a) + fg[0] * a),
                    Math.round(bg[1] * (1 - a) + fg[1] * a),
                    Math.round(bg[2] * (1 - a) + fg[2] * a),
                    255
                ];
            }

            var regionPixels = {};
            for (var ci = 0; ci < c.regionColors.length; ci++) {
                regionPixels[ci] = blend(parseColor(c.regionColors[ci]), bgRGBA);
            }

            var offCanvas = document.createElement('canvas');
            offCanvas.width = res;
            offCanvas.height = res;
            var offCtx = offCanvas.getContext('2d');
            var imgData = offCtx.createImageData(res, res);
            var data = imgData.data;

            // Use winning class from classCache for region coloring (works for both binary and multiclass)
            for (var i = 0; i < res; i++) {
                for (var j = 0; j < res; j++) {
                    var cls = boundaryClassCache[i][j];
                    var px = regionPixels[cls] || regionPixels[0];
                    var idx = ((res - 1 - j) * res + i) * 4;
                    data[idx]     = px[0];
                    data[idx + 1] = px[1];
                    data[idx + 2] = px[2];
                    data[idx + 3] = px[3];
                }
            }

            offCtx.putImageData(imgData, 0, 0);
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(offCanvas, PAD, PAD, canvasW - 2 * PAD, canvasH - 2 * PAD);

            // Contour lines
            if (numClasses === 2) {
                // Binary: single classifier boundary + margin
                drawContourFromGrid(boundaryCache[0], 0, c.boundary, 2.5);
                if (showMargin) {
                    drawContourFromGrid(boundaryCache[0], 1, c.margin, 1.5);
                    drawContourFromGrid(boundaryCache[0], -1, c.margin, 1.5);
                }
            } else {
                // Multiclass: draw where winning class changes using
                // difference between top-two classifier scores
                drawMulticlassContours(c.boundary);
            }
        }

        // Points
        for (var i = 0; i < points.length; i++) {
            var p = points[i];
            var cp = dataToCanvas(p.x, p.y);
            var isSV = showSV && trained && supportVectors.includes(i);

            if (isSV) {
                ctx.strokeStyle = c.svStroke;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(cp.x, cp.y, SV_R, 0, Math.PI * 2);
                ctx.stroke();
            }

            var classColor = c.classColors[p.classLabel] || c.classColors[0];
            ctx.fillStyle = classColor;
            ctx.strokeStyle = classColor;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, POINT_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // Hover highlight
        if (hoveredPointIdx >= 0 && hoveredPointIdx < points.length) {
            var hp = points[hoveredPointIdx];
            var hcp = dataToCanvas(hp.x, hp.y);
            ctx.strokeStyle = c.boundary;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(hcp.x, hcp.y, POINT_R + 4, 0, Math.PI * 2);
            ctx.stroke();
        }

        var overlay = document.getElementById('click-overlay');
        if (overlay) overlay.classList.toggle('hidden', points.length > 0);
    }

    function drawContourFromGrid(grid, level, color, width) {
        if (!grid) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (level !== 0) ctx.setLineDash([6, 4]);

        var res = boundaryCacheRes;
        var cellW = (canvasW - 2 * PAD) / (res - 1);
        var cellH = (canvasH - 2 * PAD) / (res - 1);

        ctx.beginPath();
        for (var i = 0; i < res - 1; i++) {
            for (var j = 0; j < res - 1; j++) {
                var v00 = grid[i][j] - level;
                var v10 = grid[i + 1][j] - level;
                var v01 = grid[i][j + 1] - level;
                var v11 = grid[i + 1][j + 1] - level;

                var cx = PAD + i * cellW;
                var cy = PAD + (res - 1 - j) * cellH;

                var segments = marchingSquareSegments(v00, v10, v01, v11, cx, cy, cellW, cellH);
                for (var s = 0; s < segments.length; s++) {
                    var seg = segments[s];
                    ctx.moveTo(seg.x1, seg.y1);
                    ctx.lineTo(seg.x2, seg.y2);
                }
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function drawMulticlassContours(color) {
        if (!boundaryCache || !boundaryClassCache) return;
        var res = boundaryCacheRes;
        var nClf = classifiers.length;
        var cellW = (canvasW - 2 * PAD) / (res - 1);
        var cellH = (canvasH - 2 * PAD) / (res - 1);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        // For each pair of classifiers, draw boundary where their scores are equal
        // but only in cells where these two are the top-two scorers
        for (var a = 0; a < nClf; a++) {
            for (var b = a + 1; b < nClf; b++) {
                var clsA = classifiers[a].classIdx;
                var clsB = classifiers[b].classIdx;

                for (var i = 0; i < res - 1; i++) {
                    for (var j = 0; j < res - 1; j++) {
                        // Check if either of these two classes wins at any corner
                        var c00 = boundaryClassCache[i][j];
                        var c10 = boundaryClassCache[i + 1][j];
                        var c01 = boundaryClassCache[i][j + 1];
                        var c11 = boundaryClassCache[i + 1][j + 1];

                        var relevant = (c00 === clsA || c00 === clsB) &&
                                       (c10 === clsA || c10 === clsB) &&
                                       (c01 === clsA || c01 === clsB) &&
                                       (c11 === clsA || c11 === clsB);
                        // Also include cells where the class actually changes between A and B
                        var hasA = c00 === clsA || c10 === clsA || c01 === clsA || c11 === clsA;
                        var hasB = c00 === clsB || c10 === clsB || c01 === clsB || c11 === clsB;
                        if (!(hasA && hasB)) continue;

                        // Difference grid: scoreA - scoreB
                        var v00 = boundaryCache[a][i][j] - boundaryCache[b][i][j];
                        var v10 = boundaryCache[a][i + 1][j] - boundaryCache[b][i + 1][j];
                        var v01 = boundaryCache[a][i][j + 1] - boundaryCache[b][i][j + 1];
                        var v11 = boundaryCache[a][i + 1][j + 1] - boundaryCache[b][i + 1][j + 1];

                        var cx = PAD + i * cellW;
                        var cy = PAD + (res - 1 - j) * cellH;

                        var segments = marchingSquareSegments(v00, v10, v01, v11, cx, cy, cellW, cellH);
                        for (var s = 0; s < segments.length; s++) {
                            var seg = segments[s];
                            ctx.moveTo(seg.x1, seg.y1);
                            ctx.lineTo(seg.x2, seg.y2);
                        }
                    }
                }
            }
        }
        ctx.stroke();
    }

    function marchingSquareSegments(v00, v10, v01, v11, cx, cy, cw, ch) {
        var segs = [];
        var config = (v00 > 0 ? 8 : 0) | (v10 > 0 ? 4 : 0) | (v11 > 0 ? 2 : 0) | (v01 > 0 ? 1 : 0);

        function interp(va, vb, pa, pb) {
            if (Math.abs(va - vb) < 1e-10) return (pa + pb) / 2;
            var t = va / (va - vb);
            return pa + t * (pb - pa);
        }

        var left   = { x: cx,      y: interp(v00, v01, cy, cy - ch) };
        var right  = { x: cx + cw, y: interp(v10, v11, cy, cy - ch) };
        var top    = { x: interp(v01, v11, cx, cx + cw), y: cy - ch };
        var bottom = { x: interp(v00, v10, cx, cx + cw), y: cy };

        var cases = {
            1: [[left, top]], 2: [[top, right]], 3: [[left, right]],
            4: [[bottom, right]], 5: [[left, bottom], [top, right]],
            6: [[bottom, top]], 7: [[left, bottom]], 8: [[left, bottom]],
            9: [[bottom, top]], 10: [[left, top], [bottom, right]],
            11: [[bottom, right]], 12: [[left, right]], 13: [[left, top]],
            14: [[top, right]]
        };

        var c = cases[config];
        if (c) {
            for (var k = 0; k < c.length; k++) {
                segs.push({ x1: c[k][0].x, y1: c[k][0].y, x2: c[k][1].x, y2: c[k][1].y });
            }
        }
        return segs;
    }

    // ============================================
    // Hover / Tooltip / Drag
    // ============================================
    function findNearestPoint(pos, threshold) {
        var closest = -1, minDist = threshold;
        for (var i = 0; i < points.length; i++) {
            var cp = dataToCanvas(points[i].x, points[i].y);
            var dist = Math.hypot(cp.x - pos.x, cp.y - pos.y);
            if (dist < minDist) { minDist = dist; closest = i; }
        }
        return closest;
    }

    function showTooltipForPoint(idx, pos) {
        var p = points[idx];
        var isSV = trained && supportVectors.includes(idx);
        var lines = [
            'Class: ' + p.classLabel,
            'x: ' + p.x.toFixed(3) + ', y: ' + p.y.toFixed(3)
        ];
        if (trained) {
            var pred = predict(p.x, p.y);
            lines.push('Predicted: ' + pred.classIdx + ' (score: ' + pred.score.toFixed(3) + ')');
            lines.push(isSV ? 'Support Vector' : 'Non-SV');
        }
        tooltip.innerHTML = lines.join('<br>');
        tooltip.classList.add('visible');
        tooltip.style.left = (pos.x + 12) + 'px';
        tooltip.style.top = (pos.y - 10) + 'px';
    }

    function scheduleDragRetrain() {
        if (dragTrainTimer) return;
        dragTrainTimer = requestAnimationFrame(function() {
            dragTrainTimer = null;
            if (dragIdx < 0) return;
            train(20, true);
            computeBoundary(DRAG_BOUNDARY_RES);
            updateMetrics();
            renderFitPanel();
            if (dragIdx >= 0) renderCalcPanel(dragIdx);
            setStatus('Trained: ' + supportVectors.length + ' SVs');
            render();
            if (dragLastPos && dragIdx >= 0) {
                showTooltipForPoint(dragIdx, dragLastPos);
            }
        });
    }

    function onCanvasMouseDown(e) {
        var pos = window.VizLib.CanvasUtils.getMousePosition(canvas, e);
        dragStartPos = pos;
        dragMoved = false;

        var near = findNearestPoint(pos, 15);
        if (near >= 0) {
            dragIdx = near;
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
            return;
        }
        dragIdx = -1;
    }

    function onCanvasMouseMove(e) {
        var pos = window.VizLib.CanvasUtils.getMousePosition(canvas, e);

        if (dragIdx >= 0) {
            dragMoved = true;
            var d = canvasToData(pos.x, pos.y);
            d.x = Math.max(0, Math.min(1, d.x));
            d.y = Math.max(0, Math.min(1, d.y));
            points[dragIdx].x = d.x;
            points[dragIdx].y = d.y;
            dragLastPos = pos;
            render();
            showTooltipForPoint(dragIdx, pos);
            if (trained) scheduleDragRetrain();
            return;
        }

        var closest = findNearestPoint(pos, 15);
        if (closest !== hoveredPointIdx) {
            hoveredPointIdx = closest;
            render();
        }

        if (hoveredPointIdx >= 0) {
            showTooltipForPoint(hoveredPointIdx, pos);
            renderCalcPanel(hoveredPointIdx);
            canvas.style.cursor = 'grab';
        } else {
            tooltip.classList.remove('visible');
            renderCalcPanel(-1);
            canvas.style.cursor = 'crosshair';
        }
    }

    function onCanvasMouseUp(e) {
        if (dragIdx >= 0) {
            if (dragTrainTimer) { cancelAnimationFrame(dragTrainTimer); dragTrainTimer = null; }
            if (trained && dragMoved) {
                train();
                computeBoundary(BOUNDARY_RES);
                updateMetrics();
                renderFitPanel();
                setStatus('Trained: ' + supportVectors.length + ' SVs');
                render();
            }
            dragIdx = -1;
            canvas.style.cursor = 'grab';
            return;
        }

        if (!dragMoved) {
            handleClick(e);
        }
        dragIdx = -1;
        dragMoved = false;
    }

    function onCanvasMouseLeave() {
        var needsRender = false;
        if (dragIdx >= 0) {
            if (dragTrainTimer) { cancelAnimationFrame(dragTrainTimer); dragTrainTimer = null; }
            if (trained) {
                train();
                computeBoundary(BOUNDARY_RES);
                updateMetrics();
                renderFitPanel();
                setStatus('Trained: ' + supportVectors.length + ' SVs');
            }
            dragIdx = -1;
            needsRender = true;
        }
        if (hoveredPointIdx >= 0) {
            hoveredPointIdx = -1;
            needsRender = true;
        }
        if (needsRender) render();
        tooltip.classList.remove('visible');
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        var set = function(id, val) { var el = document.getElementById(id); if (el) el.textContent = val; };

        set('metric-points', points.length);
        set('metric-kernel', kernel.charAt(0).toUpperCase() + kernel.slice(1));
        set('metric-classes', numClasses);

        if (trained) {
            set('metric-sv', supportVectors.length);

            var correct = 0;
            for (var k = 0; k < points.length; k++) {
                var pred = predict(points[k].x, points[k].y);
                if (pred.classIdx === points[k].classLabel) correct++;
            }
            set('metric-accuracy', (100 * correct / points.length).toFixed(1) + '%');

            if (kernel === 'linear' && numClasses === 2 && classifiers.length > 0) {
                var clf = classifiers[0];
                var w = [0, 0];
                for (var i = 0; i < points.length; i++) {
                    if (clf.alphas[i] > 1e-8) {
                        var label = points[i].classLabel === clf.classIdx ? 1 : -1;
                        w[0] += clf.alphas[i] * label * points[i].x;
                        w[1] += clf.alphas[i] * label * points[i].y;
                    }
                }
                var wNorm = Math.sqrt(w[0] ** 2 + w[1] ** 2);
                set('metric-margin', wNorm > 0 ? (2 / wNorm).toFixed(4) : '-');
            } else {
                set('metric-margin', numClasses > 2 ? 'N/A (multiclass)' : 'N/A (non-linear)');
            }
        } else {
            set('metric-sv', '-');
            set('metric-accuracy', '-');
            set('metric-margin', '-');
        }
    }

    function setStatus(msg) {
        var el = document.getElementById('metric-status');
        if (el) el.textContent = msg;
        var ps = document.getElementById('playback-step');
        if (ps) ps.textContent = msg;
    }

    // ============================================
    // Live Math Panels
    // ============================================

    function renderFitPanel() {
        var el = document.getElementById('svm-fit-panel');
        if (!el) return;

        if (!trained || classifiers.length === 0) {
            el.innerHTML = '<span class="formula-note">Train the SVM to see model parameters.</span>';
            return;
        }

        var html = '';

        // Model summary
        html += '<div class="svm-calc-label">Model Parameters</div>';
        html += '<div class="svm-param-grid">';
        html += '<div class="svm-calc-row"><span>Kernel</span><span class="svm-val-badge svm-val-neutral">' + kernel + '</span></div>';
        html += '<div class="svm-calc-row"><span>C</span><span class="svm-val-badge svm-val-neutral">' + C.toFixed(2) + '</span></div>';
        if (kernel === 'rbf' || kernel === 'polynomial') {
            html += '<div class="svm-calc-row"><span>Gamma</span><span class="svm-val-badge svm-val-neutral">' + gamma.toFixed(2) + '</span></div>';
        }
        if (kernel === 'polynomial') {
            html += '<div class="svm-calc-row"><span>Degree</span><span class="svm-val-badge svm-val-neutral">' + degree + '</span></div>';
        }
        html += '</div>';

        // OvR classifiers
        html += '<div class="svm-calc-label" style="margin-top:6px">OvR Classifiers (' + classifiers.length + ')</div>';

        for (var c = 0; c < classifiers.length; c++) {
            var clf = classifiers[c];
            var nSV = clf.svIndices.length;
            var classColor = getComputedStyle(document.documentElement).getPropertyValue('--viz-class-' + clf.classIdx).trim();
            html += '<div class="svm-calc-class-block">';
            html += '<div style="font-weight:700;font-size:10px;margin-bottom:2px;color:' + classColor + '">Class ' + clf.classIdx + ' vs Rest</div>';
            html += '<div class="svm-calc-row"><span>Support Vectors</span><span class="svm-val-badge svm-val-sv">' + nSV + '</span></div>';
            html += '<div class="svm-calc-row"><span>Bias (b)</span><span class="svm-val-badge svm-val-neutral">' + clf.bias.toFixed(4) + '</span></div>';

            // For linear kernel, show weight vector
            if (kernel === 'linear') {
                var w = [0, 0];
                for (var i = 0; i < points.length; i++) {
                    if (clf.alphas[i] > 1e-8) {
                        var label = points[i].classLabel === clf.classIdx ? 1 : -1;
                        w[0] += clf.alphas[i] * label * points[i].x;
                        w[1] += clf.alphas[i] * label * points[i].y;
                    }
                }
                html += '<div class="svm-calc-row"><span>w</span><span class="svm-val-badge svm-val-neutral">[' + w[0].toFixed(3) + ', ' + w[1].toFixed(3) + ']</span></div>';
            }

            html += '</div>';
        }

        // Decision function formula
        html += '<div class="svm-calc-label" style="margin-top:6px">Decision Function</div>';
        html += '<div style="font-size:10px;line-height:1.5">';
        if (kernel === 'linear') {
            html += '<code>f(x) = w&middot;x + b</code>';
        } else {
            html += '<code>f(x) = &Sigma;<sub>i</sub> &alpha;<sub>i</sub> y<sub>i</sub> K(x<sub>i</sub>, x) + b</code>';
        }
        if (numClasses > 2) {
            html += '<br><span style="color:var(--viz-text-muted)">Predict: argmax<sub>c</sub> f<sub>c</sub>(x)</span>';
        } else {
            html += '<br><span style="color:var(--viz-text-muted)">Predict: sign(f(x))</span>';
        }
        html += '</div>';

        el.innerHTML = html;
    }

    function renderCalcPanel(pointIdx) {
        var el = document.getElementById('svm-calc-panel');
        if (!el) return;

        if (pointIdx < 0 || !trained || classifiers.length === 0) {
            el.innerHTML = '<span class="formula-note">Hover a point to see the classification breakdown.</span>';
            return;
        }

        var p = points[pointIdx];
        var isSV = supportVectors.includes(pointIdx);
        var html = '';

        // Point info
        var pointColor = getComputedStyle(document.documentElement).getPropertyValue('--viz-class-' + p.classLabel).trim();
        html += '<div class="svm-calc-row" style="margin-bottom:4px">';
        html += '<span>Point #' + pointIdx + ' <span style="color:' + pointColor + ';font-weight:700">(Class ' + p.classLabel + ')</span></span>';
        if (isSV) {
            html += '<span class="svm-val-badge svm-val-sv">Support Vector</span>';
        }
        html += '</div>';
        html += '<div class="svm-calc-row" style="margin-bottom:6px;color:var(--viz-text-muted)">';
        html += '<span>x=' + p.x.toFixed(3) + ', y=' + p.y.toFixed(3) + '</span>';
        html += '</div>';

        // Per-classifier scores
        html += '<div class="svm-calc-label">OvR Classifier Scores</div>';

        var bestScore = -Infinity, bestClass = 0;
        var scores = [];
        for (var c = 0; c < classifiers.length; c++) {
            var clf = classifiers[c];
            var score = classifierScore(clf, p.x, p.y);
            scores.push({ classIdx: clf.classIdx, score: score });
            if (score > bestScore) {
                bestScore = score;
                bestClass = clf.classIdx;
            }
        }

        for (var c = 0; c < scores.length; c++) {
            var s = scores[c];
            var classColor = getComputedStyle(document.documentElement).getPropertyValue('--viz-class-' + s.classIdx).trim();
            var isWinner = s.classIdx === bestClass;
            var badgeClass = s.score >= 0 ? 'svm-val-positive' : 'svm-val-negative';

            html += '<div class="svm-calc-class-block' + (isWinner ? ' svm-calc-formula-row' : '') + '">';
            html += '<div class="svm-calc-row">';
            html += '<span style="color:' + classColor + ';font-weight:' + (isWinner ? '700' : '400') + '">f<sub>' + s.classIdx + '</sub>(x) — Class ' + s.classIdx + ' vs Rest</span>';
            html += '</div>';

            // Show the score computation breakdown
            var clf = classifiers[c];
            var svTerms = [];
            for (var i = 0; i < points.length; i++) {
                if (clf.alphas[i] > 1e-8) {
                    var label = points[i].classLabel === clf.classIdx ? 1 : -1;
                    var kVal = kernelFn(points[i], { x: p.x, y: p.y });
                    svTerms.push({
                        idx: i,
                        alpha: clf.alphas[i],
                        label: label,
                        kVal: kVal,
                        contrib: clf.alphas[i] * label * kVal
                    });
                }
            }

            // Show top contributing SVs (up to 3)
            svTerms.sort(function(a, b) { return Math.abs(b.contrib) - Math.abs(a.contrib); });
            var showCount = Math.min(svTerms.length, 3);
            for (var t = 0; t < showCount; t++) {
                var term = svTerms[t];
                var sign = term.contrib >= 0 ? '+' : '';
                html += '<div class="svm-calc-row" style="font-size:10px;color:var(--viz-text-muted)">';
                html += '<span>&alpha;·y·K(x<sub>' + term.idx + '</sub>,x)</span>';
                html += '<span>' + sign + term.contrib.toFixed(4) + '</span>';
                html += '</div>';
            }
            if (svTerms.length > showCount) {
                html += '<div class="svm-calc-row" style="font-size:10px;color:var(--viz-text-muted)">';
                html += '<span>...+' + (svTerms.length - showCount) + ' more SV terms</span>';
                html += '</div>';
            }

            // Bias + total
            html += '<div class="svm-calc-row" style="font-size:10px;color:var(--viz-text-muted)">';
            html += '<span>+ bias</span><span>' + (clf.bias >= 0 ? '+' : '') + clf.bias.toFixed(4) + '</span>';
            html += '</div>';

            html += '<div class="svm-calc-row svm-calc-formula-row">';
            html += '<span>= f<sub>' + s.classIdx + '</sub>(x)</span>';
            html += '<span class="svm-val-badge ' + badgeClass + '">' + s.score.toFixed(4) + '</span>';
            html += '</div>';

            html += '</div>';
        }

        // Final prediction
        var predColor = getComputedStyle(document.documentElement).getPropertyValue('--viz-class-' + bestClass).trim();
        var correct = bestClass === p.classLabel;
        html += '<div class="svm-calc-result">';
        html += '<div class="svm-calc-row">';
        html += '<span>Predicted</span>';
        html += '<span style="color:' + predColor + '">Class ' + bestClass;
        html += correct ? ' &#10003;' : ' &#10007;';
        html += '</span>';
        html += '</div>';
        html += '</div>';

        el.innerHTML = html;
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
        setTimeout(function() {
            var success = train();
            if (success) {
                computeBoundary();
                setStatus('Trained: ' + supportVectors.length + ' SVs (' + numClasses + ' classes)');
            } else {
                setStatus('Training failed — need at least 2 classes');
            }
            updateMetrics();
            renderFitPanel();
            renderCalcPanel(hoveredPointIdx);
            render();
        }, 10);
    }

    function doReset() {
        trained = false;
        classifiers = [];
        supportVectors = [];
        boundaryCache = null;
        boundaryClassCache = null;
        boundaryCacheRes = 0;
        setStatus('Ready');
        updateMetrics();
        renderFitPanel();
        renderCalcPanel(-1);
        render();
    }

    // ============================================
    // Event handlers
    // ============================================
    function handleClick(e) {
        var pos = window.VizLib.CanvasUtils.getMousePosition(canvas, e);
        var d = canvasToData(pos.x, pos.y);

        if (d.x < 0 || d.x > 1 || d.y < 0 || d.y > 1) return;

        if (findNearestPoint(pos, 15) >= 0) return;
        points.push({ x: d.x, y: d.y, classLabel: selectedClass });
        selectedClass = (selectedClass + 1) % numClasses;

        if (trained) {
            train();
            computeBoundary();
            updateMetrics();
            renderFitPanel();
            setStatus('Trained: ' + supportVectors.length + ' SVs');
            render();
        } else {
            updateMetrics();
            render();
        }
    }

    function onResize() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            setupCanvas();
            if (trained) computeBoundary();
            render();
        }, 150);
    }

    function init() {
        canvas = document.getElementById('svm-canvas');
        tooltip = document.getElementById('svm-tooltip');

        setupCanvas();

        canvas.addEventListener('mousedown', onCanvasMouseDown);
        canvas.addEventListener('mousemove', onCanvasMouseMove);
        canvas.addEventListener('mouseup', onCanvasMouseUp);
        canvas.addEventListener('mouseleave', onCanvasMouseLeave);
        window.addEventListener('resize', onResize);

        // Clear
        document.getElementById('btn-clear-points').addEventListener('click', function() {
            points = [];
            doReset();
        });

        // Dataset
        document.getElementById('dataset-select').addEventListener('change', function() {
            generateDataset(this.value);
            doTrain();
        });

        // Kernel — auto-retrain if already trained
        document.getElementById('kernel-select').addEventListener('change', function() {
            kernel = this.value;
            document.getElementById('gamma-row').style.display = (kernel === 'rbf' || kernel === 'polynomial') ? '' : 'none';
            document.getElementById('degree-row').style.display = kernel === 'polynomial' ? '' : 'none';
            if (trained) {
                doTrain();
            } else {
                updateMetrics();
                render();
            }
        });

        // C slider (log scale) — retrain live
        var cSlider = document.getElementById('c-slider');
        cSlider.addEventListener('input', function() {
            C = Math.pow(10, parseFloat(this.value));
            document.getElementById('c-value').textContent = C.toFixed(1);
            if (trained) doTrain();
        });

        // Gamma slider (log scale) — retrain live
        var gammaSlider = document.getElementById('gamma-slider');
        gammaSlider.addEventListener('input', function() {
            gamma = Math.pow(10, parseFloat(this.value));
            document.getElementById('gamma-value').textContent = gamma.toFixed(2);
            if (trained) doTrain();
        });

        // Degree stepper — auto-retrain
        document.getElementById('degree-minus').addEventListener('click', function() {
            degree = Math.max(2, degree - 1);
            document.getElementById('degree-value').value = degree;
            if (trained) doTrain();
        });
        document.getElementById('degree-plus').addEventListener('click', function() {
            degree = Math.min(6, degree + 1);
            document.getElementById('degree-value').value = degree;
            if (trained) doTrain();
        });

        // Checkboxes
        document.getElementById('show-margin').addEventListener('change', function() { showMargin = this.checked; render(); });
        document.getElementById('show-boundary').addEventListener('change', function() { showBoundary = this.checked; render(); });
        document.getElementById('show-sv').addEventListener('change', function() { showSV = this.checked; render(); });

        // Buttons
        document.getElementById('btn-train').addEventListener('click', doTrain);
        document.getElementById('btn-reset').addEventListener('click', doReset);

        // Info tabs
        document.querySelectorAll('.info-panel-tabs .btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var tabId = this.dataset.tab;
                document.querySelectorAll('.info-panel-tabs .btn').forEach(function(b) { b.classList.remove('active'); });
                this.classList.add('active');
                document.querySelectorAll('.info-tab-content').forEach(function(c) { c.classList.remove('active'); });
                var content = document.getElementById('tab-' + tabId);
                if (content) content.classList.add('active');
            });
        });

        // Theme
        document.addEventListener('themechange', function() {
            render();
        });

        // Initial gamma row visibility
        document.getElementById('gamma-row').style.display = 'none';

        // Load default dataset and train
        generateDataset('linear');
        document.getElementById('dataset-select').value = 'linear';
        train();
        computeBoundary();
        setStatus('Trained: ' + supportVectors.length + ' SVs');
        updateMetrics();
        renderFitPanel();
        render();
    }

    window.addEventListener('vizlib-ready', init);
})();
