/**
 * Clustering Visualizer
 *
 * Interactive visualization of K-Means clustering with step-by-step
 * centroid animation, Voronoi regions, and centroid history trails.
 * Supports preset datasets and custom point placement.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_W = 560, CANVAS_H = 480;
    const POINT_R = 5;
    const CENTROID_R = 10;
    const AXIS_PAD = 40;
    const PLOT_L = AXIS_PAD, PLOT_T = 10;
    const PLOT_W = CANVAS_W - AXIS_PAD - 10;
    const PLOT_H = CANVAS_H - AXIS_PAD - 10;
    const MAX_ITERATIONS = 100;
    const CONVERGENCE_THRESHOLD = 0.001;

    // Data space: [0, 1]
    const DATA_MIN = 0, DATA_MAX = 1;

    // ============================================
    // State
    // ============================================
    let points = [];            // [{x, y}] in data space [0,1]
    let K = 3;
    let centroids = [];         // [{x, y}] current centroids
    let assignments = [];       // [clusterIndex] per point
    let centroidHistory = [];   // [[{x, y}]] history per iteration
    let iteration = 0;
    let converged = false;
    let isRunning = false;
    let animTimer = null;
    let editMode = 'add';
    let showVoronoi = false;
    let showHistory = false;
    let algorithmPhase = 'idle'; // 'idle', 'assign', 'update', 'converged'

    // Canvas
    let canvas, ctx, dpr;

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
        assignments = [];
        const gen = window.VizLib.DatasetGenerators;

        switch (type) {
            case 'blobs': {
                // Use blobs generator, strip classLabel
                const raw = gen.blobs(120, 3);
                points = raw.map(function(p) { return { x: p.x, y: p.y }; });
                break;
            }
            case 'moons': {
                const raw = gen.moons(120, 0.1);
                points = raw.map(function(p) { return { x: p.x, y: p.y }; });
                break;
            }
            case 'circles': {
                const raw = gen.circles(120, 0.05);
                points = raw.map(function(p) { return { x: p.x, y: p.y }; });
                break;
            }
            case 'random': {
                for (var i = 0; i < 100; i++) {
                    points.push({
                        x: 0.05 + Math.random() * 0.9,
                        y: 0.05 + Math.random() * 0.9
                    });
                }
                break;
            }
            case 'custom':
                // Keep existing points or start empty
                break;
        }
    }

    // ============================================
    // K-Means algorithm
    // ============================================
    function initializeCentroids() {
        centroids = [];
        centroidHistory = [];
        assignments = new Array(points.length).fill(-1);
        iteration = 0;
        converged = false;
        algorithmPhase = 'idle';

        if (points.length === 0) return;

        // Random initialization: pick K random positions within data range
        var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (var i = 0; i < points.length; i++) {
            if (points[i].x < minX) minX = points[i].x;
            if (points[i].x > maxX) maxX = points[i].x;
            if (points[i].y < minY) minY = points[i].y;
            if (points[i].y > maxY) maxY = points[i].y;
        }

        // Add small padding to range
        var padX = (maxX - minX) * 0.1 || 0.05;
        var padY = (maxY - minY) * 0.1 || 0.05;

        for (var j = 0; j < K; j++) {
            centroids.push({
                x: minX + Math.random() * (maxX - minX + padX),
                y: minY + Math.random() * (maxY - minY + padY)
            });
        }

        // Save initial positions
        centroidHistory.push(centroids.map(function(c) { return { x: c.x, y: c.y }; }));
    }

    function euclideanDist(a, b) {
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function assignStep() {
        // Assign each point to the nearest centroid
        var changed = false;
        for (var i = 0; i < points.length; i++) {
            var minDist = Infinity;
            var minCluster = 0;
            for (var k = 0; k < centroids.length; k++) {
                var d = euclideanDist(points[i], centroids[k]);
                if (d < minDist) {
                    minDist = d;
                    minCluster = k;
                }
            }
            if (assignments[i] !== minCluster) {
                changed = true;
            }
            assignments[i] = minCluster;
        }
        algorithmPhase = 'assign';
        return changed;
    }

    function updateStep() {
        // Move centroids to the mean of their assigned points
        var totalMovement = 0;
        for (var k = 0; k < centroids.length; k++) {
            var sumX = 0, sumY = 0, count = 0;
            for (var i = 0; i < points.length; i++) {
                if (assignments[i] === k) {
                    sumX += points[i].x;
                    sumY += points[i].y;
                    count++;
                }
            }
            if (count > 0) {
                var newX = sumX / count;
                var newY = sumY / count;
                totalMovement += euclideanDist(centroids[k], { x: newX, y: newY });
                centroids[k] = { x: newX, y: newY };
            }
        }

        // Save centroid positions to history
        centroidHistory.push(centroids.map(function(c) { return { x: c.x, y: c.y }; }));

        algorithmPhase = 'update';
        return totalMovement;
    }

    function doOneIteration() {
        if (points.length < K) {
            setStatus('Need at least K points');
            return false;
        }

        // If centroids not initialized, do so
        if (centroids.length === 0) {
            initializeCentroids();
        }

        // Assignment step
        assignStep();

        // Update step
        var movement = updateStep();
        iteration++;

        // Check convergence
        if (movement < CONVERGENCE_THRESHOLD) {
            converged = true;
            algorithmPhase = 'converged';
            return false; // done
        }

        if (iteration >= MAX_ITERATIONS) {
            converged = true;
            algorithmPhase = 'converged';
            return false; // done
        }

        return true; // more iterations needed
    }

    function computeInertia() {
        if (centroids.length === 0 || points.length === 0) return 0;
        var inertia = 0;
        for (var i = 0; i < points.length; i++) {
            var k = assignments[i];
            if (k >= 0 && k < centroids.length) {
                var dx = points[i].x - centroids[k].x;
                var dy = points[i].y - centroids[k].y;
                inertia += dx * dx + dy * dy;
            }
        }
        return inertia;
    }

    // ============================================
    // Theme colors
    // ============================================
    function getThemeColors() {
        var style = getComputedStyle(document.documentElement);
        var colors = {
            bg: style.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa',
            text: style.getPropertyValue('--viz-text').trim() || '#333',
            muted: style.getPropertyValue('--viz-text-muted').trim() || '#6c757d',
            border: style.getPropertyValue('--viz-border').trim() || '#dee2e6',
            centroidStroke: style.getPropertyValue('--cluster-centroid-stroke').trim() || '#333333',
            centroidFill: style.getPropertyValue('--cluster-centroid-fill').trim() || '#ffffff',
            voronoiOpacity: parseFloat(style.getPropertyValue('--cluster-voronoi-opacity').trim()) || 0.12,
            historyOpacity: parseFloat(style.getPropertyValue('--cluster-history-opacity').trim()) || 0.4,
            clusterColors: []
        };

        for (var i = 0; i < 8; i++) {
            var c = style.getPropertyValue('--viz-class-' + i).trim();
            colors.clusterColors.push(c || '#999');
        }

        return colors;
    }

    // ============================================
    // Drawing
    // ============================================
    function render() {
        var c = getThemeColors();
        var CU = window.VizLib.CanvasUtils;
        CU.resetCanvasTransform(ctx, dpr);
        CU.clearCanvas(ctx, CANVAS_W, CANVAS_H, c.bg);

        drawAxes(c);
        drawGridLines(c);

        // Voronoi regions
        if (showVoronoi && centroids.length > 0) {
            drawVoronoi(c);
        }

        // Centroid history trails
        if (showHistory && centroidHistory.length > 1) {
            drawHistoryTrails(c);
        }

        // Points colored by cluster assignment
        for (var i = 0; i < points.length; i++) {
            var cp = dataToCanvas(points[i].x, points[i].y);
            var clusterIdx = assignments[i];
            if (clusterIdx >= 0 && clusterIdx < c.clusterColors.length) {
                ctx.fillStyle = c.clusterColors[clusterIdx];
            } else {
                ctx.fillStyle = c.muted;
            }
            ctx.strokeStyle = c.bg;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, POINT_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Centroids (diamond shape)
        for (var k = 0; k < centroids.length; k++) {
            var cc = dataToCanvas(centroids[k].x, centroids[k].y);
            drawDiamond(cc.x, cc.y, CENTROID_R, c.clusterColors[k] || c.muted, c.centroidStroke, c.centroidFill);
        }

        // Click overlay
        var overlay = document.getElementById('click-overlay');
        if (overlay) overlay.classList.toggle('hidden', points.length > 0);
    }

    function drawDiamond(cx, cy, size, ringColor, strokeColor, fillColor) {
        // Outer colored ring
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(Math.PI / 4);

        // Filled background
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.rect(-size / 2, -size / 2, size, size);
        ctx.fill();
        ctx.stroke();

        // Inner border
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(-size / 2, -size / 2, size, size);
        ctx.stroke();

        ctx.restore();
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
        for (var v = 0; v <= 1; v += 0.2) {
            var pos = dataToCanvas(v, 0);
            ctx.fillText(v.toFixed(1), pos.x, PLOT_T + PLOT_H + 16);
        }
        ctx.textAlign = 'right';
        for (var v2 = 0; v2 <= 1; v2 += 0.2) {
            var pos2 = dataToCanvas(0, v2);
            ctx.fillText(v2.toFixed(1), PLOT_L - 6, pos2.y + 4);
        }
    }

    function drawGridLines(c) {
        ctx.strokeStyle = c.border;
        ctx.lineWidth = 0.3;
        ctx.setLineDash([2, 4]);
        for (var v = 0.2; v <= 1; v += 0.2) {
            var h = dataToCanvas(0, v);
            ctx.beginPath();
            ctx.moveTo(PLOT_L, h.y);
            ctx.lineTo(PLOT_L + PLOT_W, h.y);
            ctx.stroke();
            var vv = dataToCanvas(v, 0);
            ctx.beginPath();
            ctx.moveTo(vv.x, PLOT_T);
            ctx.lineTo(vv.x, PLOT_T + PLOT_H);
            ctx.stroke();
        }
        ctx.setLineDash([]);
    }

    function drawVoronoi(c) {
        // Pixel-based Voronoi: for each pixel in the plot area, color by nearest centroid
        var resolution = 4; // sample every N pixels for performance
        ctx.save();
        ctx.beginPath();
        ctx.rect(PLOT_L, PLOT_T, PLOT_W, PLOT_H);
        ctx.clip();

        for (var px = PLOT_L; px < PLOT_L + PLOT_W; px += resolution) {
            for (var py = PLOT_T; py < PLOT_T + PLOT_H; py += resolution) {
                var d = canvasToData(px, py);
                var minDist = Infinity;
                var minK = 0;
                for (var k = 0; k < centroids.length; k++) {
                    var dist = euclideanDist(d, centroids[k]);
                    if (dist < minDist) {
                        minDist = dist;
                        minK = k;
                    }
                }
                ctx.fillStyle = c.clusterColors[minK] || c.muted;
                ctx.globalAlpha = c.voronoiOpacity;
                ctx.fillRect(px, py, resolution, resolution);
            }
        }

        ctx.globalAlpha = 1.0;
        ctx.restore();
    }

    function drawHistoryTrails(c) {
        ctx.globalAlpha = c.historyOpacity;
        for (var k = 0; k < K; k++) {
            ctx.strokeStyle = c.clusterColors[k] || c.muted;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            var started = false;
            for (var h = 0; h < centroidHistory.length; h++) {
                if (k < centroidHistory[h].length) {
                    var pt = dataToCanvas(centroidHistory[h][k].x, centroidHistory[h][k].y);
                    if (!started) {
                        ctx.moveTo(pt.x, pt.y);
                        started = true;
                    } else {
                        ctx.lineTo(pt.x, pt.y);
                    }
                }
            }
            ctx.stroke();

            // Draw dots at each historical position
            for (var h2 = 0; h2 < centroidHistory.length - 1; h2++) {
                if (k < centroidHistory[h2].length) {
                    var hp = dataToCanvas(centroidHistory[h2][k].x, centroidHistory[h2][k].y);
                    ctx.fillStyle = c.clusterColors[k] || c.muted;
                    ctx.beginPath();
                    ctx.arc(hp.x, hp.y, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        var setEl = function(id, val) {
            var el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setEl('metric-points', points.length);
        setEl('metric-k', K);
        setEl('metric-iteration', iteration > 0 ? iteration : '-');

        if (centroids.length > 0 && iteration > 0) {
            var inertia = computeInertia();
            setEl('metric-inertia', inertia.toFixed(4));
        } else {
            setEl('metric-inertia', '-');
        }

        setEl('metric-converged', converged ? 'Yes' : (iteration > 0 ? 'No' : '-'));
    }

    function setStatus(msg) {
        var el = document.getElementById('metric-status');
        if (el) el.textContent = msg;
        var ps = document.getElementById('playback-step');
        if (ps) ps.textContent = msg;
    }

    // ============================================
    // Playback controls
    // ============================================
    function startPlayback() {
        if (points.length < K) {
            setStatus('Need at least ' + K + ' points');
            return;
        }

        if (converged) {
            setStatus('Already converged. Reset to run again.');
            return;
        }

        if (centroids.length === 0) {
            initializeCentroids();
            // Do initial assignment for visual feedback
            assignStep();
            render();
            updateMetrics();
        }

        isRunning = true;
        document.getElementById('btn-play').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-step').disabled = true;
        setStatus('Running...');

        playNextIteration();
    }

    function playNextIteration() {
        if (!isRunning) return;

        var moreNeeded = doOneIteration();

        updateMetrics();
        render();

        if (moreNeeded) {
            setStatus('Iteration ' + iteration);
            var speed = parseInt(document.getElementById('speed-slider').value);
            var delay = 1050 - speed * 100;
            animTimer = setTimeout(playNextIteration, delay);
        } else {
            stopPlayback();
            if (converged) {
                setStatus('Converged at iteration ' + iteration);
            }
        }
    }

    function stopPlayback() {
        isRunning = false;
        if (animTimer) {
            clearTimeout(animTimer);
            animTimer = null;
        }
        document.getElementById('btn-play').disabled = false;
        document.getElementById('btn-pause').disabled = true;
        document.getElementById('btn-step').disabled = false;
    }

    function doStep() {
        if (points.length < K) {
            setStatus('Need at least ' + K + ' points');
            return;
        }

        if (converged) {
            setStatus('Already converged. Reset to run again.');
            return;
        }

        if (centroids.length === 0) {
            initializeCentroids();
            assignStep();
            setStatus('Initialized centroids');
            render();
            updateMetrics();
            return;
        }

        var moreNeeded = doOneIteration();
        updateMetrics();
        render();

        if (moreNeeded) {
            setStatus('Iteration ' + iteration);
        } else if (converged) {
            setStatus('Converged at iteration ' + iteration);
        }
    }

    function doReset() {
        stopPlayback();
        centroids = [];
        assignments = new Array(points.length).fill(-1);
        centroidHistory = [];
        iteration = 0;
        converged = false;
        algorithmPhase = 'idle';

        document.getElementById('btn-play').disabled = false;
        document.getElementById('btn-pause').disabled = true;
        document.getElementById('btn-step').disabled = false;

        setStatus('Ready');
        updateMetrics();
        render();
    }

    // ============================================
    // Event handlers
    // ============================================
    function onCanvasClick(e) {
        var rect = canvas.getBoundingClientRect();
        var cx = e.clientX - rect.left;
        var cy = e.clientY - rect.top;
        var d = canvasToData(cx, cy);

        if (d.x < DATA_MIN || d.x > DATA_MAX || d.y < DATA_MIN || d.y > DATA_MAX) return;

        if (editMode === 'add') {
            points.push({ x: d.x, y: d.y });
            assignments.push(-1);
        } else if (editMode === 'delete') {
            var closest = -1, minDist = 15;
            for (var i = 0; i < points.length; i++) {
                var cp = dataToCanvas(points[i].x, points[i].y);
                var dist = Math.hypot(cp.x - cx, cp.y - cy);
                if (dist < minDist) {
                    minDist = dist;
                    closest = i;
                }
            }
            if (closest >= 0) {
                points.splice(closest, 1);
                assignments.splice(closest, 1);
            }
        }

        updateMetrics();
        render();
    }

    // ============================================
    // Initialization
    // ============================================
    function init() {
        // Main canvas
        canvas = document.getElementById('clustering-canvas');
        if (!canvas) return;

        var setup = window.VizLib.CanvasUtils.setupHiDPICanvas(canvas);
        ctx = setup.ctx;
        dpr = setup.dpr;

        // Canvas click
        canvas.addEventListener('click', onCanvasClick);

        // Edit mode buttons
        document.querySelectorAll('.edit-mode-buttons .btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.edit-mode-buttons .btn').forEach(function(b) {
                    b.classList.remove('active');
                });
                this.classList.add('active');
                editMode = this.dataset.mode;
                canvas.style.cursor = editMode === 'delete' ? 'pointer' : 'crosshair';
            });
        });

        // Clear button
        document.getElementById('btn-clear-points').addEventListener('click', function() {
            points = [];
            assignments = [];
            doReset();
        });

        // Dataset select
        document.getElementById('dataset-select').addEventListener('change', function() {
            if (this.value !== 'custom') {
                generateDataset(this.value);
            } else {
                // Switch to custom mode, keep existing points
            }
            doReset();
        });

        // K stepper
        document.getElementById('k-minus').addEventListener('click', function() {
            if (K > 2) {
                K--;
                document.getElementById('k-value').value = K;
                doReset();
            }
        });
        document.getElementById('k-plus').addEventListener('click', function() {
            if (K < 8) {
                K++;
                document.getElementById('k-value').value = K;
                doReset();
            }
        });

        // Checkboxes
        document.getElementById('show-voronoi').addEventListener('change', function() {
            showVoronoi = this.checked;
            render();
        });
        document.getElementById('show-history').addEventListener('change', function() {
            showHistory = this.checked;
            render();
        });

        // Playback buttons
        document.getElementById('btn-play').addEventListener('click', startPlayback);
        document.getElementById('btn-pause').addEventListener('click', stopPlayback);
        document.getElementById('btn-step').addEventListener('click', doStep);
        document.getElementById('btn-reset').addEventListener('click', doReset);

        // Info tabs
        document.querySelectorAll('.info-panel-tabs .btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var tabId = this.dataset.tab;
                document.querySelectorAll('.info-panel-tabs .btn').forEach(function(b) {
                    b.classList.remove('active');
                });
                this.classList.add('active');
                document.querySelectorAll('.info-tab-content').forEach(function(c) {
                    c.classList.remove('active');
                });
                var content = document.getElementById('tab-' + tabId);
                if (content) content.classList.add('active');
            });
        });

        // Theme change listener
        document.addEventListener('themechange', function() {
            render();
        });

        // Initial render
        render();
        updateMetrics();
    }

    // Wait for VizLib
    window.addEventListener('vizlib-ready', init);
})();
