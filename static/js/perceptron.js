/**
 * Perceptron Learning Visualizer
 *
 * Interactive visualization of the perceptron learning rule.
 * Shows step-by-step updates, decision boundary, and convergence behavior.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_WIDTH = 560;
    const CANVAS_HEIGHT = 480;


    const POINT_RADIUS = 6;
    const HIGHLIGHT_RADIUS = 9;
    const DEFAULT_NUM_POINTS = 120;

    const STEP_PHASES = ['activation', 'prediction', 'update', 'redraw'];

    // ============================================
    // Helpers (clamp assigned on load after VizLib is available)
    // ============================================
    let clamp;

    const toSigned = (label) => (label === 1 ? 1 : -1);
    const toLabel = (sign) => (sign === 1 ? 1 : 0);

    const formatNumber = (value, decimals = 3) => {
        if (Number.isNaN(value) || !Number.isFinite(value)) return '-';
        return value.toFixed(decimals);
    };

    // ============================================
    // Dataset State
    // ============================================
    class DatasetState {
        constructor() {
            this.points = [];
            this.currentType = 'linear';
        }

        loadDataset(type, numPoints = DEFAULT_NUM_POINTS) {
            this.currentType = type;
            this.points = [];

            switch (type) {
                case 'linear':
                    this._generateLinear(numPoints);
                    break;
                case 'moons':
                    this._generateMoons(numPoints);
                    break;
                case 'xor':
                    this._generateXOR(numPoints);
                    break;
                case 'blobs':
                    this._generateBlobs(numPoints);
                    break;
                default:
                    this._generateLinear(numPoints);
            }

            return this.points;
        }

        _generateLinear(n, noise = 0.08) {
            const nPerClass = Math.floor(n / 2);
            for (let i = 0; i < nPerClass; i++) {
                const x0 = 0.1 + 0.8 * Math.random();
                const y0 = x0 + (Math.random() - 0.5) * noise;
                this.points.push({
                    x: clamp(x0 + (Math.random() - 0.5) * noise, 0.02, 0.98),
                    y: clamp(y0 + (Math.random() - 0.5) * noise, 0.02, 0.98),
                    classLabel: 1
                });
                const x1 = 0.1 + 0.8 * Math.random();
                const y1 = x1 - 0.4 + (Math.random() - 0.5) * noise;
                this.points.push({
                    x: clamp(x1 + (Math.random() - 0.5) * noise, 0.02, 0.98),
                    y: clamp(y1 + (Math.random() - 0.5) * noise, 0.02, 0.98),
                    classLabel: 0
                });
            }
        }

        _generateMoons(n, noise = 0.1) {
            const nPerClass = Math.floor(n / 2);
            for (let i = 0; i < nPerClass; i++) {
                const angle1 = Math.PI * i / nPerClass;
                const x1 = Math.cos(angle1) + this._noise(noise);
                const y1 = Math.sin(angle1) + this._noise(noise);
                this.points.push({
                    x: this._normalize(x1, -1.5, 2.5),
                    y: this._normalize(y1, -0.6, 1.6),
                    classLabel: 1
                });
                const x2 = 1 - Math.cos(angle1) + this._noise(noise);
                const y2 = 0.5 - Math.sin(angle1) + this._noise(noise);
                this.points.push({
                    x: this._normalize(x2, -1.5, 2.5),
                    y: this._normalize(y2, -0.6, 1.6),
                    classLabel: 0
                });
            }
        }

        _generateXOR(n, noise = 0.08) {
            const nPerQuadrant = Math.floor(n / 4);
            const centers = [
                { x: 0.25, y: 0.25, label: 0 },
                { x: 0.75, y: 0.75, label: 0 },
                { x: 0.25, y: 0.75, label: 1 },
                { x: 0.75, y: 0.25, label: 1 }
            ];

            centers.forEach(center => {
                for (let i = 0; i < nPerQuadrant; i++) {
                    this.points.push({
                        x: clamp(center.x + this._gaussian() * noise, 0.02, 0.98),
                        y: clamp(center.y + this._gaussian() * noise, 0.02, 0.98),
                        classLabel: center.label
                    });
                }
            });
        }

        _generateBlobs(n, noise = 0.08) {
            const centers = [
                { x: 0.3, y: 0.7, label: 0 },
                { x: 0.7, y: 0.3, label: 1 }
            ];
            const nPerBlob = Math.floor(n / centers.length);
            centers.forEach(center => {
                for (let i = 0; i < nPerBlob; i++) {
                    this.points.push({
                        x: clamp(center.x + this._gaussian() * noise, 0.02, 0.98),
                        y: clamp(center.y + this._gaussian() * noise, 0.02, 0.98),
                        classLabel: center.label
                    });
                }
            });
        }

        _gaussian() {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        }

        _noise(scale) {
            return (Math.random() - 0.5) * scale;
        }

        _normalize(value, min, max) {
            return (value - min) / (max - min);
        }
    }

    // ============================================
    // Perceptron Model
    // ============================================
    class PerceptronModel {
        constructor(learningRate = 0.005) {
            this.learningRate = learningRate;
            this.weights = [0, 0];
            this.bias = 0;
        }

        reset(initMode = 'zero') {
            if (initMode === 'random') {
                this.weights = [Math.random() * 2 - 1, Math.random() * 2 - 1];
                this.bias = Math.random() * 0.2 - 0.1;
            } else {
                this.weights = [0, 0];
                this.bias = 0;
            }
        }

        activation(x, y) {
            return this.weights[0] * x + this.weights[1] * y + this.bias;
        }

        predict(x, y) {
            const act = this.activation(x, y);
            const sign = act >= 0 ? 1 : -1;
            return { activation: act, sign, label: toLabel(sign) };
        }

        applyUpdate(x, y, targetSign, predictedSign) {
            const error = targetSign - predictedSign;
            if (error === 0) {
                return { updated: false, delta: [0, 0], deltaBias: 0 };
            }
            const deltaW0 = this.learningRate * error * x;
            const deltaW1 = this.learningRate * error * y;
            const deltaB = this.learningRate * error;

            this.weights[0] += deltaW0;
            this.weights[1] += deltaW1;
            this.bias += deltaB;

            return { updated: true, delta: [deltaW0, deltaW1], deltaBias: deltaB };
        }
    }

    // ============================================
    // Trainer
    // ============================================
    class Trainer {
        constructor(dataset, model, options) {
            this.dataset = dataset;
            this.model = model;
            this.epochs = options.epochs ?? 50;
            this.shuffle = options.shuffle ?? true;
        }

        generateSteps() {
            const steps = [];
            let globalStep = 0;
            const points = [...this.dataset];

            for (let epoch = 1; epoch <= this.epochs; epoch++) {
                const samples = this.shuffle ? this._shuffled(points) : [...points];
                let mistakes = 0;

                for (let index = 0; index < samples.length; index++) {
                    const point = samples[index];
                    const before = {
                        weights: [...this.model.weights],
                        bias: this.model.bias
                    };
                    const prediction = this.model.predict(point.x, point.y);
                    const targetSign = toSigned(point.classLabel);
                    const correct = prediction.sign === targetSign;

                    const activationStep = this._buildStep({
                        phase: 'activation',
                        epoch,
                        sampleIndex: index + 1,
                        totalSamples: samples.length,
                        globalStep: globalStep++,
                        point,
                        targetLabel: point.classLabel,
                        prediction,
                        correct,
                        mistakes,
                        weights: before.weights,
                        bias: before.bias
                    });

                    const predictionStep = this._buildStep({
                        phase: 'prediction',
                        epoch,
                        sampleIndex: index + 1,
                        totalSamples: samples.length,
                        globalStep: globalStep++,
                        point,
                        targetLabel: point.classLabel,
                        prediction,
                        correct,
                        mistakes,
                        weights: before.weights,
                        bias: before.bias
                    });

                    const updateResult = this.model.applyUpdate(point.x, point.y, targetSign, prediction.sign);
                    if (updateResult.updated) {
                        mistakes += 1;
                    }

                    const updateStep = this._buildStep({
                        phase: 'update',
                        epoch,
                        sampleIndex: index + 1,
                        totalSamples: samples.length,
                        globalStep: globalStep++,
                        point,
                        targetLabel: point.classLabel,
                        prediction,
                        correct,
                        mistakes,
                        weights: [...this.model.weights],
                        bias: this.model.bias,
                        updateResult
                    });

                    const accuracy = this._computeAccuracy(this.model.weights, this.model.bias, points);
                    const misclassifiedIndices = this._computeMisclassifiedIndices(this.model.weights, this.model.bias, points);
                    const redrawStep = this._buildStep({
                        phase: 'redraw',
                        epoch,
                        sampleIndex: index + 1,
                        totalSamples: samples.length,
                        globalStep: globalStep++,
                        point,
                        targetLabel: point.classLabel,
                        prediction,
                        correct,
                        mistakes,
                        weights: [...this.model.weights],
                        bias: this.model.bias,
                        accuracy,
                        misclassifiedIndices
                    });

                    // Share the misclassified set across all 4 phases of this sample
                    activationStep.misclassifiedIndices = misclassifiedIndices;
                    predictionStep.misclassifiedIndices = misclassifiedIndices;
                    updateStep.misclassifiedIndices = misclassifiedIndices;
                    redrawStep.misclassifiedIndices = misclassifiedIndices;

                    steps.push(activationStep, predictionStep, updateStep, redrawStep);
                }

                // Early stopping: converged if no mistakes this epoch
                if (mistakes === 0) break;
            }

            return steps;
        }

        _buildStep(payload) {
            const phaseIndex = STEP_PHASES.indexOf(payload.phase) + 1;
            return {
                phase: payload.phase,
                phaseIndex,
                epoch: payload.epoch,
                sampleIndex: payload.sampleIndex,
                totalSamples: payload.totalSamples,
                globalStep: payload.globalStep,
                point: payload.point,
                targetLabel: payload.targetLabel,
                prediction: payload.prediction,
                correct: payload.correct,
                mistakes: payload.mistakes,
                weights: payload.weights,
                bias: payload.bias,
                updateResult: payload.updateResult ?? null,
                accuracy: payload.accuracy ?? null,
                misclassifiedIndices: payload.misclassifiedIndices ?? null
            };
        }

        _computeMisclassifiedIndices(weights, bias, points) {
            const set = new Set();
            points.forEach((point, idx) => {
                const activation = weights[0] * point.x + weights[1] * point.y + bias;
                const sign = activation >= 0 ? 1 : -1;
                if (toLabel(sign) !== point.classLabel) {
                    set.add(idx);
                }
            });
            return set;
        }

        _computeAccuracy(weights, bias, points) {
            let correct = 0;
            points.forEach(point => {
                const activation = weights[0] * point.x + weights[1] * point.y + bias;
                const sign = activation >= 0 ? 1 : -1;
                const label = toLabel(sign);
                if (label === point.classLabel) {
                    correct += 1;
                }
            });
            return correct / points.length;
        }

        _shuffled(points) {
            const array = [...points];
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
            return array;
        }
    }

    // ============================================
    // UI Renderer
    // ============================================
    class UIRenderer {
        constructor() {
            this.canvas = document.getElementById('perceptron-canvas');
            this.ctx = this.canvas?.getContext('2d');
            this.displayScale = 1;
            this.classColors = ['#e41a1c', '#377eb8'];
            this._updateColors();
            this._setupHiDPI();
        }

        _setupHiDPI() {
            if (!this.canvas || !this.ctx) return;

            const wrapper = this.canvas.parentElement;
            const displayWidth = wrapper.clientWidth;
            const displayHeight = Math.round(displayWidth * (CANVAS_HEIGHT / CANVAS_WIDTH));
            const dpr = window.devicePixelRatio || 1;
            const scale = displayWidth / CANVAS_WIDTH;

            this.canvas.width = displayWidth * dpr;
            this.canvas.height = displayHeight * dpr;
            this.canvas.style.width = displayWidth + 'px';
            this.canvas.style.height = displayHeight + 'px';

            this.ctx.setTransform(scale * dpr, 0, 0, scale * dpr, 0, 0);
            this.displayScale = scale;
        }

        _updateColors() {
            if (window.VizLib?.ThemeManager) {
                const colors = window.VizLib.ThemeManager.getColors('categorical');
                this.classColors = [colors[0], colors[1]];
            }
            const style = getComputedStyle(document.documentElement);
            this.boundaryColor = style.getPropertyValue('--perceptron-boundary').trim() || '#111827';
            this.vectorColor = style.getPropertyValue('--perceptron-vector').trim() || '#0f766e';
            this.marginColor = style.getPropertyValue('--perceptron-margin').trim() || 'rgba(17,24,39,0.25)';
            this.misclassifiedColor = style.getPropertyValue('--perceptron-misclassified').trim() || '#f59e0b';
            this.highlightColor = style.getPropertyValue('--perceptron-highlight').trim() || '#facc15';
            this.gridColor = style.getPropertyValue('--viz-border').trim() || '#dee2e6';
            this.textColor = style.getPropertyValue('--viz-text').trim() || '#333333';
        }

        _resetTransform() {
            if (!this.ctx) return;
            const dpr = window.devicePixelRatio || 1;
            this.ctx.setTransform(this.displayScale * dpr, 0, 0, this.displayScale * dpr, 0, 0);
        }

        dataToCanvas(x, y) {
            return {
                x: x * CANVAS_WIDTH,
                y: (1 - y) * CANVAS_HEIGHT
            };
        }

        render(state) {
            if (!this.ctx) return;
            this._updateColors();
            this._resetTransform();

            this.ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

            if (state.showBoundary) {
                this._drawRegions(state.weights, state.bias);
            }
            this._drawGrid();

            if (state.showBoundary) {
                this._drawBoundary(state.weights, state.bias);
            }
            if (state.showMargin) {
                this._drawMargin(state.weights, state.bias);
            }
            if (state.showVector) {
                this._drawWeightVector(state.weights);
            }

            this._drawPoints(state.points, state.misclassifiedSet);

            if (state.currentPoint) {
                this._drawHighlight(state.currentPoint);
            }
        }

        _drawGrid() {
            const ctx = this.ctx;
            ctx.save();
            ctx.strokeStyle = this.gridColor;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.4;

            const steps = 10;
            for (let i = 1; i < steps; i++) {
                const x = (CANVAS_WIDTH / steps) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, CANVAS_HEIGHT);
                ctx.stroke();

                const y = (CANVAS_HEIGHT / steps) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(CANVAS_WIDTH, y);
                ctx.stroke();
            }

            ctx.restore();
        }

        _drawPoints(points, misclassifiedSet) {
            const ctx = this.ctx;
            points.forEach((point, idx) => {
                const pos = this.dataToCanvas(point.x, point.y);
                const isMisclassified = misclassifiedSet?.has(idx);
                ctx.beginPath();
                ctx.fillStyle = this.classColors[point.classLabel];
                ctx.arc(pos.x, pos.y, POINT_RADIUS, 0, Math.PI * 2);
                ctx.fill();

                if (isMisclassified) {
                    const s = POINT_RADIUS * 0.7;
                    ctx.strokeStyle = this.misclassifiedColor;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(pos.x - s, pos.y - s);
                    ctx.lineTo(pos.x + s, pos.y + s);
                    ctx.moveTo(pos.x + s, pos.y - s);
                    ctx.lineTo(pos.x - s, pos.y + s);
                    ctx.stroke();
                }
            });
        }

        _drawHighlight(point) {
            const ctx = this.ctx;
            const pos = this.dataToCanvas(point.x, point.y);
            ctx.beginPath();
            ctx.fillStyle = this.highlightColor;
            ctx.globalAlpha = 0.35;
            ctx.arc(pos.x, pos.y, HIGHLIGHT_RADIUS, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        _drawRegions(weights, bias) {
            const [w1, w2] = weights;
            if (Math.abs(w1) < 1e-6 && Math.abs(w2) < 1e-6) return;

            const ctx = this.ctx;
            ctx.save();
            ctx.globalAlpha = 0.08;

            // For each pixel column, compute where boundary crosses and fill above/below
            // Positive side (w·x + b > 0) = class 1, negative = class 0
            const steps = 40;
            const dx = CANVAS_WIDTH / steps;
            for (let i = 0; i < steps; i++) {
                const cx = i * dx;
                const cx2 = (i + 1) * dx;
                // Convert canvas x back to data x
                const dataX1 = cx / CANVAS_WIDTH;
                const dataX2 = cx2 / CANVAS_WIDTH;

                // Boundary y in data space: w1*x + w2*y + b = 0 => y = (-b - w1*x) / w2
                // Fill from top of canvas to boundary = one class, boundary to bottom = other
                // In canvas coords: top = data y=1, bottom = data y=0

                const dataX = (dataX1 + dataX2) / 2;

                if (Math.abs(w2) > 1e-9) {
                    const boundaryDataY = (-bias - w1 * dataX) / w2;
                    const boundaryCanvasY = (1 - boundaryDataY) * CANVAS_HEIGHT;
                    const clampedY = Math.max(0, Math.min(CANVAS_HEIGHT, boundaryCanvasY));

                    // w2 > 0: positive side (class 1) is above boundary in data space = top in canvas
                    // w2 < 0: positive side is below boundary in data space = bottom in canvas
                    const topClass = w2 > 0 ? 1 : 0;
                    const botClass = 1 - topClass;

                    ctx.fillStyle = this.classColors[topClass];
                    ctx.fillRect(cx, 0, dx, clampedY);

                    ctx.fillStyle = this.classColors[botClass];
                    ctx.fillRect(cx, clampedY, dx, CANVAS_HEIGHT);
                } else {
                    // Vertical line in data space — entire column is one class
                    const sign = w1 * dataX + bias;
                    ctx.fillStyle = this.classColors[sign >= 0 ? 1 : 0];
                    ctx.fillRect(cx, 0, dx, CANVAS_HEIGHT);
                }
            }

            ctx.restore();
        }

        _drawBoundary(weights, bias) {
            const line = this._lineForWeights(weights, bias);
            if (!line) return;

            const ctx = this.ctx;
            ctx.save();
            ctx.strokeStyle = this.boundaryColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(line.x1, line.y1);
            ctx.lineTo(line.x2, line.y2);
            ctx.stroke();
            ctx.restore();
        }

        _drawMargin(weights, bias) {
            const margin = this._marginLines(weights, bias);
            if (!margin) return;

            const ctx = this.ctx;
            ctx.save();
            ctx.strokeStyle = this.marginColor;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            margin.forEach(line => {
                ctx.beginPath();
                ctx.moveTo(line.x1, line.y1);
                ctx.lineTo(line.x2, line.y2);
                ctx.stroke();
            });
            ctx.restore();
        }

        _drawWeightVector(weights) {
            const ctx = this.ctx;
            const center = this.dataToCanvas(0.5, 0.5);
            const magnitude = Math.sqrt(weights[0] ** 2 + weights[1] ** 2) || 1;
            const nx = weights[0] / magnitude;
            const ny = weights[1] / magnitude;
            const scale = 0.25;
            const end = this.dataToCanvas(0.5 + nx * scale, 0.5 + ny * scale);

            ctx.save();
            ctx.strokeStyle = this.vectorColor;
            ctx.fillStyle = this.vectorColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(center.x, center.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();

            const angle = Math.atan2(end.y - center.y, end.x - center.x);
            const arrowSize = 8;
            ctx.beginPath();
            ctx.moveTo(end.x, end.y);
            ctx.lineTo(
                end.x - arrowSize * Math.cos(angle - Math.PI / 6),
                end.y - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            ctx.lineTo(
                end.x - arrowSize * Math.cos(angle + Math.PI / 6),
                end.y - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }

        _lineForWeights(weights, bias) {
            const [w1, w2] = weights;
            if (Math.abs(w1) < 1e-6 && Math.abs(w2) < 1e-6) return null;

            // Compute two points far apart on the line w1*x + w2*y + b = 0.
            // Canvas clipping handles the rest — no manual intersection needed.
            let p1, p2;
            if (Math.abs(w2) > Math.abs(w1)) {
                // Line is more horizontal — parameterize by x
                p1 = { x: -10, y: (-bias - w1 * -10) / w2 };
                p2 = { x:  10, y: (-bias - w1 *  10) / w2 };
            } else {
                // Line is more vertical — parameterize by y
                p1 = { x: (-bias - w2 * -10) / w1, y: -10 };
                p2 = { x: (-bias - w2 *  10) / w1, y:  10 };
            }

            const c1 = this.dataToCanvas(p1.x, p1.y);
            const c2 = this.dataToCanvas(p2.x, p2.y);
            return { x1: c1.x, y1: c1.y, x2: c2.x, y2: c2.y };
        }

        _marginLines(weights, bias) {
            const magnitude = Math.sqrt(weights[0] ** 2 + weights[1] ** 2);
            if (magnitude < 1e-6) return null;

            const marginOffset = 1 / magnitude;
            const upper = this._lineForWeights(weights, bias + marginOffset);
            const lower = this._lineForWeights(weights, bias - marginOffset);
            if (!upper || !lower) return null;
            return [upper, lower];
        }
    }

    // ============================================
    // Perceptron Visualizer Controller
    // ============================================
    class PerceptronViz {
        constructor() {
            this.dataset = new DatasetState();
            this.model = new PerceptronModel();
            this.ui = new UIRenderer();
            this.playback = null;
            this.steps = [];

            this.state = {
                showBoundary: true,
                showVector: true,
                showMargin: false
            };

            this._init();
        }

        _init() {
            if (window.VizLib?.PlaybackController) {
                this._setupPlayback();
                this._setupEventListeners();
                this._resetTraining();
            } else {
                window.addEventListener('vizlib-ready', () => {
                    this._setupPlayback();
                    this._setupEventListeners();
                    this._resetTraining();
                });
            }

            if (window.VizLib?.ThemeManager) {
                window.VizLib.ThemeManager.onThemeChange(() => {
                    this._renderCurrent();
                });
            }

            window.addEventListener('resize', () => {
                this.ui._setupHiDPI();
                this._renderCurrent();
            });
        }

        _setupPlayback() {
            const PlaybackController = window.VizLib?.PlaybackController;
            if (!PlaybackController) return;

            this.playback = new PlaybackController({
                initialSpeed: 6,
                getDelayFn: (speed) => {
                    // Speed 1 = 500ms, Speed 6 = 30ms, Speed 10 = 1ms
                    const delays = [500, 300, 150, 80, 50, 30, 15, 8, 4, 1];
                    return delays[Math.min(speed - 1, delays.length - 1)] || 30;
                },
                onRenderStep: (step) => this._onRenderStep(step),
                onPlayStateChange: (isPlaying) => this._onPlayStateChange(isPlaying),
                onStepChange: (index, total) => this._onStepChange(index, total),
                onFinished: () => this._onPlaybackFinished(),
                onReset: () => this._onPlaybackReset()
            });
        }

        _setupEventListeners() {
            document.getElementById('dataset-select')?.addEventListener('change', () => this._resetTraining());
            document.getElementById('init-select')?.addEventListener('change', () => this._resetTraining());
            document.getElementById('shuffle-toggle')?.addEventListener('change', () => this._resetTraining());

            const lrSlider = document.getElementById('lr-slider');
            const lrValue = document.getElementById('lr-value');
            lrSlider?.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (lrValue) lrValue.textContent = value.toFixed(2);
                this._resetTraining();
            });

            VizLib.DomUtils.wireStepper('epochs-minus', 'epochs-plus', 'epochs-value', {
                min: 1, max: 50, step: 1, onChange: () => this._resetTraining()
            });

            document.getElementById('show-boundary')?.addEventListener('change', (e) => {
                this.state.showBoundary = e.target.checked;
                this._renderCurrent();
            });
            document.getElementById('show-vector')?.addEventListener('change', (e) => {
                this.state.showVector = e.target.checked;
                this._renderCurrent();
            });
            document.getElementById('show-margin')?.addEventListener('change', (e) => {
                this.state.showMargin = e.target.checked;
                this._renderCurrent();
            });

            document.getElementById('btn-play')?.addEventListener('click', () => this.playback?.play());
            document.getElementById('btn-pause')?.addEventListener('click', () => this.playback?.pause());
            document.getElementById('btn-step-forward')?.addEventListener('click', () => this.playback?.stepForward());
            document.getElementById('btn-step-back')?.addEventListener('click', () => this.playback?.stepBackward());
            document.getElementById('btn-reset')?.addEventListener('click', () => this._resetTraining());

            document.getElementById('speed-slider')?.addEventListener('input', (e) => {
                const value = parseInt(e.target.value, 10);
                this.playback?.setSpeed(value);
            });

            // Info tab button switching (for btn-group variant)
            document.querySelectorAll('.info-panel-tabs .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tabId = btn.getAttribute('data-tab');
                    btn.closest('.info-panel-tabs').querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    const panel = btn.closest('.panel');
                    panel.querySelectorAll('.info-tab-content').forEach(t => t.classList.remove('active'));
                    const target = panel.querySelector('#tab-' + tabId);
                    if (target) target.classList.add('active');
                });
            });
        }

        _resetTraining() {
            if (this.playback) {
                this.playback.pause();
            }

            const datasetType = document.getElementById('dataset-select')?.value || 'linear';
            const initMode = document.getElementById('init-select')?.value || 'zero';
            const lr = parseFloat(document.getElementById('lr-slider')?.value || '0.005');
            const epochs = parseInt(document.getElementById('epochs-value')?.value || '50', 10);
            const shuffle = Boolean(document.getElementById('shuffle-toggle')?.checked);

            const points = this.dataset.loadDataset(datasetType, DEFAULT_NUM_POINTS);
            this.model.learningRate = lr;
            this.model.reset(initMode);

            const trainingModel = new PerceptronModel(lr);
            trainingModel.weights = [...this.model.weights];
            trainingModel.bias = this.model.bias;
            const trainer = new Trainer(points, trainingModel, { epochs, shuffle });
            this.steps = trainer.generateSteps();
            this.epochs = epochs;

            if (this.playback) {
                this.playback.load(this.steps);
            }

            this._renderCurrent();
            this._updatePlaybackButtons();
            this._updateMetrics(null);
            this._updateStepList(null);
            const nonSeparable = datasetType === 'moons' || datasetType === 'xor';
            this._setStatus(nonSeparable ? 'Ready (non-separable data)' : 'Ready');
            this._onStepChange(-1, this.steps.length);
        }

        _renderCurrent() {
            const currentStep = this.playback?.currentStepIndex >= 0
                ? this.steps[this.playback.currentStepIndex]
                : null;

            const weights = currentStep?.weights ?? this.model.weights;
            const bias = currentStep?.bias ?? this.model.bias;

            const misclassifiedSet = this._computeMisclassified(weights, bias);

            this.ui.render({
                points: this.dataset.points,
                weights,
                bias,
                currentPoint: currentStep?.point || null,
                misclassifiedSet,
                showBoundary: this.state.showBoundary,
                showVector: this.state.showVector,
                showMargin: this.state.showMargin
            });
        }

        _computeMisclassified(weights, bias) {
            const set = new Set();
            this.dataset.points.forEach((point, idx) => {
                const activation = weights[0] * point.x + weights[1] * point.y + bias;
                const sign = activation >= 0 ? 1 : -1;
                const label = toLabel(sign);
                if (label !== point.classLabel) {
                    set.add(idx);
                }
            });
            return set;
        }

        _onRenderStep(step) {
            if (!step) return;

            this.ui.render({
                points: this.dataset.points,
                weights: step.weights,
                bias: step.bias,
                currentPoint: step.point,
                misclassifiedSet: step.misclassifiedIndices,
                showBoundary: this.state.showBoundary,
                showVector: this.state.showVector,
                showMargin: this.state.showMargin
            });

            this._updateMetrics(step);
            this._updateStepList(step.phase);
        }

        _onPlayStateChange(isPlaying) {
            const playBtn = document.getElementById('btn-play');
            const pauseBtn = document.getElementById('btn-pause');
            if (playBtn) playBtn.disabled = isPlaying;
            if (pauseBtn) pauseBtn.disabled = !isPlaying;
            this._updatePlaybackButtons();
        }

        _onStepChange(index, total) {
            const display = document.getElementById('playback-step');
            if (display) {
                const step = this.steps[index];
                if (step) {
                    display.textContent = `Epoch ${step.epoch} / ${this.epochs} · Sample ${step.sampleIndex} / ${step.totalSamples}`;
                } else {
                    display.textContent = `Epoch 0 / ${this.epochs}`;
                }
            }
            this._updatePlaybackButtons();
        }

        _onPlaybackFinished() {
            const lastStep = this.steps[this.steps.length - 1];
            const converged = lastStep && lastStep.epoch < this.epochs;
            this._setStatus(converged ? 'Converged!' : 'Finished');
            this._updatePlaybackButtons();
        }

        _onPlaybackReset() {
            this._setStatus('Ready');
            this._updatePlaybackButtons();
            this._updateStepList(null);
            this._updateMetrics(null);
            this._renderCurrent();
        }

        _updatePlaybackButtons() {
            const hasSteps = this.steps.length > 0;
            const isPlaying = this.playback?.isPlaying;
            const atStart = (this.playback?.currentStepIndex ?? -1) <= 0;
            const atEnd = (this.playback?.currentStepIndex ?? -1) >= this.steps.length - 1;

            const stepForward = document.getElementById('btn-step-forward');
            const stepBack = document.getElementById('btn-step-back');
            const playBtn = document.getElementById('btn-play');

            if (stepForward) stepForward.disabled = !hasSteps || atEnd || isPlaying;
            if (stepBack) stepBack.disabled = !hasSteps || atStart || isPlaying;
            if (playBtn) playBtn.disabled = !hasSteps || isPlaying;
        }

        _updateMetrics(step) {
            const epochEl = document.getElementById('metric-epoch');
            const stepEl = document.getElementById('metric-step');
            const mistakesEl = document.getElementById('metric-mistakes');
            const accuracyEl = document.getElementById('metric-accuracy');
            const weightsEl = document.getElementById('metric-weights');
            const biasEl = document.getElementById('metric-bias');

            const sampleEl = document.getElementById('metric-sample');
            const activationEl = document.getElementById('metric-activation');
            const predictionEl = document.getElementById('metric-prediction');
            const targetEl = document.getElementById('metric-target');

            if (!step) {
                if (epochEl) epochEl.textContent = '0';
                if (stepEl) stepEl.textContent = '0';
                if (mistakesEl) mistakesEl.textContent = '0';
                if (accuracyEl) accuracyEl.textContent = '0%';
                if (weightsEl) weightsEl.textContent = `[${formatNumber(this.model.weights[0])}, ${formatNumber(this.model.weights[1])}]`;
                if (biasEl) biasEl.textContent = formatNumber(this.model.bias);
                if (sampleEl) sampleEl.textContent = '-';
                if (activationEl) activationEl.textContent = '-';
                if (predictionEl) predictionEl.textContent = '-';
                if (targetEl) targetEl.textContent = '-';
                return;
            }

            if (epochEl) epochEl.textContent = String(step.epoch);
            if (stepEl) stepEl.textContent = `${step.sampleIndex} / ${step.totalSamples}`;
            if (mistakesEl) mistakesEl.textContent = String(step.mistakes);

            if (weightsEl) {
                weightsEl.textContent = `[${formatNumber(step.weights[0])}, ${formatNumber(step.weights[1])}]`;
            }
            if (biasEl) biasEl.textContent = formatNumber(step.bias);

            if (step.accuracy !== null) {
                if (accuracyEl) accuracyEl.textContent = `${Math.round(step.accuracy * 100)}%`;
            }

            if (sampleEl) {
                sampleEl.textContent = `(${formatNumber(step.point.x, 2)}, ${formatNumber(step.point.y, 2)})`;
            }
            if (activationEl) activationEl.textContent = formatNumber(step.prediction.activation);
            if (predictionEl) predictionEl.textContent = String(step.prediction.label);
            if (targetEl) targetEl.textContent = String(step.targetLabel);

            if (step.phase === 'update') {
                const status = step.updateResult?.updated ? 'Update applied' : 'No update (correct)';
                this._setStatus(status);
            }
        }

        _updateStepList(phase) {
            const ids = {
                activation: 'step-activation',
                prediction: 'step-prediction',
                update: 'step-update',
                redraw: 'step-redraw'
            };

            Object.values(ids).forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });

            if (phase && ids[phase]) {
                const activeEl = document.getElementById(ids[phase]);
                if (activeEl) activeEl.classList.add('active');
            }
        }

        _setStatus(text) {
            const statusEl = document.getElementById('metric-status');
            if (statusEl) statusEl.textContent = text;
        }
    }

    window.addEventListener('load', () => {
        clamp = VizLib.MathUtils.clamp;
        new PerceptronViz();
    });
})();
