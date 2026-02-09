/**
 * RNN & LSTM Visualizer
 *
 * Interactive visualization of recurrent neural networks processing sequences
 * token by token. Supports vanilla RNN and LSTM architectures with gate
 * visualization, hidden state heatmaps, and step-through animation.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const NETWORK_W = 560;
    const NETWORK_H = 300;
    const HEATMAP_W = 560;
    const HEATMAP_H = 160;
    const GATE_W = 150;
    const GATE_H = 80;
    const HIDDEN_DIM = 8;
    const MIN_SEQ_LEN = 4;
    const MAX_SEQ_LEN = 12;

    // ============================================
    // Utility helpers (clamp assigned in init() after VizLib is available)
    // ============================================
    let clamp;

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-clamp(x, -500, 500)));
    }

    function tanhFn(x) {
        return Math.tanh(x);
    }

    function vecNorm(v) {
        let s = 0;
        for (let i = 0; i < v.length; i++) s += v[i] * v[i];
        return Math.sqrt(s);
    }

    // Small-scale random init for weight matrices
    function randomMatrix(rows, cols, scale) {
        const m = [];
        for (let r = 0; r < rows; r++) {
            const row = [];
            for (let c = 0; c < cols; c++) {
                row.push((Math.random() * 2 - 1) * scale);
            }
            m.push(row);
        }
        return m;
    }

    function randomVector(n, scale) {
        const v = [];
        for (let i = 0; i < n; i++) v.push((Math.random() * 2 - 1) * scale);
        return v;
    }

    function zeroVector(n) {
        return new Array(n).fill(0);
    }

    // Matrix-vector multiply: out = M * v
    function matvec(M, v) {
        const out = [];
        for (let r = 0; r < M.length; r++) {
            let s = 0;
            for (let c = 0; c < v.length; c++) {
                s += M[r][c] * v[c];
            }
            out.push(s);
        }
        return out;
    }

    // Element-wise add
    function vecAdd(a, b) {
        return a.map((v, i) => v + b[i]);
    }

    // Element-wise multiply (Hadamard)
    function vecMul(a, b) {
        return a.map((v, i) => v * b[i]);
    }

    // Apply scalar function element-wise
    function vecApply(v, fn) {
        return v.map(fn);
    }

    // ============================================
    // Theme-aware color helper
    // ============================================
    function getThemeColors() {
        const style = getComputedStyle(document.documentElement);
        return {
            cellBg:         style.getPropertyValue('--rnn-cell-bg').trim(),
            cellBorder:     style.getPropertyValue('--rnn-cell-border').trim(),
            cellActiveBg:   style.getPropertyValue('--rnn-cell-active-bg').trim(),
            cellActiveBorder: style.getPropertyValue('--rnn-cell-active-border').trim(),
            inputColor:     style.getPropertyValue('--rnn-input-color').trim(),
            hiddenColor:    style.getPropertyValue('--rnn-hidden-color').trim(),
            outputColor:    style.getPropertyValue('--rnn-output-color').trim(),
            arrowColor:     style.getPropertyValue('--rnn-arrow-color').trim(),
            gateForget:     style.getPropertyValue('--rnn-gate-forget').trim(),
            gateInput:      style.getPropertyValue('--rnn-gate-input').trim(),
            gateOutput:     style.getPropertyValue('--rnn-gate-output').trim(),
            gateBarBg:      style.getPropertyValue('--rnn-gate-bar-bg').trim(),
            heatmapNeg:     style.getPropertyValue('--rnn-heatmap-neg').trim(),
            heatmapZero:    style.getPropertyValue('--rnn-heatmap-zero').trim(),
            heatmapPos:     style.getPropertyValue('--rnn-heatmap-pos').trim(),
            heatmapBorder:  style.getPropertyValue('--rnn-heatmap-border').trim(),
            labelColor:     style.getPropertyValue('--rnn-label-color').trim(),
            cellText:       style.getPropertyValue('--rnn-cell-text').trim(),
            canvasGrid:     style.getPropertyValue('--rnn-canvas-grid').trim(),
            canvasBg:       style.getPropertyValue('--viz-canvas-bg').trim(),
            textMuted:      style.getPropertyValue('--viz-text-muted').trim()
        };
    }

    // Parse hex color to [r,g,b]
    function hexToRgb(hex) {
        hex = hex.replace('#', '');
        if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
        const n = parseInt(hex, 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }

    // Lerp between two [r,g,b] colors
    function lerpColor(c1, c2, t) {
        return [
            Math.round(c1[0] + (c2[0] - c1[0]) * t),
            Math.round(c1[1] + (c2[1] - c1[1]) * t),
            Math.round(c1[2] + (c2[2] - c1[2]) * t)
        ];
    }

    function rgbStr(rgb) {
        return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }

    // ============================================
    // Sequence Generators
    // ============================================
    function generateSequence(type, length) {
        const seq = [];
        switch (type) {
            case 'counting':
                for (let i = 0; i < length; i++) seq.push((i + 1) / length);
                break;
            case 'sine':
                for (let i = 0; i < length; i++) seq.push(Math.sin(2 * Math.PI * i / length));
                break;
            case 'custom':
                // Will be parsed from input field
                return null;
            default:
                for (let i = 0; i < length; i++) seq.push((i + 1) / length);
        }
        return seq;
    }

    // ============================================
    // Vanilla RNN Model
    // ============================================
    class VanillaRNN {
        constructor(inputDim, hiddenDim) {
            this.hiddenDim = hiddenDim;
            this.Wh = randomMatrix(hiddenDim, hiddenDim, 0.3);
            this.Wx = randomMatrix(hiddenDim, inputDim, 0.5);
            this.b = randomVector(hiddenDim, 0.1);
        }

        step(x, hPrev) {
            // h_t = tanh(Wh * h_{t-1} + Wx * x_t + b)
            const whh = matvec(this.Wh, hPrev);
            const wxx = matvec(this.Wx, Array.isArray(x) ? x : [x]);
            const pre = vecAdd(vecAdd(whh, wxx), this.b);
            const h = vecApply(pre, tanhFn);
            return { h, gates: null };
        }
    }

    // ============================================
    // LSTM Model
    // ============================================
    class LSTMModel {
        constructor(inputDim, hiddenDim) {
            this.hiddenDim = hiddenDim;
            const concatDim = hiddenDim + inputDim;
            const s = 0.3;
            // Forget gate
            this.Wf = randomMatrix(hiddenDim, concatDim, s);
            this.bf = randomVector(hiddenDim, 0.1);
            // Input gate
            this.Wi = randomMatrix(hiddenDim, concatDim, s);
            this.bi = randomVector(hiddenDim, 0.1);
            // Output gate
            this.Wo = randomMatrix(hiddenDim, concatDim, s);
            this.bo = randomVector(hiddenDim, 0.1);
            // Cell candidate
            this.Wc = randomMatrix(hiddenDim, concatDim, s);
            this.bc = randomVector(hiddenDim, 0.1);
        }

        step(x, hPrev, cPrev) {
            const xArr = Array.isArray(x) ? x : [x];
            const concat = hPrev.concat(xArr);

            const ft = vecApply(vecAdd(matvec(this.Wf, concat), this.bf), sigmoid);
            const it = vecApply(vecAdd(matvec(this.Wi, concat), this.bi), sigmoid);
            const ot = vecApply(vecAdd(matvec(this.Wo, concat), this.bo), sigmoid);
            const cHat = vecApply(vecAdd(matvec(this.Wc, concat), this.bc), tanhFn);

            const c = vecAdd(vecMul(ft, cPrev), vecMul(it, cHat));
            const h = vecMul(ot, vecApply(c, tanhFn));

            return { h, c, gates: { forget: ft, input: it, output: ot } };
        }
    }

    // ============================================
    // Main Visualizer Class
    // ============================================
    class RNNVisualizer {
        constructor() {
            // Canvases
            this.networkCanvas = document.getElementById('network-canvas');
            this.heatmapCanvas = document.getElementById('heatmap-canvas');
            this.forgetGateCanvas = document.getElementById('forget-gate-canvas');
            this.inputGateCanvas = document.getElementById('input-gate-canvas');
            this.outputGateCanvas = document.getElementById('output-gate-canvas');

            this.networkCtx = this.networkCanvas.getContext('2d');
            this.heatmapCtx = this.heatmapCanvas.getContext('2d');
            this.forgetGateCtx = this.forgetGateCanvas.getContext('2d');
            this.inputGateCtx = this.inputGateCanvas.getContext('2d');
            this.outputGateCtx = this.outputGateCanvas.getContext('2d');

            // Setup HiDPI
            if (window.VizLib && window.VizLib.setupHiDPICanvas) {
                window.VizLib.setupHiDPICanvas(this.networkCanvas, NETWORK_W, NETWORK_H);
                window.VizLib.setupHiDPICanvas(this.heatmapCanvas, HEATMAP_W, HEATMAP_H);
                window.VizLib.setupHiDPICanvas(this.forgetGateCanvas, GATE_W, GATE_H);
                window.VizLib.setupHiDPICanvas(this.inputGateCanvas, GATE_W, GATE_H);
                window.VizLib.setupHiDPICanvas(this.outputGateCanvas, GATE_W, GATE_H);
            }

            // State
            this.architecture = 'rnn';
            this.seqType = 'counting';
            this.seqLength = 6;
            this.sequence = [];
            this.currentStep = 0;
            this.isProcessing = false;
            this.animTimer = null;
            this.speed = 5;

            // Model state
            this.model = null;
            this.hiddenStates = [];   // array of h vectors, one per completed step
            this.cellStates = [];     // LSTM only
            this.gateHistory = [];    // LSTM gate values per step
            this.currentH = null;
            this.currentC = null;

            // DOM
            this.archSelect = document.getElementById('arch-select');
            this.seqSelect = document.getElementById('sequence-select');
            this.customSeqRow = document.getElementById('custom-seq-row');
            this.customSeqInput = document.getElementById('custom-seq-input');
            this.seqLenVal = document.getElementById('seq-len-val');
            this.btnProcess = document.getElementById('btn-process');
            this.btnStep = document.getElementById('btn-step');
            this.btnReset = document.getElementById('btn-reset');
            this.speedSlider = document.getElementById('speed-slider');
            this.speedValue = document.getElementById('speed-value');
            this.archBadge = document.getElementById('arch-badge');
            this.heatmapBadge = document.getElementById('heatmap-badge');
            this.gatePanel = document.getElementById('gate-panel');

            this.bindEvents();
            this.reset();
        }

        // ============================================
        // Event Binding
        // ============================================
        bindEvents() {
            this.archSelect.addEventListener('change', () => {
                this.architecture = this.archSelect.value;
                this.archBadge.textContent = this.architecture === 'lstm' ? 'LSTM' : 'Vanilla RNN';
                this.gatePanel.style.display = this.architecture === 'lstm' ? '' : 'none';
                this.reset();
            });

            this.seqSelect.addEventListener('change', () => {
                this.seqType = this.seqSelect.value;
                this.customSeqRow.style.display = this.seqType === 'custom' ? '' : 'none';
                this.reset();
            });

            document.getElementById('seq-len-dec').addEventListener('click', () => {
                if (this.seqLength > MIN_SEQ_LEN) {
                    this.seqLength--;
                    this.seqLenVal.value = this.seqLength;
                    this.reset();
                }
            });

            document.getElementById('seq-len-inc').addEventListener('click', () => {
                if (this.seqLength < MAX_SEQ_LEN) {
                    this.seqLength++;
                    this.seqLenVal.value = this.seqLength;
                    this.reset();
                }
            });

            this.btnProcess.addEventListener('click', () => this.processAll());
            this.btnStep.addEventListener('click', () => this.stepOnce());
            this.btnReset.addEventListener('click', () => this.reset());

            this.speedSlider.addEventListener('input', () => {
                this.speed = parseInt(this.speedSlider.value);
                this.speedValue.textContent = this.speed;
            });

            // Theme change
            document.addEventListener('themechange', () => this.drawAll());
        }

        // ============================================
        // Reset
        // ============================================
        reset() {
            this.stopAnimation();
            this.currentStep = 0;
            this.hiddenStates = [];
            this.cellStates = [];
            this.gateHistory = [];

            // Build sequence
            if (this.seqType === 'custom') {
                const raw = this.customSeqInput.value.trim();
                if (raw) {
                    this.sequence = raw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
                    this.seqLength = clamp(this.sequence.length, MIN_SEQ_LEN, MAX_SEQ_LEN);
                    this.sequence = this.sequence.slice(0, this.seqLength);
                    while (this.sequence.length < this.seqLength) this.sequence.push(0);
                } else {
                    this.sequence = generateSequence('counting', this.seqLength);
                }
            } else {
                this.sequence = generateSequence(this.seqType, this.seqLength);
            }

            this.seqLenVal.value = this.seqLength;

            // Init model
            if (this.architecture === 'lstm') {
                this.model = new LSTMModel(1, HIDDEN_DIM);
            } else {
                this.model = new VanillaRNN(1, HIDDEN_DIM);
            }

            this.currentH = zeroVector(HIDDEN_DIM);
            this.currentC = zeroVector(HIDDEN_DIM);

            this.updateMetrics();
            this.drawAll();
        }

        // ============================================
        // Step through one time step
        // ============================================
        stepOnce() {
            if (this.currentStep >= this.seqLength) return;

            const x = this.sequence[this.currentStep];

            if (this.architecture === 'lstm') {
                const result = this.model.step(x, this.currentH, this.currentC);
                this.currentH = result.h;
                this.currentC = result.c;
                this.gateHistory.push(result.gates);
                this.cellStates.push(result.c.slice());
            } else {
                const result = this.model.step(x, this.currentH);
                this.currentH = result.h;
            }

            this.hiddenStates.push(this.currentH.slice());
            this.currentStep++;

            this.updateMetrics();
            this.drawAll();
        }

        // ============================================
        // Process all steps with animation
        // ============================================
        processAll() {
            if (this.isProcessing) return;

            // If already done, reset first
            if (this.currentStep >= this.seqLength) {
                this.reset();
            }

            this.isProcessing = true;
            this.btnProcess.disabled = true;
            this.setStatus('Processing...');

            const interval = Math.max(100, 1200 - this.speed * 100);

            this.animTimer = setInterval(() => {
                if (this.currentStep >= this.seqLength) {
                    this.stopAnimation();
                    this.setStatus('Complete');
                    return;
                }
                this.stepOnce();
            }, interval);
        }

        stopAnimation() {
            if (this.animTimer) {
                clearInterval(this.animTimer);
                this.animTimer = null;
            }
            this.isProcessing = false;
            this.btnProcess.disabled = false;
        }

        // ============================================
        // Metrics
        // ============================================
        updateMetrics() {
            const step = document.getElementById('metric-timestep');
            const hdim = document.getElementById('metric-hidden-dim');
            const ctype = document.getElementById('metric-cell-type');
            const cinput = document.getElementById('metric-current-input');
            const hnorm = document.getElementById('metric-hidden-norm');

            step.textContent = `${this.currentStep} / ${this.seqLength}`;
            hdim.textContent = HIDDEN_DIM;
            ctype.textContent = this.architecture === 'lstm' ? 'LSTM' : 'Vanilla RNN';

            if (this.currentStep > 0) {
                const idx = this.currentStep - 1;
                cinput.textContent = this.sequence[idx].toFixed(3);
                hnorm.textContent = vecNorm(this.currentH).toFixed(4);
            } else {
                cinput.textContent = '-';
                hnorm.textContent = '-';
            }

            if (this.currentStep >= this.seqLength && this.seqLength > 0) {
                this.setStatus('Complete');
            } else if (this.currentStep > 0) {
                this.setStatus('Step ' + this.currentStep);
            } else {
                this.setStatus('Ready');
            }
        }

        setStatus(text) {
            document.getElementById('metric-status').textContent = text;
        }

        // ============================================
        // Draw Everything
        // ============================================
        drawAll() {
            this.drawNetworkDiagram();
            this.drawHeatmap();
            if (this.architecture === 'lstm') {
                this.drawGates();
            }
        }

        // ============================================
        // Network Diagram (Unrolled RNN)
        // ============================================
        drawNetworkDiagram() {
            const ctx = this.networkCtx;
            const W = NETWORK_W;
            const H = NETWORK_H;
            const colors = getThemeColors();

            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = colors.canvasBg;
            ctx.fillRect(0, 0, W, H);

            const n = this.seqLength;
            const cellW = 60;
            const cellH = 50;
            const totalWidth = n * cellW + (n - 1) * 20;
            const startX = (W - totalWidth) / 2;
            const cellY = H / 2 - cellH / 2;

            for (let t = 0; t < n; t++) {
                const cx = startX + t * (cellW + 20);
                const cy = cellY;
                const isActive = (t === this.currentStep - 1);
                const isProcessed = (t < this.currentStep);

                // Cell rectangle
                ctx.fillStyle = isActive ? colors.cellActiveBg : (isProcessed ? colors.cellBg : 'rgba(128,128,128,0.1)');
                ctx.strokeStyle = isActive ? colors.cellActiveBorder : (isProcessed ? colors.cellBorder : colors.arrowColor);
                ctx.lineWidth = isActive ? 3 : 1.5;

                this.roundRect(ctx, cx, cy, cellW, cellH, 6);
                ctx.fill();
                ctx.stroke();

                // Cell label
                ctx.fillStyle = isActive ? colors.cellActiveBorder : (isProcessed ? colors.cellText : colors.labelColor);
                ctx.font = 'bold 11px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const label = this.architecture === 'lstm' ? 'LSTM' : 'RNN';
                ctx.fillText(label, cx + cellW / 2, cy + cellH / 2);

                // Time step label below cell
                ctx.fillStyle = colors.labelColor;
                ctx.font = '10px ' + this.getMonoFont();
                ctx.fillText('t=' + t, cx + cellW / 2, cy + cellH + 14);

                // Input arrow (from below)
                const inputX = cx + cellW / 2;
                const inputTopY = cy + cellH;
                const inputBottomY = H - 15;

                ctx.strokeStyle = colors.inputColor;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(inputX, inputBottomY);
                ctx.lineTo(inputX, inputTopY + 4);
                ctx.stroke();
                this.drawArrowHead(ctx, inputX, inputTopY + 4, -Math.PI / 2, colors.inputColor);

                // Input label
                ctx.fillStyle = colors.inputColor;
                ctx.font = '10px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                const inputVal = this.sequence[t] !== undefined ? this.sequence[t].toFixed(2) : '?';
                ctx.fillText('x=' + inputVal, inputX, inputBottomY + 10);

                // Output arrow (upward)
                const outputTopY = 20;
                const outputBottomY = cy;

                ctx.strokeStyle = colors.outputColor;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(inputX, outputBottomY - 4);
                ctx.lineTo(inputX, outputTopY + 8);
                ctx.stroke();
                this.drawArrowHead(ctx, inputX, outputTopY + 8, -Math.PI / 2, colors.outputColor);

                // Output label
                ctx.fillStyle = colors.outputColor;
                ctx.font = '10px ' + this.getMonoFont();
                if (isProcessed && this.hiddenStates[t]) {
                    const outVal = this.hiddenStates[t][0].toFixed(2);
                    ctx.fillText('h=' + outVal, inputX, outputTopY);
                } else {
                    ctx.fillText('h_' + t, inputX, outputTopY);
                }

                // Hidden state arrow (right, to next cell)
                if (t < n - 1) {
                    const fromX = cx + cellW;
                    const toX = cx + cellW + 20;
                    const arrowY = cy + cellH / 2;

                    ctx.strokeStyle = isProcessed ? colors.hiddenColor : colors.arrowColor;
                    ctx.lineWidth = isProcessed ? 2 : 1.5;
                    ctx.beginPath();
                    ctx.moveTo(fromX + 2, arrowY);
                    ctx.lineTo(toX - 4, arrowY);
                    ctx.stroke();
                    this.drawArrowHead(ctx, toX - 4, arrowY, 0, isProcessed ? colors.hiddenColor : colors.arrowColor);

                    // h label on arrow
                    if (isProcessed) {
                        ctx.fillStyle = colors.hiddenColor;
                        ctx.font = '9px ' + this.getMonoFont();
                        ctx.fillText('h_' + t, fromX + 10, arrowY - 8);
                    }
                }

                // Glow effect for active cell
                if (isActive) {
                    ctx.save();
                    ctx.shadowColor = colors.cellActiveBorder;
                    ctx.shadowBlur = 12;
                    ctx.strokeStyle = colors.cellActiveBorder;
                    ctx.lineWidth = 2;
                    this.roundRect(ctx, cx, cy, cellW, cellH, 6);
                    ctx.stroke();
                    ctx.restore();
                }
            }

            // Initial hidden state arrow (from left)
            if (n > 0) {
                const firstCX = startX;
                const arrowY = cellY + cellH / 2;
                ctx.strokeStyle = colors.arrowColor;
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(firstCX - 30, arrowY);
                ctx.lineTo(firstCX - 4, arrowY);
                ctx.stroke();
                ctx.setLineDash([]);
                this.drawArrowHead(ctx, firstCX - 4, arrowY, 0, colors.arrowColor);

                ctx.fillStyle = colors.labelColor;
                ctx.font = '9px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                ctx.fillText('h_0=0', firstCX - 18, arrowY - 10);
            }
        }

        // ============================================
        // Hidden State Heatmap
        // ============================================
        drawHeatmap() {
            const ctx = this.heatmapCtx;
            const W = HEATMAP_W;
            const H = HEATMAP_H;
            const colors = getThemeColors();

            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = colors.canvasBg;
            ctx.fillRect(0, 0, W, H);

            if (this.hiddenStates.length === 0) {
                ctx.fillStyle = colors.textMuted;
                ctx.font = '13px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Process sequence to see hidden states', W / 2, H / 2);
                return;
            }

            const numSteps = this.hiddenStates.length;
            const dim = HIDDEN_DIM;
            const margin = { top: 25, bottom: 20, left: 50, right: 20 };
            const plotW = W - margin.left - margin.right;
            const plotH = H - margin.top - margin.bottom;
            const cellW = Math.min(plotW / this.seqLength, 50);
            const cellH = Math.min(plotH / dim, 14);

            const negRgb = hexToRgb(colors.heatmapNeg);
            const zeroRgb = hexToRgb(colors.heatmapZero);
            const posRgb = hexToRgb(colors.heatmapPos);

            // Find max absolute value for normalization
            let maxAbs = 0;
            for (const h of this.hiddenStates) {
                for (const v of h) {
                    if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
                }
            }
            if (maxAbs < 0.01) maxAbs = 1;

            // Draw cells
            for (let t = 0; t < numSteps; t++) {
                for (let d = 0; d < dim; d++) {
                    const val = this.hiddenStates[t][d];
                    const norm = clamp(val / maxAbs, -1, 1);

                    let color;
                    if (norm < 0) {
                        color = lerpColor(zeroRgb, negRgb, -norm);
                    } else {
                        color = lerpColor(zeroRgb, posRgb, norm);
                    }

                    const x = margin.left + t * cellW;
                    const y = margin.top + d * cellH;

                    ctx.fillStyle = rgbStr(color);
                    ctx.fillRect(x, y, cellW - 1, cellH - 1);

                    // Value text in cell (only if cells are big enough)
                    if (cellW > 35 && cellH > 10) {
                        ctx.fillStyle = Math.abs(norm) > 0.5 ? colors.heatmapZero : colors.cellText;
                        ctx.font = '8px ' + this.getMonoFont();
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(val.toFixed(1), x + cellW / 2 - 0.5, y + cellH / 2);
                    }
                }
            }

            // Y-axis labels (hidden dimensions)
            ctx.fillStyle = colors.labelColor;
            ctx.font = '9px ' + this.getMonoFont();
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let d = 0; d < dim; d++) {
                ctx.fillText('h[' + d + ']', margin.left - 5, margin.top + d * cellH + cellH / 2);
            }

            // X-axis labels (time steps)
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            for (let t = 0; t < numSteps; t++) {
                ctx.fillText('t=' + t, margin.left + t * cellW + cellW / 2, margin.top + dim * cellH + 4);
            }

            // Title
            ctx.fillStyle = colors.labelColor;
            ctx.font = '10px ' + this.getMonoFont();
            ctx.textAlign = 'center';
            ctx.fillText('Hidden State Values (blue=negative, red=positive)', W / 2, 8);
        }

        // ============================================
        // Gate Visualization (LSTM only)
        // ============================================
        drawGates() {
            this.drawGateCanvas(this.forgetGateCtx, 'forget');
            this.drawGateCanvas(this.inputGateCtx, 'input');
            this.drawGateCanvas(this.outputGateCtx, 'output');
        }

        drawGateCanvas(ctx, gateType) {
            const W = GATE_W;
            const H = GATE_H;
            const colors = getThemeColors();

            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = colors.canvasBg;
            ctx.fillRect(0, 0, W, H);

            if (this.gateHistory.length === 0 || this.currentStep === 0) {
                ctx.fillStyle = colors.textMuted;
                ctx.font = '10px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No data', W / 2, H / 2);
                return;
            }

            const latestGates = this.gateHistory[this.gateHistory.length - 1];
            const values = latestGates[gateType];
            const colorMap = {
                forget: colors.gateForget,
                input: colors.gateInput,
                output: colors.gateOutput
            };
            const barColor = colorMap[gateType];

            const margin = { top: 8, bottom: 14, left: 6, right: 6 };
            const plotW = W - margin.left - margin.right;
            const plotH = H - margin.top - margin.bottom;
            const barW = Math.min(plotW / values.length - 2, 14);
            const gap = (plotW - barW * values.length) / (values.length + 1);

            for (let i = 0; i < values.length; i++) {
                const val = clamp(values[i], 0, 1);
                const x = margin.left + gap + i * (barW + gap);
                const barH = val * plotH;
                const y = margin.top + plotH - barH;

                // Background bar
                ctx.fillStyle = colors.gateBarBg;
                ctx.fillRect(x, margin.top, barW, plotH);

                // Value bar
                ctx.fillStyle = barColor;
                ctx.fillRect(x, y, barW, barH);

                // Label below
                ctx.fillStyle = colors.labelColor;
                ctx.font = '7px ' + this.getMonoFont();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(val.toFixed(1), x + barW / 2, margin.top + plotH + 2);
            }
        }

        // ============================================
        // Drawing Helpers
        // ============================================
        roundRect(ctx, x, y, w, h, r) {
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
        }

        drawArrowHead(ctx, x, y, angle, color) {
            const size = 6;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
            ctx.lineTo(
                x + Math.cos(angle + 2.5) * size,
                y + Math.sin(angle + 2.5) * size
            );
            ctx.lineTo(
                x + Math.cos(angle - 2.5) * size,
                y + Math.sin(angle - 2.5) * size
            );
            ctx.closePath();
            ctx.fill();
        }

        getMonoFont() {
            return "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";
        }
    }

    // ============================================
    // Bootstrap
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new RNNVisualizer();

        // Wire up info-panel tabs (btn-group variant)
        const tabButtons = document.querySelectorAll('.info-panel-tabs [data-tab]');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                tabButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const panel = btn.closest('.panel');
                panel.querySelectorAll('.info-tab-content').forEach(c => c.classList.remove('active'));
                const target = panel.querySelector('#tab-' + btn.dataset.tab);
                if (target) target.classList.add('active');
            });
        });
    }

    window.addEventListener('vizlib-ready', init);
})();
