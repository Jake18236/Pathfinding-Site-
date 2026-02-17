/**
 * RNN Inference Pipeline Visualizer
 *
 * Animated visualization of the full RNN inference pipeline:
 * text -> tokenizer -> embeddings -> RNN cells -> softmax ->
 * probability distribution -> argmax -> decoder -> output tokens.
 *
 * Data flows bottom-to-top through 9 layers with phase-based animation.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_W = 720;
    const CANVAS_H = 700;
    const HIDDEN_DIM = 8;
    const EMBED_DIM = 8;
    const MONO = "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";

    // Layout: 9 layers bottom-to-top
    const LAYER_MARGIN_TOP = 18;
    const LAYER_H = 56;
    const LAYER_GAP = 14;
    const COL_W = 130;
    const ELLIPSIS_W = 40;

    // Animation phases
    const PHASES = [
        'IDLE',
        'INPUT_TEXT',
        'TOKENIZE',
        // Per-timestep phases are generated dynamically
        'COMPLETE'
    ];

    // Per-timestep sub-phases
    const STEP_PHASES = [
        'EMBED', 'RNN_PROCESS', 'SOFTMAX', 'PROB_DIST',
        'ARGMAX', 'TOKEN_DECODE', 'OUTPUT_TOKEN'
    ];

    // ============================================
    // Seeded RNG (from transformer.js pattern)
    // ============================================
    function hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0;
        }
        return hash;
    }

    function mulberry32(seed) {
        return function() {
            seed |= 0;
            seed = seed + 0x6D2B79F5 | 0;
            let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    // ============================================
    // Math helpers
    // ============================================
    let clamp;

    function tanhFn(x) { return Math.tanh(x); }

    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    function matvec(M, v) {
        const out = [];
        for (let r = 0; r < M.length; r++) {
            let s = 0;
            for (let c = 0; c < v.length; c++) s += M[r][c] * v[c];
            out.push(s);
        }
        return out;
    }

    function vecAdd(a, b) { return a.map((v, i) => v + b[i]); }
    function vecApply(v, fn) { return v.map(fn); }
    function zeroVector(n) { return new Array(n).fill(0); }

    function seededMatrix(rows, cols, scale, rng) {
        const m = [];
        for (let r = 0; r < rows; r++) {
            const row = [];
            for (let c = 0; c < cols; c++) row.push((rng() * 2 - 1) * scale);
            m.push(row);
        }
        return m;
    }

    function seededVector(n, scale, rng) {
        const v = [];
        for (let i = 0; i < n; i++) v.push((rng() * 2 - 1) * scale);
        return v;
    }

    // ============================================
    // Vocabulary & Tokenizer
    // ============================================
    const VOCAB = [
        'I', 'you', 'we', 'they', 'he', 'she', 'it',
        'like', 'love', 'hate', 'want', 'see', 'eat',
        'cute', 'big', 'small', 'happy', 'sad', 'fast',
        'cats', 'dogs', 'kittens', 'puppies', 'birds',
        'the', 'a', 'and', 'but', 'very', 'so',
        'run', 'jump', 'play', 'sleep', 'sing'
    ];

    function buildVocab() {
        const word2idx = {};
        VOCAB.forEach((w, i) => { word2idx[w.toLowerCase()] = i; });
        return { words: VOCAB, word2idx, size: VOCAB.length };
    }

    function tokenize(text, vocab) {
        const words = text.trim().split(/\s+/);
        return words.map(w => {
            const lower = w.toLowerCase().replace(/[^a-z]/g, '');
            return vocab.word2idx[lower] !== undefined ? vocab.word2idx[lower] : 0;
        });
    }

    function buildEmbeddings(vocabSize, dim) {
        const rng = mulberry32(42);
        const embeddings = [];
        for (let i = 0; i < vocabSize; i++) {
            const v = [];
            for (let d = 0; d < dim; d++) v.push((rng() * 2 - 1) * 0.5);
            embeddings.push(v);
        }
        return embeddings;
    }

    // ============================================
    // Vanilla RNN with output layer
    // ============================================
    class VanillaRNN {
        constructor(inputDim, hiddenDim, vocabSize) {
            const rng = mulberry32(123);
            this.hiddenDim = hiddenDim;
            this.vocabSize = vocabSize;
            this.Wh = seededMatrix(hiddenDim, hiddenDim, 0.3, rng);
            this.Wx = seededMatrix(hiddenDim, inputDim, 0.5, rng);
            this.bh = seededVector(hiddenDim, 0.1, rng);
            this.Wy = seededMatrix(vocabSize, hiddenDim, 0.4, rng);
            this.by = seededVector(vocabSize, 0.1, rng);
        }

        step(embedding, hPrev) {
            const whh = matvec(this.Wh, hPrev);
            const wxx = matvec(this.Wx, embedding);
            const pre = vecAdd(vecAdd(whh, wxx), this.bh);
            const h = vecApply(pre, tanhFn);
            return h;
        }

        output(h) {
            const logits = vecAdd(matvec(this.Wy, h), this.by);
            const probs = softmax(logits);
            return { logits, probs };
        }
    }

    // ============================================
    // Theme-aware color helper
    // ============================================
    function getThemeColors() {
        const s = getComputedStyle(document.documentElement);
        const g = name => s.getPropertyValue(name).trim();
        return {
            inputBg:        g('--rnn-input-bg'),
            inputBorder:    g('--rnn-input-border'),
            tokenBg:        g('--rnn-token-bg'),
            tokenBorder:    g('--rnn-token-border'),
            embedBg:        g('--rnn-embed-bg'),
            embedBorder:    g('--rnn-embed-border'),
            cellBg:         g('--rnn-cell-bg'),
            cellBorder:     g('--rnn-cell-border'),
            cellActiveBg:   g('--rnn-cell-active-bg'),
            cellActiveBorder: g('--rnn-cell-active-border'),
            softmaxBg:      g('--rnn-softmax-bg'),
            softmaxBorder:  g('--rnn-softmax-border'),
            probBar:        g('--rnn-prob-bar'),
            probBarAlt:     g('--rnn-prob-bar-alt'),
            argmaxBg:       g('--rnn-argmax-bg'),
            argmaxBorder:   g('--rnn-argmax-border'),
            decodeBg:       g('--rnn-decode-bg'),
            decodeBorder:   g('--rnn-decode-border'),
            outputBg:       g('--rnn-output-bg'),
            outputBorder:   g('--rnn-output-border'),
            arrowColor:     g('--rnn-arrow-color'),
            labelColor:     g('--rnn-label-color'),
            cellText:       g('--rnn-cell-text'),
            ellipsisColor:  g('--rnn-ellipsis-color'),
            hiddenColor:    g('--rnn-hidden-color'),
            canvasBg:       g('--viz-canvas-bg'),
            textMuted:      g('--viz-text-muted')
        };
    }

    // ============================================
    // Main Visualizer Class
    // ============================================
    class RNNPipelineVisualizer {
        constructor() {
            this.canvas = document.getElementById('pipeline-canvas');
            this.ctx = this.canvas.getContext('2d');

            // HiDPI setup
            if (window.VizLib && window.VizLib.CanvasUtils) {
                window.VizLib.CanvasUtils.setupHiDPICanvas(this.canvas);
            }

            // Vocab & embeddings
            this.vocab = buildVocab();
            this.embeddings = buildEmbeddings(this.vocab.size, EMBED_DIM);

            // State
            this.inputText = 'I like cute kittens and';
            this.tokens = [];
            this.tokenIds = [];
            this.numTimesteps = 0;
            this.model = null;
            this.speed = 5;
            this.isProcessing = false;
            this.animId = null;

            // Animation state machine
            this.phase = 'IDLE';
            this.currentTimestep = 0;
            this.stepPhaseIdx = 0;
            this.phaseProgress = 0;       // 0..1 for fade-in
            this.phaseStartTime = 0;

            // Computed results per timestep
            this.embeddingsUsed = [];     // embedding vectors per timestep
            this.hiddenStates = [];       // h vectors
            this.currentH = null;
            this.probDists = [];          // probability arrays
            this.argmaxIndices = [];      // predicted indices
            this.decodedTokens = [];      // predicted words
            this.outputTokens = [];       // final output tokens

            // Visible column indices: we show max 3 timesteps + ellipsis
            this.visibleCols = [];

            // DOM
            this.inputField = document.getElementById('input-text');
            this.btnProcess = document.getElementById('btn-process');
            this.btnStep = document.getElementById('btn-step');
            this.btnReset = document.getElementById('btn-reset');
            this.speedSlider = document.getElementById('speed-slider');
            this.speedValue = document.getElementById('speed-value');

            this.bindEvents();
            this.reset();
        }

        bindEvents() {
            this.btnProcess.addEventListener('click', () => this.processAll());
            this.btnStep.addEventListener('click', () => this.stepOnce());
            this.btnReset.addEventListener('click', () => this.reset());
            this.speedSlider.addEventListener('input', () => {
                this.speed = parseInt(this.speedSlider.value);
                this.speedValue.textContent = this.speed;
            });
            this.inputField.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') this.reset();
            });
            document.addEventListener('themechange', () => this.draw());
        }

        // ============================================
        // Reset
        // ============================================
        reset() {
            this.stopAnimation();
            this.inputText = this.inputField.value.trim() || 'I like cute kittens and';
            this.inputField.value = this.inputText;

            // Tokenize
            const words = this.inputText.split(/\s+/);
            this.tokens = words;
            this.tokenIds = tokenize(this.inputText, this.vocab);
            this.numTimesteps = this.tokens.length;

            // Build model
            this.model = new VanillaRNN(EMBED_DIM, HIDDEN_DIM, this.vocab.size);
            this.currentH = zeroVector(HIDDEN_DIM);

            // Clear results
            this.embeddingsUsed = [];
            this.hiddenStates = [];
            this.probDists = [];
            this.argmaxIndices = [];
            this.decodedTokens = [];
            this.outputTokens = [];

            // Compute visible columns
            this.computeVisibleCols();

            // Reset animation
            this.phase = 'IDLE';
            this.currentTimestep = 0;
            this.stepPhaseIdx = 0;
            this.phaseProgress = 1;
            this.isProcessing = false;

            this.updateMetrics();
            this.draw();
        }

        computeVisibleCols() {
            const n = this.numTimesteps;
            if (n <= 3) {
                this.visibleCols = [];
                for (let i = 0; i < n; i++) this.visibleCols.push(i);
                this.hasEllipsis = false;
            } else {
                // Show first 2, ellipsis, then last
                this.visibleCols = [0, 1, n - 1];
                this.hasEllipsis = true;
            }
        }

        // ============================================
        // Phase management
        // ============================================
        getPhaseDuration() {
            const base = Math.max(200, 1400 - this.speed * 120);
            return base;
        }

        getNextPhase() {
            if (this.phase === 'IDLE') return 'INPUT_TEXT';
            if (this.phase === 'INPUT_TEXT') return 'TOKENIZE';

            if (this.phase === 'TOKENIZE') {
                this.currentTimestep = 0;
                this.stepPhaseIdx = 0;
                return 'STEP_EMBED';
            }

            // Per-timestep phases
            if (this.phase.startsWith('STEP_')) {
                const subPhases = ['STEP_EMBED', 'STEP_RNN', 'STEP_SOFTMAX', 'STEP_PROB', 'STEP_ARGMAX', 'STEP_DECODE', 'STEP_OUTPUT'];
                const idx = subPhases.indexOf(this.phase);

                if (idx < subPhases.length - 1) {
                    return subPhases[idx + 1];
                }
                // End of timestep sub-phases, advance timestep
                this.currentTimestep++;
                if (this.currentTimestep < this.numTimesteps) {
                    this.stepPhaseIdx = 0;
                    return 'STEP_EMBED';
                }
                return 'COMPLETE';
            }

            return 'COMPLETE';
        }

        advancePhase() {
            const prev = this.phase;
            const next = this.getNextPhase();
            if (next === prev && next === 'COMPLETE') return false;

            // Execute computation for the entering phase
            this.executePhaseComputation(next);

            this.phase = next;
            this.phaseProgress = 0;
            this.phaseStartTime = performance.now();
            this.updateMetrics();
            return true;
        }

        executePhaseComputation(phase) {
            const t = this.currentTimestep;

            if (phase === 'STEP_EMBED') {
                const emb = this.embeddings[this.tokenIds[t]];
                this.embeddingsUsed[t] = emb;
            }
            else if (phase === 'STEP_RNN') {
                const emb = this.embeddingsUsed[t];
                const h = this.model.step(emb, this.currentH);
                this.currentH = h;
                this.hiddenStates[t] = h.slice();
            }
            else if (phase === 'STEP_SOFTMAX') {
                const { probs } = this.model.output(this.hiddenStates[t]);
                this.probDists[t] = probs;
            }
            else if (phase === 'STEP_ARGMAX') {
                const probs = this.probDists[t];
                let maxIdx = 0;
                for (let i = 1; i < probs.length; i++) {
                    if (probs[i] > probs[maxIdx]) maxIdx = i;
                }
                this.argmaxIndices[t] = maxIdx;
            }
            else if (phase === 'STEP_DECODE') {
                this.decodedTokens[t] = this.vocab.words[this.argmaxIndices[t]];
            }
            else if (phase === 'STEP_OUTPUT') {
                this.outputTokens[t] = this.decodedTokens[t];
            }
        }

        // ============================================
        // Step / Process
        // ============================================
        stepOnce() {
            if (this.phase === 'COMPLETE') return;
            this.stopAnimation();
            const advanced = this.advancePhase();
            if (advanced) {
                this.phaseProgress = 1;
                this.draw();
            }
        }

        processAll() {
            if (this.isProcessing) return;
            if (this.phase === 'COMPLETE') this.reset();

            this.isProcessing = true;
            this.btnProcess.disabled = true;
            this.setStatus('Processing...');

            const animate = (now) => {
                if (!this.isProcessing) return;

                const elapsed = now - this.phaseStartTime;
                const duration = this.getPhaseDuration();
                this.phaseProgress = clamp(elapsed / duration, 0, 1);

                this.draw();

                if (this.phaseProgress >= 1) {
                    if (this.phase === 'COMPLETE') {
                        this.stopAnimation();
                        this.setStatus('Complete');
                        return;
                    }
                    const advanced = this.advancePhase();
                    if (!advanced) {
                        this.stopAnimation();
                        this.setStatus('Complete');
                        return;
                    }
                }

                this.animId = requestAnimationFrame(animate);
            };

            // Kick off
            if (this.phase === 'IDLE') {
                this.advancePhase();
            }
            this.phaseStartTime = performance.now();
            this.animId = requestAnimationFrame(animate);
        }

        stopAnimation() {
            if (this.animId) {
                cancelAnimationFrame(this.animId);
                this.animId = null;
            }
            this.isProcessing = false;
            this.btnProcess.disabled = false;
        }

        // ============================================
        // Metrics
        // ============================================
        updateMetrics() {
            const phaseEl = document.getElementById('metric-phase');
            const tsEl = document.getElementById('metric-timestep');
            const vocabEl = document.getElementById('metric-vocab-size');
            const hdimEl = document.getElementById('metric-hidden-dim');
            const predEl = document.getElementById('metric-predicted');

            const phaseNames = {
                'IDLE': 'Idle',
                'INPUT_TEXT': 'Input Text',
                'TOKENIZE': 'Tokenize',
                'STEP_EMBED': 'Embedding',
                'STEP_RNN': 'RNN Cell',
                'STEP_SOFTMAX': 'Softmax',
                'STEP_PROB': 'Prob. Dist.',
                'STEP_ARGMAX': 'Argmax',
                'STEP_DECODE': 'Token Decode',
                'STEP_OUTPUT': 'Output Token',
                'COMPLETE': 'Complete'
            };

            phaseEl.textContent = phaseNames[this.phase] || this.phase;
            tsEl.textContent = `${this.currentTimestep} / ${this.numTimesteps}`;
            vocabEl.textContent = this.vocab.size;
            hdimEl.textContent = HIDDEN_DIM;

            if (this.outputTokens.length > 0) {
                predEl.textContent = this.outputTokens.filter(Boolean).join(', ');
            } else {
                predEl.textContent = '-';
            }

            // Update status
            if (this.phase === 'IDLE') {
                this.setStatus('Ready');
            } else if (this.phase === 'COMPLETE') {
                this.setStatus('Complete');
            } else if (this.isProcessing) {
                this.setStatus('Processing...');
            } else {
                this.setStatus('Stepping (t=' + this.currentTimestep + ')');
            }
        }

        setStatus(text) {
            document.getElementById('metric-status').textContent = text;
        }

        // ============================================
        // Drawing
        // ============================================
        draw() {
            const ctx = this.ctx;
            const C = getThemeColors();
            const dpr = window.devicePixelRatio || 1;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
            ctx.fillStyle = C.canvasBg;
            ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

            // Compute layout positions
            const layout = this.computeLayout();

            // Draw from bottom to top (only layers that are visible based on phase)
            this.drawInputTextLayer(ctx, layout, C);
            this.drawTokenizerLayer(ctx, layout, C);
            this.drawColumnarLayers(ctx, layout, C);
        }

        computeLayout() {
            // 9 layers from top (output) to bottom (input)
            // Drawing bottom-to-top: input at bottom, output at top
            const layers = [
                'output',      // 0 - top
                'decode',      // 1
                'argmax',      // 2
                'probDist',    // 3
                'softmax',     // 4
                'rnnCell',     // 5
                'embedding',   // 6
                'tokenizer',   // 7
                'inputText'    // 8 - bottom
            ];

            const positions = {};
            for (let i = 0; i < layers.length; i++) {
                positions[layers[i]] = {
                    y: LAYER_MARGIN_TOP + i * (LAYER_H + LAYER_GAP),
                    h: LAYER_H
                };
            }

            // Column x-positions
            const numVisCols = this.visibleCols.length;
            const totalColsWidth = numVisCols * COL_W + (this.hasEllipsis ? ELLIPSIS_W : 0);
            const startX = (CANVAS_W - totalColsWidth) / 2;

            const colPositions = [];
            for (let i = 0; i < numVisCols; i++) {
                let x;
                if (this.hasEllipsis && i === numVisCols - 1) {
                    // Last col after ellipsis
                    x = startX + 2 * COL_W + ELLIPSIS_W;
                } else {
                    x = startX + i * COL_W;
                }
                colPositions.push({ x, w: COL_W - 10, tsIndex: this.visibleCols[i] });
            }

            // Ellipsis x-position
            let ellipsisX = null;
            if (this.hasEllipsis) {
                ellipsisX = startX + 2 * COL_W;
            }

            return { positions, colPositions, ellipsisX, startX };
        }

        isLayerVisible(layerName) {
            const phaseOrder = ['IDLE', 'INPUT_TEXT', 'TOKENIZE',
                'STEP_EMBED', 'STEP_RNN', 'STEP_SOFTMAX', 'STEP_PROB',
                'STEP_ARGMAX', 'STEP_DECODE', 'STEP_OUTPUT', 'COMPLETE'];

            const layerPhaseMap = {
                'inputText':  'INPUT_TEXT',
                'tokenizer':  'TOKENIZE',
                'embedding':  'STEP_EMBED',
                'rnnCell':    'STEP_RNN',
                'softmax':    'STEP_SOFTMAX',
                'probDist':   'STEP_PROB',
                'argmax':     'STEP_ARGMAX',
                'decode':     'STEP_DECODE',
                'output':     'STEP_OUTPUT'
            };

            const requiredPhase = layerPhaseMap[layerName];
            if (!requiredPhase) return false;

            const currentIdx = phaseOrder.indexOf(this.phase);
            const requiredIdx = phaseOrder.indexOf(requiredPhase);

            return currentIdx >= requiredIdx;
        }

        getLayerAlpha(layerName) {
            const phaseOrder = ['IDLE', 'INPUT_TEXT', 'TOKENIZE',
                'STEP_EMBED', 'STEP_RNN', 'STEP_SOFTMAX', 'STEP_PROB',
                'STEP_ARGMAX', 'STEP_DECODE', 'STEP_OUTPUT', 'COMPLETE'];

            const layerPhaseMap = {
                'inputText':  'INPUT_TEXT',
                'tokenizer':  'TOKENIZE',
                'embedding':  'STEP_EMBED',
                'rnnCell':    'STEP_RNN',
                'softmax':    'STEP_SOFTMAX',
                'probDist':   'STEP_PROB',
                'argmax':     'STEP_ARGMAX',
                'decode':     'STEP_DECODE',
                'output':     'STEP_OUTPUT'
            };

            const requiredPhase = layerPhaseMap[layerName];
            const currentIdx = phaseOrder.indexOf(this.phase);
            const requiredIdx = phaseOrder.indexOf(requiredPhase);

            if (currentIdx > requiredIdx) return 1;
            if (currentIdx === requiredIdx) return this.phaseProgress;
            return 0;
        }

        // For per-column layers, check if this column's timestep has reached the phase
        isColVisible(layerName, tsIndex) {
            const layerPhaseMap = {
                'embedding':  'STEP_EMBED',
                'rnnCell':    'STEP_RNN',
                'softmax':    'STEP_SOFTMAX',
                'probDist':   'STEP_PROB',
                'argmax':     'STEP_ARGMAX',
                'decode':     'STEP_DECODE',
                'output':     'STEP_OUTPUT'
            };

            const subPhaseOrder = ['STEP_EMBED', 'STEP_RNN', 'STEP_SOFTMAX', 'STEP_PROB',
                'STEP_ARGMAX', 'STEP_DECODE', 'STEP_OUTPUT'];

            const requiredPhase = layerPhaseMap[layerName];
            const reqSubIdx = subPhaseOrder.indexOf(requiredPhase);
            const curSubIdx = subPhaseOrder.indexOf(this.phase);

            // Past timesteps are fully visible
            if (tsIndex < this.currentTimestep) return 1;
            // Future timesteps not visible
            if (tsIndex > this.currentTimestep) return 0;
            // Current timestep - check sub-phase
            if (!this.phase.startsWith('STEP_')) return 0;

            if (curSubIdx > reqSubIdx) return 1;
            if (curSubIdx === reqSubIdx) return this.phaseProgress;
            return 0;
        }

        // ============================================
        // Layer Drawing Methods
        // ============================================

        drawInputTextLayer(ctx, layout, C) {
            const alpha = this.getLayerAlpha('inputText');
            if (alpha <= 0) return;

            ctx.save();
            ctx.globalAlpha = alpha;

            const pos = layout.positions.inputText;
            const fullW = CANVAS_W - 60;
            const x = 30;

            // Background
            ctx.fillStyle = C.inputBg;
            ctx.strokeStyle = C.inputBorder;
            ctx.lineWidth = 1.5;
            this.roundRect(ctx, x, pos.y, fullW, pos.h, 6);
            ctx.fill();
            ctx.stroke();

            // Label
            ctx.fillStyle = C.cellText;
            ctx.font = 'bold 11px ' + MONO;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('Input Text', x + 10, pos.y + 6);

            // Text content
            ctx.font = '13px ' + MONO;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('X = "' + this.inputText + '"', CANVAS_W / 2, pos.y + pos.h / 2 + 6);

            ctx.restore();
        }

        drawTokenizerLayer(ctx, layout, C) {
            const alpha = this.getLayerAlpha('tokenizer');
            if (alpha <= 0) return;

            ctx.save();
            ctx.globalAlpha = alpha;

            const pos = layout.positions.tokenizer;
            const fullW = CANVAS_W - 60;
            const x = 30;

            // Background
            ctx.fillStyle = C.tokenBg;
            ctx.strokeStyle = C.tokenBorder;
            ctx.lineWidth = 1.5;
            this.roundRect(ctx, x, pos.y, fullW, pos.h, 6);
            ctx.fill();
            ctx.stroke();

            // Label
            ctx.fillStyle = C.cellText;
            ctx.font = 'bold 11px ' + MONO;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('Tokenizer', x + 10, pos.y + 6);

            // Token display
            ctx.font = '11px ' + MONO;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const tokenStr = this.tokenIds.map((id, i) => {
                if (this.numTimesteps > 5 && i >= 2 && i < this.numTimesteps - 1) {
                    return i === 2 ? '...' : null;
                }
                return 'xt_' + i + '=' + id;
            }).filter(Boolean).join('  ');

            ctx.fillText('[' + tokenStr + '] = tokenizer(X)', CANVAS_W / 2, pos.y + pos.h / 2 + 6);

            // Draw arrow from input to tokenizer
            this.drawUpArrow(ctx, CANVAS_W / 2, layout.positions.inputText.y, pos.y + pos.h, C.arrowColor);

            ctx.restore();
        }

        drawColumnarLayers(ctx, layout, C) {
            const colLayers = ['embedding', 'rnnCell', 'softmax', 'probDist', 'argmax', 'decode', 'output'];

            for (const col of layout.colPositions) {
                const t = col.tsIndex;

                for (const layer of colLayers) {
                    const vis = this.isColVisible(layer, t);
                    if (vis <= 0) continue;

                    ctx.save();
                    ctx.globalAlpha = vis;

                    const pos = layout.positions[layer];
                    const cx = col.x;
                    const cw = col.w;

                    this.drawLayerBox(ctx, layer, cx, pos.y, cw, pos.h, t, C);

                    // Draw arrow from layer below
                    const layerBelow = this.getLayerBelow(layer);
                    if (layerBelow) {
                        const belowPos = layout.positions[layerBelow];
                        if (layerBelow === 'tokenizer') {
                            // Arrow from full-width tokenizer to per-column embedding
                            this.drawUpArrow(ctx, cx + cw / 2, belowPos.y, pos.y + pos.h, C.arrowColor);
                        } else {
                            this.drawUpArrow(ctx, cx + cw / 2, belowPos.y, pos.y + pos.h, C.arrowColor);
                        }
                    }

                    ctx.restore();
                }

                // Draw hidden state arrow between RNN cells (horizontal)
                if (t < this.numTimesteps - 1 && this.hiddenStates[t]) {
                    const rnnPos = layout.positions.rnnCell;
                    const nextCol = layout.colPositions.find(c => c.tsIndex === t + 1);
                    // Only draw if both columns visible
                    if (nextCol && this.isColVisible('rnnCell', t) >= 1) {
                        ctx.save();
                        ctx.strokeStyle = C.hiddenColor;
                        ctx.fillStyle = C.hiddenColor;
                        ctx.lineWidth = 2;
                        const fromX = col.x + col.w;
                        const toX = nextCol.x;
                        const arrowY = rnnPos.y + rnnPos.h / 2;

                        if (toX > fromX + 5) {
                            ctx.beginPath();
                            ctx.moveTo(fromX + 2, arrowY);
                            ctx.lineTo(toX - 8, arrowY);
                            ctx.stroke();
                            this.drawArrowHead(ctx, toX - 6, arrowY, 0, C.hiddenColor);

                            ctx.font = '8px ' + MONO;
                            ctx.textAlign = 'center';
                            ctx.fillText('h_' + t, (fromX + toX) / 2, arrowY - 8);
                        }
                        ctx.restore();
                    }
                }
            }

            // Draw h_0 arrow to first column
            if (layout.colPositions.length > 0 && this.isColVisible('rnnCell', this.visibleCols[0]) > 0) {
                const firstCol = layout.colPositions[0];
                const rnnPos = layout.positions.rnnCell;
                const arrowY = rnnPos.y + rnnPos.h / 2;
                ctx.save();
                ctx.strokeStyle = C.arrowColor;
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(firstCol.x - 30, arrowY);
                ctx.lineTo(firstCol.x - 4, arrowY);
                ctx.stroke();
                ctx.setLineDash([]);
                this.drawArrowHead(ctx, firstCol.x - 4, arrowY, 0, C.arrowColor);
                ctx.fillStyle = C.labelColor;
                ctx.font = '9px ' + MONO;
                ctx.textAlign = 'center';
                ctx.fillText('h₀=0', firstCol.x - 18, arrowY - 10);
                ctx.restore();
            }

            // Draw ellipsis column
            if (this.hasEllipsis && layout.ellipsisX != null) {
                this.drawEllipsisColumn(ctx, layout, C);
            }
        }

        drawLayerBox(ctx, layer, x, y, w, h, t, C) {
            const configs = {
                embedding: { bg: C.embedBg, border: C.embedBorder, label: 'Embed' },
                rnnCell:   { bg: C.cellBg, border: C.cellBorder, label: 'RNN' },
                softmax:   { bg: C.softmaxBg, border: C.softmaxBorder, label: 'Softmax' },
                probDist:  { bg: 'transparent', border: C.probBar, label: '' },
                argmax:    { bg: C.argmaxBg, border: C.argmaxBorder, label: 'Argmax' },
                decode:    { bg: C.decodeBg, border: C.decodeBorder, label: 'Decode' },
                output:    { bg: C.outputBg, border: C.outputBorder, label: 'Output' }
            };

            const cfg = configs[layer];

            // Active cell highlight for current timestep RNN
            const isActive = (layer === 'rnnCell' && t === this.currentTimestep &&
                              this.phase.startsWith('STEP_') && this.phase === 'STEP_RNN');

            const bg = isActive ? C.cellActiveBg : cfg.bg;
            const border = isActive ? C.cellActiveBorder : cfg.border;

            // Special: prob dist draws bar chart
            if (layer === 'probDist') {
                this.drawProbBars(ctx, x, y, w, h, t, C);
                return;
            }

            // Box
            if (bg !== 'transparent') {
                ctx.fillStyle = bg;
                ctx.strokeStyle = border;
                ctx.lineWidth = isActive ? 2.5 : 1.5;
                this.roundRect(ctx, x, y, w, h, 5);
                ctx.fill();
                ctx.stroke();

                if (isActive) {
                    ctx.save();
                    ctx.shadowColor = C.cellActiveBorder;
                    ctx.shadowBlur = 10;
                    ctx.strokeStyle = C.cellActiveBorder;
                    ctx.lineWidth = 2;
                    this.roundRect(ctx, x, y, w, h, 5);
                    ctx.stroke();
                    ctx.restore();
                }
            }

            // Label
            ctx.fillStyle = C.cellText;
            ctx.font = 'bold 10px ' + MONO;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            if (cfg.label) ctx.fillText(cfg.label, x + 5, y + 4);

            // Content
            ctx.font = '10px ' + MONO;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const cy = y + h / 2 + 4;

            if (layer === 'embedding') {
                const emb = this.embeddingsUsed[t];
                if (emb) {
                    const shortEmb = emb.slice(0, 3).map(v => v.toFixed(1)).join(',');
                    ctx.fillText('e_' + t + '=[' + shortEmb + '...]', x + w / 2, cy);
                } else {
                    ctx.fillText('e_' + t, x + w / 2, cy);
                }
            }
            else if (layer === 'rnnCell') {
                const h_t = this.hiddenStates[t];
                if (h_t) {
                    const shortH = h_t.slice(0, 2).map(v => v.toFixed(2)).join(',');
                    ctx.fillText('h_' + t + '=[' + shortH + '...]', x + w / 2, cy);
                } else {
                    ctx.fillText('tanh(h*W+e*W+b)', x + w / 2, cy);
                }
            }
            else if (layer === 'softmax') {
                ctx.fillText('y_' + t + '=softmax(Wh+b)', x + w / 2, cy);
            }
            else if (layer === 'argmax') {
                const idx = this.argmaxIndices[t];
                ctx.fillText(idx !== undefined ? 'y_' + t + '=' + idx : 'argmax(y_' + t + ')', x + w / 2, cy);
            }
            else if (layer === 'decode') {
                const tok = this.decodedTokens[t];
                ctx.fillText(tok ? 'decode(' + this.argmaxIndices[t] + ')' : 'decode(idx)', x + w / 2, cy);
            }
            else if (layer === 'output') {
                const tok = this.outputTokens[t];
                ctx.font = 'bold 12px ' + MONO;
                ctx.fillText(tok ? '"' + tok + '"' : '?', x + w / 2, cy);
            }
        }

        drawProbBars(ctx, x, y, w, h, t, C) {
            const probs = this.probDists[t];
            if (!probs) {
                ctx.strokeStyle = C.arrowColor;
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                this.roundRect(ctx, x, y, w, h, 5);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = C.labelColor;
                ctx.font = '10px ' + MONO;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('P(vocab)', x + w / 2, y + h / 2);
                return;
            }

            // Get top-5 probabilities
            const indexed = probs.map((p, i) => ({ p, i }));
            indexed.sort((a, b) => b.p - a.p);
            const top = indexed.slice(0, 5);

            const barMargin = 4;
            const barAreaW = w - barMargin * 2;
            const barH = Math.min(8, (h - 14) / top.length - 1);
            const startY = y + 4;

            // Label
            ctx.fillStyle = C.labelColor;
            ctx.font = '8px ' + MONO;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';

            for (let i = 0; i < top.length; i++) {
                const bx = x + barMargin;
                const by = startY + i * (barH + 2);
                const bw = top[i].p * (barAreaW - 30);
                const word = this.vocab.words[top[i].i] || '?';

                // Bar background
                ctx.fillStyle = i === 0 ? C.probBar : C.probBarAlt;
                ctx.globalAlpha *= 0.8;
                ctx.fillRect(bx, by, Math.max(bw, 2), barH);
                ctx.globalAlpha /= 0.8;

                // Label
                ctx.fillStyle = C.cellText;
                ctx.font = '7px ' + MONO;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(word, bx + Math.max(bw, 2) + 2, by);
            }
        }

        drawEllipsisColumn(ctx, layout, C) {
            const colLayers = ['embedding', 'rnnCell', 'softmax', 'probDist', 'argmax', 'decode', 'output'];
            const ex = layout.ellipsisX;

            for (const layer of colLayers) {
                // Check if we have any visible columns for this layer
                if (this.visibleCols.length < 3) continue;
                const vis = this.isColVisible(layer, this.visibleCols[1]);
                if (vis <= 0) continue;

                const pos = layout.positions[layer];
                ctx.save();
                ctx.globalAlpha = vis * 0.6;
                ctx.fillStyle = C.ellipsisColor;
                ctx.font = 'bold 16px ' + MONO;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('...', ex + ELLIPSIS_W / 2, pos.y + pos.h / 2);

                // Dotted line above and below
                ctx.strokeStyle = C.ellipsisColor;
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 4]);
                ctx.beginPath();
                ctx.moveTo(ex + ELLIPSIS_W / 2, pos.y);
                ctx.lineTo(ex + ELLIPSIS_W / 2, pos.y + pos.h);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }

            // Horizontal dotted arrow through RNN layer
            const rnnPos = layout.positions.rnnCell;
            if (layout.colPositions.length >= 2) {
                const col1 = layout.colPositions[1];
                const col2 = layout.colPositions[layout.colPositions.length - 1];
                const arrowY = rnnPos.y + rnnPos.h / 2;

                if (this.isColVisible('rnnCell', this.visibleCols[1]) > 0) {
                    ctx.save();
                    ctx.strokeStyle = C.hiddenColor;
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(col1.x + col1.w + 2, arrowY);
                    ctx.lineTo(col2.x - 4, arrowY);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    this.drawArrowHead(ctx, col2.x - 4, arrowY, 0, C.hiddenColor);
                    ctx.restore();
                }
            }
        }

        getLayerBelow(layer) {
            const order = ['inputText', 'tokenizer', 'embedding', 'rnnCell', 'softmax', 'probDist', 'argmax', 'decode', 'output'];
            const idx = order.indexOf(layer);
            if (idx <= 0) return null;
            return order[idx - 1];
        }

        // ============================================
        // Drawing Helpers
        // ============================================
        drawUpArrow(ctx, x, topY, bottomY, color) {
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, bottomY - 2);
            ctx.lineTo(x, topY + 8);
            ctx.stroke();
            this.drawArrowHead(ctx, x, topY + 6, -Math.PI / 2, color);
        }

        drawArrowHead(ctx, x, y, angle, color) {
            const size = 5;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
            ctx.lineTo(x + Math.cos(angle + 2.5) * size, y + Math.sin(angle + 2.5) * size);
            ctx.lineTo(x + Math.cos(angle - 2.5) * size, y + Math.sin(angle - 2.5) * size);
            ctx.closePath();
            ctx.fill();
        }

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
    }

    // ============================================
    // Bootstrap
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new RNNPipelineVisualizer();

        // Wire up info-panel tabs
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
