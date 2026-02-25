/**
 * Attention Mechanism Visualizer (D3/SVG)
 *
 * Animated visualization of scaled dot-product attention:
 * embed tokens → project Q/K/V → compute scores → scale → softmax →
 * build heatmap → aggregate values → output. Pre-computes all data,
 * then reveals via a phase state machine with D3 transitions.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_W = 720;
    const MONO = "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";

    const ZONE_GAP = 15;
    const SUB_GAP = 10;

    const PRESETS = {
        'default': 'I love cute cats',
        'short':   'The cat sat',
        'long':    'She saw the red car',
    };

    const DIM_OPTIONS = [4, 6, 8];

    const PHASES = [
        'IDLE', 'SHOW_TOKENS', 'SHOW_EMBEDDINGS',
        'PROJECT_Q', 'PROJECT_K', 'PROJECT_V',
        'COMPUTE_SCORES', 'SCALE_SCORES', 'APPLY_SOFTMAX',
        'SHOW_OUTPUT', 'COMPLETE'
    ];

    const PHASE_DURATION_MULT = {
        'COMPUTE_SCORES': 1.5,
        'SCALE_SCORES': 0.5,
    };

    const TERM_PHASE_MAP = {
        'tokens': 'SHOW_TOKENS', 'E': 'SHOW_EMBEDDINGS',
        'Q': 'PROJECT_Q', 'K': 'PROJECT_K', 'V': 'PROJECT_V',
        'QKT': 'COMPUTE_SCORES', 'scaled': 'SCALE_SCORES', 'softmax': 'APPLY_SOFTMAX',
        'ctxVectors': 'SHOW_OUTPUT', 'resE': 'SHOW_OUTPUT',
    };
    const PHASE_TERM_MAP = Object.fromEntries(Object.entries(TERM_PHASE_MAP).map(([k,v]) => [v,k]));

    // ============================================
    // Seeded PRNG
    // ============================================
    function hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0;
        }
        return hash >>> 0;
    }

    function mulberry32(seed) {
        let s = seed | 0;
        return function() {
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // ============================================
    // Linear Algebra Helpers
    // ============================================
    function dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    function matvec(M, v) {
        const result = [];
        for (let i = 0; i < M.length; i++) {
            result.push(dot(M[i], v));
        }
        return result;
    }

    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    function seededVector(rng, dim) {
        const v = [];
        for (let i = 0; i < dim; i++) {
            const u1 = Math.max(1e-10, rng());
            const u2 = rng();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            v.push(z * 0.5);
        }
        return v;
    }

    function seededMatrix(rng, rows, cols) {
        const M = [];
        for (let r = 0; r < rows; r++) {
            M.push(seededVector(rng, cols));
        }
        return M;
    }

    // ============================================
    // Pre-computation (single head)
    // ============================================
    function precomputeHead(embeddings, d, rng) {
        const N = embeddings.length;
        const W_Q = seededMatrix(rng, d, d);
        const W_K = seededMatrix(rng, d, d);
        const W_V = seededMatrix(rng, d, d);

        const Q = embeddings.map(e => matvec(W_Q, e));
        const K = embeddings.map(e => matvec(W_K, e));
        const V = embeddings.map(e => matvec(W_V, e));

        const rawScores = [];
        for (let i = 0; i < N; i++) {
            const row = [];
            for (let j = 0; j < N; j++) {
                row.push(dot(Q[i], K[j]));
            }
            rawScores.push(row);
        }

        const scaleFactor = Math.sqrt(d);
        const scaledScores = rawScores.map(row => row.map(s => s / scaleFactor));
        const attentionWeights = scaledScores.map(row => softmax(row));

        const contextVectors = [];
        const outputs = [];
        for (let i = 0; i < N; i++) {
            const ctx = new Array(d).fill(0);
            for (let j = 0; j < N; j++) {
                const w = attentionWeights[i][j];
                for (let k = 0; k < d; k++) {
                    ctx[k] += w * V[j][k];
                }
            }
            contextVectors.push(ctx);
            // Residual connection: add input embedding
            const out = ctx.map((c, k) => c + embeddings[i][k]);
            outputs.push(out);
        }

        return { W_Q, W_K, W_V, Q, K, V, rawScores, scaleFactor, scaledScores, attentionWeights, contextVectors, outputs };
    }

    function precompute(sentence, embedDim, numHeads) {
        const tokens = sentence.trim().split(/\s+/).filter(t => t.length > 0);
        const N = tokens.length;
        const d = embedDim;

        const rng = mulberry32(hashCode(sentence + d));

        const embeddings = [];
        for (let i = 0; i < N; i++) {
            embeddings.push(seededVector(rng, d));
        }

        const heads = [];
        for (let h = 0; h < numHeads; h++) {
            const headRng = mulberry32(hashCode(sentence + d + '_head_' + h));
            heads.push(precomputeHead(embeddings, d, headRng));
        }

        const head0 = heads[0];
        return {
            tokens, embeddings,
            W_Q: head0.W_Q, W_K: head0.W_K, W_V: head0.W_V,
            Q: head0.Q, K: head0.K, V: head0.V,
            rawScores: head0.rawScores, scaleFactor: head0.scaleFactor,
            scaledScores: head0.scaledScores, attentionWeights: head0.attentionWeights,
            contextVectors: head0.contextVectors, outputs: head0.outputs,
            heads: heads,
        };
    }

    // ============================================
    // Theme-aware color helper
    // ============================================
    function getThemeColors() {
        const s = getComputedStyle(document.documentElement);
        const g = name => s.getPropertyValue(name).trim();
        return {
            qBg:          g('--attn-q-bg'),
            qBorder:      g('--attn-q-border'),
            qFill:        g('--attn-q-fill'),
            qText:        g('--attn-q-text'),
            kBg:          g('--attn-k-bg'),
            kBorder:      g('--attn-k-border'),
            kFill:        g('--attn-k-fill'),
            kText:        g('--attn-k-text'),
            vBg:          g('--attn-v-bg'),
            vBorder:      g('--attn-v-border'),
            vFill:        g('--attn-v-fill'),
            vText:        g('--attn-v-text'),
            tokenBg:      g('--attn-token-bg'),
            tokenBorder:  g('--attn-token-border'),
            tokenText:    g('--attn-token-text'),
            embedPos:     g('--attn-embed-pos'),
            embedNeg:     g('--attn-embed-neg'),
            cellBg:       g('--attn-cell-bg'),
            cellBorder:   g('--attn-cell-border'),
            cellText:     g('--attn-cell-text'),
            activeGlow:   g('--attn-active-glow'),
            activeBorder: g('--attn-active-border'),
            heatCool:     g('--attn-heat-cool'),
            heatMid:      g('--attn-heat-mid'),
            heatWarm:     g('--attn-heat-warm'),
            outputFill:   g('--attn-output-fill'),
            outputBorder: g('--attn-output-border'),
            outputText:   g('--attn-output-text'),
            sectionTitle: g('--attn-section-title'),
            labelColor:   g('--attn-label-color'),
            canvasText:   g('--attn-canvas-text'),
            arrowColor:   g('--attn-arrow-color'),
            arrowGlow:    g('--attn-arrow-glow'),
            dashedLine:   g('--attn-dashed-line'),
            divider:      g('--attn-divider'),
            scaleLabel:   g('--attn-scale-label'),
            canvasBg:     g('--viz-canvas-bg'),
            textMuted:    g('--viz-text-muted'),
        };
    }

    // ============================================
    // Color interpolation for heatmap
    // ============================================
    function parseColor(str) {
        if (str.startsWith('#')) {
            const hex = str.slice(1);
            if (hex.length === 3) {
                return [parseInt(hex[0]+hex[0],16), parseInt(hex[1]+hex[1],16), parseInt(hex[2]+hex[2],16)];
            }
            return [parseInt(hex.slice(0,2),16), parseInt(hex.slice(2,4),16), parseInt(hex.slice(4,6),16)];
        }
        const m = str.match(/(\d+)/g);
        if (m && m.length >= 3) return [+m[0], +m[1], +m[2]];
        return [128, 128, 128];
    }

    function interpolateHeatColor(t, coolStr, midStr, warmStr) {
        const cool = parseColor(coolStr);
        const mid  = parseColor(midStr);
        const warm = parseColor(warmStr);
        let r, g, b;
        if (t <= 0.5) {
            const f = t * 2;
            r = cool[0] + (mid[0] - cool[0]) * f;
            g = cool[1] + (mid[1] - cool[1]) * f;
            b = cool[2] + (mid[2] - cool[2]) * f;
        } else {
            const f = (t - 0.5) * 2;
            r = mid[0] + (warm[0] - mid[0]) * f;
            g = mid[1] + (warm[1] - mid[1]) * f;
            b = mid[2] + (warm[2] - mid[2]) * f;
        }
        return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
    }

    // ============================================
    // Stagger alpha helper
    // ============================================
    function staggerAlpha(progress, index, total) {
        if (total <= 1) return progress;
        const stagger = 0.4;
        const itemStart = (index / total) * stagger;
        const itemEnd = itemStart + (1 - stagger);
        if (progress <= itemStart) return 0;
        if (progress >= itemEnd) return 1;
        return (progress - itemStart) / (itemEnd - itemStart);
    }

    // ============================================
    // Head color palettes
    // ============================================
    const HEAD_COLORS_LIGHT = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8','#ffbb78'];
    const HEAD_COLORS_DARK  = ['#83a598','#fe8019','#b8bb26','#fb4934','#d3869b','#8ec07c','#fabd2f','#a89984','#d79921','#689d6a','#458588','#cc241d'];

    function getHeadColors() {
        const theme = document.documentElement.getAttribute('data-theme');
        return theme === 'gruvbox-dark' ? HEAD_COLORS_DARK : HEAD_COLORS_LIGHT;
    }

    // ============================================
    // Main Visualizer Class (D3/SVG)
    // ============================================
    let clamp;

    class AttentionVisualizer {
        constructor() {
            this.container = d3.select('#attention-svg-container');
            this.svg = this.container.append('svg')
                .attr('viewBox', `0 0 ${CANVAS_W} 700`)
                .attr('preserveAspectRatio', 'xMidYMid meet');

            // Background rect
            this.bgRect = this.svg.append('rect')
                .attr('class', 'svg-bg')
                .attr('width', CANVAS_W)
                .attr('height', 700)
                .attr('fill', 'var(--viz-canvas-bg)');

            // SVG filter definitions for glow effects
            const defs = this.svg.append('defs');

            // Q (query) glow filter
            const filterQ = defs.append('filter')
                .attr('id', 'token-glow-q')
                .attr('x', '-50%').attr('y', '-50%')
                .attr('width', '200%').attr('height', '200%');
            filterQ.append('feGaussianBlur')
                .attr('stdDeviation', '3')
                .attr('result', 'blur');
            filterQ.append('feMerge')
                .selectAll('feMergeNode')
                .data(['blur', 'SourceGraphic'])
                .join('feMergeNode')
                .attr('in', d => d);

            // K (key) glow filter
            const filterK = defs.append('filter')
                .attr('id', 'token-glow-k')
                .attr('x', '-50%').attr('y', '-50%')
                .attr('width', '200%').attr('height', '200%');
            filterK.append('feGaussianBlur')
                .attr('stdDeviation', '3')
                .attr('result', 'blur');
            filterK.append('feMerge')
                .selectAll('feMergeNode')
                .data(['blur', 'SourceGraphic'])
                .join('feMergeNode')
                .attr('in', d => d);

            // Zone groups for layering
            this.gB = this.svg.append('g').attr('class', 'zone-b');
            this.gC = this.svg.append('g').attr('class', 'zone-c');

            // State
            this.sentence = PRESETS['default'];
            this.embedDim = 64;
            this.speed = 5;
            this.data = null;

            // Data mode
            this.dataMode = 'gpt2';  // 'synthetic' | 'bert' | 'gpt2'
            this.modelData = null;        // loaded JSON (BERT or GPT-2)
            this.modelLayer = 0;          // selected layer (0-11)
            this.modelHead = 0;           // selected head for single-head view (0-11)

            // Animation state machine
            this.phase = 'IDLE';
            this.phaseProgress = 0;
            this.phaseStartTime = 0;
            this.isProcessing = false;
            this.d3Timer = null;

            // Equation bar state — Set allows multiple chips expanded at once
            this.expandedTerms = new Set();
            this.eqRightMode = 'matrix';  // 'matrix' (N×d output) | 'bertviz' (N×N attention heatmap)

            // Head-lines view state
            this.numHeads = 12;
            this.hoveredToken = null;
            this.hoveredSide = null;
            this.visibleHeads = new Array(4).fill(true);

            // View mode: 'lines' (BertViz-style) or 'heatmap' (grid)
            this.zoneViewMode = 'lines';
            this.selectedCell = null;  // {i, j} for highlighted cell in heatmap

            // Layer comparison mode
            this.layerCompareMode = false;

            // Layer animation state
            this.layerAnimating = false;
            this.layerAnimTimer = null;
            this.layerAnimProgress = 0;
            this.layerAnimStartLayer = 0;

            // DOM elements
            this.sentenceSelect = document.getElementById('sentence-select');
            this.customSentenceRow = document.getElementById('custom-sentence-row');
            this.customSentenceInput = document.getElementById('custom-sentence');
            this.customSentenceHint = document.getElementById('custom-sentence-hint');
            this.embedDimValue = document.getElementById('embed-dim-value');
            this.btnDimDown = document.getElementById('btn-dim-down');
            this.btnDimUp = document.getElementById('btn-dim-up');
            this.btnRun = document.getElementById('btn-run');
            this.btnStep = document.getElementById('btn-step');
            this.btnReset = document.getElementById('btn-reset');
            this.speedSlider = document.getElementById('speed-slider');
            this.speedValueEl = document.getElementById('speed-value');

            this.numHeadsValue = document.getElementById('num-heads-value');
            this.btnHeadsDown = document.getElementById('btn-heads-down');
            this.btnHeadsUp = document.getElementById('btn-heads-up');
            this.headColorBar = document.getElementById('head-color-bar');

            // View mode toggle
            this.viewModeBar = document.getElementById('view-mode-bar');
            this.btnViewLines = document.getElementById('btn-view-lines');
            this.btnViewHeatmap = document.getElementById('btn-view-heatmap');
            this.btnLayerCompare = document.getElementById('btn-layer-compare');

            // Data mode controls
            this.btnDataSynthetic = document.getElementById('btn-data-synthetic');
            this.btnDataBert = document.getElementById('btn-data-bert');
            this.btnDataGpt2 = document.getElementById('btn-data-gpt2');
            this.dataBtns = [this.btnDataSynthetic, this.btnDataBert, this.btnDataGpt2];
            this.layerSelectRow = document.getElementById('layer-select-row');
            this.layerValue = document.getElementById('layer-value');
            this.btnLayerDown = document.getElementById('btn-layer-down');
            this.btnLayerUp = document.getElementById('btn-layer-up');
            this.dimStepper = this.btnDimDown.closest('.attn-stepper');
            this.headsStepper = this.btnHeadsDown.closest('.attn-stepper');

            // Layer animation controls
            this.btnLayerPlay = document.getElementById('btn-layer-play');
            this.layerAnimIndicator = document.getElementById('layer-anim-indicator');

            this.bindEvents();
            this.buildHeadColorBar();
            this.init();
        }

        async init() {
            // Default to BERT mode — load data before first render
            try {
                await this.loadModelData('gpt2', 'default');
                this.updateModelModeUI();
            } catch (e) {
                console.error('Failed to load BERT data, falling back to synthetic:', e);
                this.dataMode = 'synthetic';
                this.embedDim = 4;
                this.numHeads = 4;
                this.updateModelModeUI();
            }
            this.reset();
        }

        bindEvents() {
            this.btnRun.addEventListener('click', () => this.runAll());
            this.btnStep.addEventListener('click', () => this.stepOnce());
            this.btnReset.addEventListener('click', () => this.reset());

            this.speedSlider.addEventListener('input', () => {
                this.speed = parseInt(this.speedSlider.value);
                this.speedValueEl.textContent = this.speed;
            });

            this.sentenceSelect.addEventListener('change', async () => {
                if (this.sentenceSelect.value === 'custom') {
                    this.customSentenceRow.classList.add('visible');
                    this.customSentenceHint.classList.add('visible');
                    // Custom sentences fall back to synthetic mode
                    if (this.dataMode !== 'synthetic') {
                        this.dataMode = 'synthetic';
                        this.btnDataSynthetic.classList.add('active');
                        this.btnDataBert.classList.remove('active');
                        this.btnDataGpt2.classList.remove('active');
                        this.updateModelModeUI();
                    }
                } else {
                    this.customSentenceRow.classList.remove('visible');
                    this.customSentenceHint.classList.remove('visible');
                    if (this.dataMode !== 'synthetic') {
                        try {
                            await this.loadModelData(this.dataMode, this.sentenceSelect.value);
                        } catch (e) {
                            console.error('Failed to load model data:', e);
                        }
                    }
                }
                this.reset();
            });

            this.customSentenceInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') this.reset();
            });

            this.btnDimDown.addEventListener('click', () => {
                const idx = DIM_OPTIONS.indexOf(this.embedDim);
                if (idx > 0) {
                    this.embedDim = DIM_OPTIONS[idx - 1];
                    this.embedDimValue.textContent = this.embedDim;
                    this.reset();
                }
            });

            this.btnDimUp.addEventListener('click', () => {
                const idx = DIM_OPTIONS.indexOf(this.embedDim);
                if (idx < DIM_OPTIONS.length - 1) {
                    this.embedDim = DIM_OPTIONS[idx + 1];
                    this.embedDimValue.textContent = this.embedDim;
                    this.reset();
                }
            });

            this.btnHeadsDown.addEventListener('click', () => {
                if (this.numHeads > 1) {
                    this.numHeads--;
                    this.numHeadsValue.textContent = this.numHeads;
                    this.reset();
                }
            });

            this.btnHeadsUp.addEventListener('click', () => {
                if (this.numHeads < 8) {
                    this.numHeads++;
                    this.numHeadsValue.textContent = this.numHeads;
                    this.reset();
                }
            });

            // Data mode toggle — shared handler
            const switchToMode = async (mode, btn) => {
                if (this.dataMode === mode) return;
                this.dataBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                if (mode === 'synthetic') {
                    this.dataMode = 'synthetic';
                    this.embedDim = 4;
                    this.numHeads = 4;
                    this.updateModelModeUI();
                    this.reset();
                } else {
                    this.dataMode = mode;
                    this.updateModelModeUI();
                    const sentenceKey = this.sentenceSelect.value === 'custom' ? 'default' : this.sentenceSelect.value;
                    try {
                        await this.loadModelData(mode, sentenceKey);
                        this.reset();
                    } catch (e) {
                        console.error(`Failed to load ${mode} data:`, e);
                        this.dataMode = 'synthetic';
                        this.dataBtns.forEach(b => b.classList.remove('active'));
                        this.btnDataSynthetic.classList.add('active');
                        this.updateModelModeUI();
                        this.reset();
                    }
                }
            };

            this.btnDataSynthetic.addEventListener('click', () => switchToMode('synthetic', this.btnDataSynthetic));
            this.btnDataBert.addEventListener('click', () => switchToMode('bert', this.btnDataBert));
            this.btnDataGpt2.addEventListener('click', () => switchToMode('gpt2', this.btnDataGpt2));

            // Layer stepper
            this.btnLayerDown.addEventListener('click', () => {
                if (this.modelLayer > 0) {
                    this.modelLayer--;
                    this.layerValue.textContent = this.modelLayer;
                    this.reset();
                }
            });

            this.btnLayerUp.addEventListener('click', () => {
                if (this.modelLayer < 11) {
                    this.modelLayer++;
                    this.layerValue.textContent = this.modelLayer;
                    this.reset();
                }
            });

            document.addEventListener('themechange', () => {
                this.buildHeadColorBar();
                this.draw();
            });

            // View mode toggle events
            this.btnViewLines.addEventListener('click', () => {
                if (this.zoneViewMode === 'lines') return;
                this.zoneViewMode = 'lines';
                this.btnViewLines.classList.add('active');
                this.btnViewHeatmap.classList.remove('active');
                this.selectedCell = null;
                this.draw();
            });

            this.btnViewHeatmap.addEventListener('click', () => {
                if (this.zoneViewMode === 'heatmap') return;
                this.zoneViewMode = 'heatmap';
                this.btnViewHeatmap.classList.add('active');
                this.btnViewLines.classList.remove('active');
                this.layerCompareMode = false;
                this.btnLayerCompare.classList.remove('active');
                this.hoveredToken = null;
                this.hoveredSide = null;
                this.draw();
            });

            // Layer comparison toggle
            this.btnLayerCompare.addEventListener('click', () => {
                this.layerCompareMode = !this.layerCompareMode;
                if (this.layerCompareMode) {
                    this.btnLayerCompare.classList.add('active');
                } else {
                    this.btnLayerCompare.classList.remove('active');
                }
                this.computeLayout();
                this.draw();
            });

            // Layer animation play/pause button
            if (this.btnLayerPlay) {
                this.btnLayerPlay.addEventListener('click', () => this.toggleLayerAnimation());
            }

        }

        // ============================================
        // Sentence helpers
        // ============================================
        getSentence() {
            const sel = this.sentenceSelect.value;
            if (sel === 'custom') {
                const text = this.customSentenceInput.value.trim();
                if (!text) return PRESETS['default'];
                const words = text.split(/\s+/).slice(0, 5);
                return words.join(' ');
            }
            return PRESETS[sel] || PRESETS['default'];
        }

        // ============================================
        // Model data loading & mode management
        // ============================================
        async loadModelData(modelPrefix, sentenceKey) {
            const url = `static/data/attention/${modelPrefix}-${sentenceKey}.json`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed to load ${url}`);
            this.modelData = await resp.json();
        }

        // ============================================
        // Layer Animation Methods
        // ============================================
        toggleLayerAnimation() {
            if (this.layerAnimating) {
                this.stopLayerAnimation();
            } else {
                this.startLayerAnimation();
            }
        }

        startLayerAnimation() {
            if (this.layerAnimating) return;
            if (!this.modelData) return;

            this.layerAnimating = true;
            this.layerAnimStartLayer = this.modelLayer;
            this.layerAnimProgress = 0;

            // Update button icon
            if (this.btnLayerPlay) {
                this.btnLayerPlay.innerHTML = '<i class="fa fa-pause"></i>';
                this.btnLayerPlay.classList.add('active');
            }

            const self = this;
            const layerDuration = 800; // ms per layer transition
            let lastTime = performance.now();

            this.layerAnimTimer = d3.timer(() => {
                if (!self.layerAnimating) return true;

                const now = performance.now();
                const delta = now - lastTime;
                lastTime = now;

                // Advance progress
                self.layerAnimProgress += delta / layerDuration;

                // When progress reaches 1, advance to next layer
                if (self.layerAnimProgress >= 1) {
                    self.layerAnimProgress = 0;
                    self.modelLayer++;

                    // Update layer display
                    self.layerValue.textContent = self.modelLayer;

                    // Stop at layer 11
                    if (self.modelLayer >= 11) {
                        self.modelLayer = 11;
                        self.stopLayerAnimation();
                        return true;
                    }
                }

                // Update indicator
                self.updateLayerAnimIndicator();
                self.draw();
            });
        }

        stopLayerAnimation() {
            if (this.layerAnimTimer) {
                this.layerAnimTimer.stop();
                this.layerAnimTimer = null;
            }
            this.layerAnimating = false;
            this.layerAnimProgress = 0;

            // Update button icon
            if (this.btnLayerPlay) {
                this.btnLayerPlay.innerHTML = '<i class="fa fa-play"></i>';
                this.btnLayerPlay.classList.remove('active');
            }

            // Hide indicator
            if (this.layerAnimIndicator) {
                this.layerAnimIndicator.textContent = '';
            }
        }

        updateLayerAnimIndicator() {
            if (this.layerAnimIndicator && this.layerAnimating) {
                const displayLayer = this.modelLayer + this.easeInOutCubic(this.layerAnimProgress);
                this.layerAnimIndicator.textContent = `L${displayLayer.toFixed(1)}`;
            }
        }

        easeInOutCubic(t) {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }

        interpolateLayerWeights(progress) {
            if (!this.modelData) return null;
            const fromLayer = this.modelLayer;
            const toLayer = Math.min(fromLayer + 1, 11);
            if (fromLayer === toLayer) return null;

            const t = this.easeInOutCubic(progress);
            const fromData = this.modelData.layers[fromLayer];
            const toData = this.modelData.layers[toLayer];
            const numHeads = fromData.heads.length;
            const N = this.modelData.tokens.length;

            const interpolatedHeads = [];
            for (let h = 0; h < numHeads; h++) {
                const fromWeights = fromData.heads[h].attention_weights;
                const toWeights = toData.heads[h].attention_weights;
                const interpWeights = [];

                for (let i = 0; i < N; i++) {
                    const row = [];
                    for (let j = 0; j < N; j++) {
                        row.push(fromWeights[i][j] * (1 - t) + toWeights[i][j] * t);
                    }
                    interpWeights.push(row);
                }

                interpolatedHeads.push({
                    attentionWeights: interpWeights,
                    Q: fromData.heads[h].Q,
                    K: fromData.heads[h].K,
                    V: fromData.heads[h].V,
                    outputs: fromData.heads[h].output,
                });
            }

            return interpolatedHeads;
        }

        getActiveData() {
            if (this.dataMode !== 'synthetic' && this.modelData) {
                const layerData = this.modelData.layers[this.modelLayer];
                const headData = layerData.heads[this.modelHead];
                const N = this.modelData.tokens.length;
                const d = headData.Q[0].length;  // 64

                // Compute raw scores: Q @ K.T
                const rawScores = [];
                for (let i = 0; i < N; i++) {
                    const row = [];
                    for (let j = 0; j < N; j++) {
                        let sum = 0;
                        for (let k = 0; k < d; k++) {
                            sum += headData.Q[i][k] * headData.K[j][k];
                        }
                        row.push(sum);
                    }
                    rawScores.push(row);
                }

                const scaleFactor = Math.sqrt(d);
                const scaledScores = rawScores.map(r => r.map(s => s / scaleFactor));

                // Helper: compute per-head scores
                function computeHeadScores(hData) {
                    const hd = hData.Q[0].length;
                    const rs = [];
                    for (let i = 0; i < N; i++) {
                        const row = [];
                        for (let j = 0; j < N; j++) {
                            let s = 0;
                            for (let k = 0; k < hd; k++) s += hData.Q[i][k] * hData.K[j][k];
                            row.push(s);
                        }
                        rs.push(row);
                    }
                    const sf = Math.sqrt(hd);
                    return { rawScores: rs, scaleFactor: sf, scaledScores: rs.map(r => r.map(v => v / sf)) };
                }

                // Get heads - use interpolated weights if animating
                let heads;
                if (this.layerAnimating && this.layerAnimProgress > 0) {
                    const interpolated = this.interpolateLayerWeights(this.layerAnimProgress);
                    heads = interpolated || layerData.heads.map(h => {
                        const scores = computeHeadScores(h);
                        return {
                            attentionWeights: h.attention_weights,
                            Q: h.Q, K: h.K, V: h.V, outputs: h.output,
                            rawScores: scores.rawScores, scaleFactor: scores.scaleFactor, scaledScores: scores.scaledScores,
                        };
                    });
                } else {
                    heads = layerData.heads.map(h => {
                        const scores = computeHeadScores(h);
                        return {
                            attentionWeights: h.attention_weights,
                            Q: h.Q, K: h.K, V: h.V, outputs: h.output,
                            rawScores: scores.rawScores, scaleFactor: scores.scaleFactor, scaledScores: scores.scaledScores,
                        };
                    });
                }

                return {
                    tokens: this.modelData.tokens,
                    embeddings: this.modelData.embeddings_pca,
                    Q: headData.Q, K: headData.K, V: headData.V,
                    rawScores, scaleFactor, scaledScores,
                    attentionWeights: headData.attention_weights,
                    outputs: headData.output,
                    heads: heads,
                };
            }
            return this.data;  // synthetic
        }

        updateModelModeUI() {
            const isModel = this.dataMode !== 'synthetic';
            // Show/hide layer selector
            this.layerSelectRow.style.display = isModel ? '' : 'none';
            // Disable/enable dim and heads steppers
            this.dimStepper.classList.toggle('disabled', isModel);
            this.headsStepper.classList.toggle('disabled', isModel);
            // Update displayed values
            this.embedDimValue.textContent = this.embedDim;
            this.numHeadsValue.textContent = this.numHeads;
        }

        // ============================================
        // Dynamic layout computation
        // ============================================
        computeLayout() {
            const activeData = this.getActiveData();
            const N = activeData ? activeData.tokens.length : 4;
            const isModel = this.dataMode !== 'synthetic' && this.modelData;
            const canvasW = isModel ? 1000 : CANVAS_W;

            const scoreCellSize = Math.min(32, Math.floor((canvasW - 120) / (N + 1)));
            const smallCellSize = Math.min(22, Math.floor((canvasW - 160) / (N + 1)));

            let y = 15;

            // Inline matrix cell sizes for expanded chips
            // In model mode use heatmap cells; in synthetic use text cells
            const eqScale = isModel ? 2.25 : 1;  // scale factor for equation
            const matCellW = isModel ? 8 : 30;
            const matCellH = isModel ? 8 : 16;
            const matPad = 6;
            const defaultChipH = Math.round(42 * eqScale);  // mathSize(18) + chipPadY(4)*2 + 16 for dim label, scaled
            const fracGap = Math.round(4 * eqScale);

            // Matrix dimensions for expanded chips
            const matH = N * matCellH + matPad * 2;  // height of an N-row matrix
            const et = this.expandedTerms || new Set();

            const arrowH = Math.round(20 * eqScale);
            const branchH = Math.round(30 * eqScale);
            const projRowH = defaultChipH;
            const embedContentH = et.has('E') ? matH : defaultChipH;

            // --- Vertical dataflow stages ---
            // Stage 1: Tokens chip
            const stage1H = defaultChipH;
            // Stage 2: Arrow tokens → E
            const stage2H = arrowH;
            // Stage 3: Embedding chip (may expand to matrix)
            const stage3H = embedContentH;
            // Stage 4: Branching from E to projections
            const stage4H = branchH;
            // Stage 5: Projection row [E·Wq][E·Wk][E·Wv] (no labels)
            const stage5H = projRowH;
            // Softmax box content height (shared computation for both paths)
            let softmaxBoxContentH;
            if (et.has('softmax') || et.has('scaled')) {
                softmaxBoxContentH = matH;
            } else if (et.has('QKT') || et.has('Q') || et.has('K')) {
                softmaxBoxContentH = matH + fracGap * 2 + defaultChipH;
            } else {
                softmaxBoxContentH = defaultChipH + fracGap * 2 + defaultChipH;
            }
            const softmaxBoxPadY = Math.round(16 * eqScale);
            const softmaxBoxH = softmaxBoxContentH + softmaxBoxPadY * 2;
            const ctxUnderlayPadLayout = Math.round(8 * eqScale);
            const ctxLabelSpace = Math.round(18 * eqScale);
            const vContentH = et.has('V') ? matH : defaultChipH;
            const equationRowH = Math.max(softmaxBoxH, vContentH) + ctxUnderlayPadLayout * 2 + ctxLabelSpace;

            const multiHead = this.numHeads > 1;

            // Head gallery: when multiHead, the carousel contains the full per-head equation
            // Active card = full equation height; inactive cards = compact thumbnails
            const galleryTitleH = Math.round(20 * eqScale);  // "Head 3 / 12" label
            const galleryNavH = Math.round(20 * eqScale);    // nav dots + arrows
            const galleryArrowFromProjH = arrowH;             // arrows from projections into the card
            const headCardEquationH = galleryTitleH + galleryArrowFromProjH + equationRowH;
            // Thumbnail card for inactive heads
            const thumbCardH = Math.round(N * 8 + 30);
            const stageGalleryH = multiHead ? headCardEquationH + galleryNavH : 0;

            // Stage 6: Arrows from projections down to softmax box / V (only in single-head mode)
            const stage6H = multiHead ? 0 : arrowH;
            // Stage 7: Softmax equation row (only in single-head mode; in multiHead it's inside the gallery)
            const stage7H = multiHead ? 0 : equationRowH;
            // Stage 8: Arrow from equation/gallery to heatmap
            const stage8H = arrowH;
            // Stage 9: Heatmap/output panel
            const heatmapH = Math.round(N * 45 + 60);
            const stage9H = heatmapH;

            const preGallery = stage1H + stage2H + stage3H + stage4H + stage5H;
            const postGallery = stage6H + stage7H + stage8H + 20 + stage9H;
            const flowH = preGallery + stageGalleryH + postGallery;
            const B_eq = {
                y: y, h: flowH, matCellW, matCellH,
                multiHead,
                // Stage Y positions (computed cumulatively)
                stage1Y: y,
                stage2Y: y + stage1H,
                stage3Y: y + stage1H + stage2H,
                stage4Y: y + stage1H + stage2H + stage3H,
                stage5Y: y + stage1H + stage2H + stage3H + stage4H,
                stageGalleryY: y + preGallery,
                stageGalleryH: stageGalleryH,
                stage6Y: y + preGallery + stageGalleryH,
                stage7Y: y + preGallery + stageGalleryH + stage6H,
                stage8Y: y + preGallery + stageGalleryH + stage6H + stage7H,
                stage9Y: y + preGallery + stageGalleryH + stage6H + stage7H + stage8H + 20,
                // Stage heights for drawing
                stage1H, stage2H, stage3H, stage4H, stage5H, stage6H, stage7H, stage8H, stage9H,
                softmaxBoxContentH, softmaxBoxPadY, softmaxBoxH, vContentH, equationRowH,
                defaultChipH, arrowH, branchH,
                headCardEquationH, thumbCardH, galleryTitleH, galleryNavH, galleryArrowFromProjH,
            };
            y += B_eq.h + ZONE_GAP;

            // Zone C height depends on view mode
            // In model mode: only show Zone C for layer comparison or sentence comparison
            // In synthetic mode: show for lines or heatmap view
            let zoneCHeight;
            if (isModel) {
                if (this.layerCompareMode) {
                    zoneCHeight = 200;  // layer comparison grid
                } else {
                    zoneCHeight = 0;  // no Zone C needed - BertViz is inline
                }
            } else {
                zoneCHeight = Math.max(155, N * 22 + 35);  // synthetic mode
            }
            const C = { y: y, h: zoneCHeight };
            if (zoneCHeight > 0) y += C.h + ZONE_GAP;

            const canvasH = y + 15;

            this.layout = {
                B_eq, C,
                scoreCellSize, smallCellSize, canvasH, canvasW, isModel,
            };

            // Update SVG viewBox to fit computed layout
            this.svg.attr('viewBox', `0 0 ${canvasW} ${canvasH}`);
            this.bgRect.attr('width', canvasW).attr('height', canvasH);
        }

        // ============================================
        // Reset
        // ============================================
        reset() {
            this.stopAnimation();
            this.sentence = this.getSentence();

            if (this.dataMode !== 'synthetic' && this.modelData) {
                // In model mode, use real data dimensions
                const headData = this.modelData.layers[0].heads[0];
                this.embedDim = headData.Q[0].length;
                this.numHeads = this.modelData.layers[0].heads.length;
                this.data = precompute(this.sentence, 4, 4);  // minimal synthetic fallback
            } else {
                this.data = precompute(this.sentence, this.embedDim, this.numHeads);
            }

            // In model mode, show complete view immediately; in synthetic, start at IDLE
            this.phase = (this.dataMode !== 'synthetic' && this.modelData) ? 'COMPLETE' : 'IDLE';
            this.phaseProgress = 1;
            this.isProcessing = false;
            this.expandedTerms = new Set();

            // Stop layer animation if running
            this.stopLayerAnimation();

            this.visibleHeads = new Array(this.numHeads).fill(true);
            this.hoveredToken = null;
            this.hoveredSide = null;
            this.buildHeadColorBar();

            this.computeLayout();
            this.updateMetrics();
            this.draw();
            this.updateRunButtonState();
        }

        // ============================================
        // Phase management
        // ============================================
        getPhaseDuration() {
            const base = Math.max(200, 1400 - this.speed * 120);
            const mult = PHASE_DURATION_MULT[this.phase] || 1;
            return base * mult;
        }

        getNextPhase() {
            const idx = PHASES.indexOf(this.phase);
            if (idx < 0 || idx >= PHASES.length - 1) return 'COMPLETE';
            return PHASES[idx + 1];
        }

        advancePhase() {
            const next = this.getNextPhase();
            if (next === this.phase && next === 'COMPLETE') return false;

            this.phase = next;
            this.phaseProgress = 0;
            this.phaseStartTime = performance.now();

            // Auto-expand the corresponding equation term during animation
            if (PHASE_TERM_MAP[this.phase]) {
                this.expandedTerms.clear();
                this.expandedTerms.add(PHASE_TERM_MAP[this.phase]);
                this.computeLayout();
            }
            if (['SHOW_OUTPUT','COMPLETE'].includes(this.phase)) {
                this.expandedTerms.clear();
                this.computeLayout();
            }

            this.updateMetrics();
            return true;
        }

        // ============================================
        // Step / Run
        // ============================================
        stepOnce() {
            if (this.phase === 'COMPLETE') return;
            this.stopAnimation();
            const advanced = this.advancePhase();
            if (advanced) {
                this.phaseProgress = 1;
                this.updateMetrics();
                this.draw();
            }
        }

        runAll() {
            if (this.isProcessing) return;
            if (this.phase === 'COMPLETE') {
                this.phase = 'IDLE';
                this.phaseProgress = 1;
            }

            this.isProcessing = true;
            this.btnRun.disabled = true;

            if (this.phase === 'IDLE') {
                this.advancePhase();
            }
            this.phaseStartTime = performance.now();

            this.d3Timer = d3.timer((elapsed) => {
                if (!this.isProcessing) return;

                const now = performance.now();
                const phaseElapsed = now - this.phaseStartTime;
                const duration = this.getPhaseDuration();
                this.phaseProgress = clamp(phaseElapsed / duration, 0, 1);

                this.draw();

                if (this.phaseProgress >= 1) {
                    if (this.phase === 'COMPLETE') {
                        this.stopAnimation();
                        return;
                    }
                    const advanced = this.advancePhase();
                    if (!advanced) {
                        this.stopAnimation();
                        return;
                    }
                }
            });
        }

        stopAnimation() {
            if (this.d3Timer) {
                this.d3Timer.stop();
                this.d3Timer = null;
            }
            this.isProcessing = false;
            this.btnRun.disabled = false;
            this.updateRunButtonState();
        }

        // ============================================
        // Metrics
        // ============================================
        updateMetrics() {
            const phaseNames = {
                'IDLE': 'Idle',
                'SHOW_TOKENS': 'Show Tokens',
                'SHOW_EMBEDDINGS': 'Show Embeddings',
                'PROJECT_Q': 'Project Q',
                'PROJECT_K': 'Project K',
                'PROJECT_V': 'Project V',
                'COMPUTE_SCORES': 'Compute Scores',
                'SCALE_SCORES': 'Scale Scores',
                'APPLY_SOFTMAX': 'Apply Softmax',
                'SHOW_OUTPUT': 'Show Output',
                'COMPLETE': 'Complete',
            };

            document.getElementById('metric-phase').textContent = phaseNames[this.phase] || this.phase;
            document.getElementById('metric-num-heads').textContent = this.numHeads;

            const activeData = this.getActiveData();
            if (activeData) {
                const N = activeData.tokens.length;
                document.getElementById('metric-tokens').textContent = N;
                document.getElementById('metric-embed-dim').textContent = this.embedDim;
                document.getElementById('metric-score-matrix').textContent = N + '\u00d7' + N;
                document.getElementById('metric-scale-factor').textContent =
                    '\u221a' + this.embedDim + ' = ' + activeData.scaleFactor.toFixed(2);

                const phaseIdx = PHASES.indexOf(this.phase);
                const softmaxIdx = PHASES.indexOf('APPLY_SOFTMAX');
                if (phaseIdx >= softmaxIdx && activeData.attentionWeights) {
                    let maxW = -Infinity, minW = Infinity;
                    for (const row of activeData.attentionWeights) {
                        for (const w of row) {
                            if (w > maxW) maxW = w;
                            if (w < minW) minW = w;
                        }
                    }
                    document.getElementById('metric-max-weight').textContent = maxW.toFixed(4);
                    document.getElementById('metric-min-weight').textContent = minW.toFixed(4);
                } else {
                    document.getElementById('metric-max-weight').textContent = '-';
                    document.getElementById('metric-min-weight').textContent = '-';
                }
            }
        }

        // ============================================
        // Drawing — clear and redraw all zones
        // ============================================
        draw() {
            const C = getThemeColors();

            this.bgRect.attr('fill', C.canvasBg);

            // Clear all zone groups
            this.gB.selectAll('*').remove();
            this.gC.selectAll('*').remove();

            this.drawZoneB(C);
            this.drawZoneC(C);
        }

        // ============================================
        // Zone B: Equation Bar + Expansion Panel
        // ============================================
        drawZoneB(C) {
            const phaseIdx = PHASES.indexOf(this.phase);
            const showTokensIdx = PHASES.indexOf('SHOW_TOKENS');

            if (phaseIdx < showTokensIdx) return;
            if (!this.getActiveData()) return;

            this.drawEquationBar(C);
        }

        // ---- Equation with inline matrix expansion (left) + BertViz (right) ----
        // Clicking a chip replaces it with the actual matrix values inline.
        // Hierarchy: QKT replaces numerator, scaled replaces fraction, softmax replaces softmax(…)
        drawEquationBar(C) {
            const g = this.gB;
            const phaseIdx = PHASES.indexOf(this.phase);
            const { B_eq } = this.layout;
            const canvasW = this.layout.canvasW;
            const isModel = this.layout.isModel;
            const zy = B_eq.y;
            const barH = B_eq.h;
            const self = this;
            const activeData = this.getActiveData();
            const N = activeData.tokens.length;
            const d = isModel ? activeData.Q[0].length : this.embedDim;
            const matCellW = B_eq.matCellW;
            const matCellH = B_eq.matCellH;
            const matPad = 6;
            const et = this.expandedTerms;

            // Determine effective mode for the softmax block (highest expanded level wins)
            const fracMode = et.has('softmax') ? 'softmax'
                           : et.has('scaled') ? 'scaled'
                           : et.has('QKT') ? 'QKT'
                           : 'normal';

            const SERIF = "'Georgia','Times New Roman','Times',serif";
            // Scale up equation in model mode to fill available space
            const scale = isModel ? 2.25 : 1;
            const mathSize = 18 * scale;
            const smallSize = 14 * scale;
            const charW = mathSize * 0.55;
            const smallCharW = smallSize * 0.55;
            const chipPadX = 8 * scale, chipPadY = 4 * scale;
            const defaultChipH = mathSize + chipPadY * 2 + 16 * scale;
            const fracGap = 4 * scale;
            const gap = 3 * scale;

            // --- Helpers ---
            function drawBrackets(parent, bx, by, bw, bh, color) {
                const cap = 4, inset = 1.5;
                parent.append('path')
                    .attr('d', `M${bx+cap+inset},${by+inset} L${bx+inset},${by+inset} L${bx+inset},${by+bh-inset} L${bx+cap+inset},${by+bh-inset}`)
                    .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.2);
                parent.append('path')
                    .attr('d', `M${bx+bw-cap-inset},${by+inset} L${bx+bw-inset},${by+inset} L${bx+bw-inset},${by+bh-inset} L${bx+bw-cap-inset},${by+bh-inset}`)
                    .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.2);
            }

            function drawMatrixGrid(parent, mx, my, data, rows, cols, color, decimals) {
                if (isModel) {
                    // Heatmap mode for model data: colored cells instead of text
                    let minVal = Infinity, maxVal = -Infinity;
                    for (let i = 0; i < rows; i++) {
                        for (let j = 0; j < cols; j++) {
                            const v = data[i][j];
                            if (v < minVal) minVal = v;
                            if (v > maxVal) maxVal = v;
                        }
                    }
                    const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 0.001);
                    for (let i = 0; i < rows; i++) {
                        for (let j = 0; j < cols; j++) {
                            const cx = mx + matPad + j * matCellW;
                            const cy = my + matPad + i * matCellH;
                            const v = data[i][j];
                            // Diverging scale: blue(negative) -> white(zero) -> red(positive)
                            const t = (v / absMax + 1) / 2;  // 0..1
                            const fillColor = interpolateHeatColor(t, '#4575b4', '#f7f7f7', '#d73027');
                            parent.append('rect')
                                .attr('x', cx).attr('y', cy)
                                .attr('width', matCellW).attr('height', matCellH)
                                .attr('fill', fillColor);
                        }
                    }
                } else {
                    // Text mode for synthetic
                    const dec = decimals || 1;
                    for (let i = 0; i < rows; i++) {
                        for (let j = 0; j < cols; j++) {
                            const cx = mx + matPad + j * matCellW;
                            const cy = my + matPad + i * matCellH;
                            parent.append('text')
                                .attr('x', cx + matCellW / 2).attr('y', cy + matCellH / 2)
                                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                                .attr('font-family', MONO).attr('font-size', 8)
                                .attr('fill', color)
                                .text(data[i][j].toFixed(dec));
                        }
                    }
                }
            }

            function makeChip(chipX, chipY, chipW, chipH, chip, opts) {
                const requiredPhase = TERM_PHASE_MAP[chip.id];
                const requiredIdx = PHASES.indexOf(requiredPhase);
                const isLocked = phaseIdx < requiredIdx;
                const isActive = et.has(chip.id);
                const color = isLocked ? C.textMuted : (chip.border || chip.color);
                const noBrackets = opts && opts.noBrackets;
                const isMatrix = opts && opts.isMatrix;

                // Use dedicated bg color if available, otherwise fall back to color+opacity
                const hasBg = !isLocked && chip.bg;
                const bgColor = hasBg ? chip.bg : chip.color;
                const bgOpacity = hasBg ? 1 : (isActive ? 0.18 : 0);

                const chipG = g.append('g')
                    .attr('class', 'eq-chip' + (isLocked ? ' locked' : '') + (isActive ? ' active' : ''));

                // White undercoat so semi-transparent bg colors aren't affected by what's behind
                chipG.append('rect').attr('class', 'eq-chip-undercoat')
                    .attr('x', chipX).attr('y', chipY).attr('width', chipW).attr('height', chipH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', C.canvasBg || '#ffffff');
                chipG.append('rect').attr('class', 'eq-chip-bg')
                    .attr('x', chipX).attr('y', chipY).attr('width', chipW).attr('height', chipH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', bgColor).attr('opacity', bgOpacity);

                chipG.append('rect').attr('class', 'eq-chip-border')
                    .attr('x', chipX).attr('y', chipY).attr('width', chipW).attr('height', chipH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', 'none').attr('stroke', color)
                    .attr('stroke-opacity', 1)
                    .attr('opacity', 1)
                    .attr('stroke-width', isActive ? 2.5 : 2);

                // Corner brackets removed — using plain borders only

                if (!isLocked && !(opts && opts.nonExpandable)) {
                    chipG.attr('cursor', 'pointer')
                        .on('click', function() {
                            if (et.has(chip.id)) {
                                et.delete(chip.id);
                            } else {
                                et.add(chip.id);
                            }
                            self.computeLayout();
                            self.draw();
                        });
                }
                return chipG;
            }

            function staticText(tx, ty, text, size, fill) {
                g.append('text')
                    .attr('x', tx).attr('y', ty)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                    .attr('font-family', SERIF).attr('font-size', size)
                    .attr('fill', fill || C.canvasText).text(text);
            }

            // Draw dimension label below an element
            function dimLabel(cx, bottomY, dimText) {
                g.append('text')
                    .attr('x', cx).attr('y', bottomY + 12 * scale)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
                    .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                    .attr('fill', C.textMuted).attr('opacity', 0.7)
                    .text(dimText);
            }
            // Draw dimension label inside a chip, below the variable name
            function chipDimLabel(parent, cx, cy, dimText) {
                parent.append('text')
                    .attr('x', cx).attr('y', cy)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
                    .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                    .attr('fill', C.textMuted).attr('opacity', 0.7)
                    .text(dimText);
            }

            // --- Chip definitions (color = text/border, bg = fill background, fill = accent) ---
            const chipTokens = { id: 'tokens', color: C.tokenText, border: C.tokenBorder };
            const chipE   = { id: 'E',       color: C.embedPos };
            const chipQ   = { id: 'Q',       color: C.qText,   bg: C.qBg,   border: C.qBorder,   fill: C.qFill };
            const chipK   = { id: 'K',       color: C.kText,   bg: C.kBg,   border: C.kBorder,   fill: C.kFill };
            const chipV   = { id: 'V',       color: C.vText,   bg: C.vBg,   border: C.vBorder,   fill: C.vFill };
            const chipQKT = { id: 'QKT',     color: C.scaleLabel };
            const chipSc  = { id: 'scaled',  color: C.scaleLabel };
            const chipSm  = { id: 'softmax', color: C.sectionTitle, bg: 'rgba(150,150,150,0.10)', border: C.sectionTitle };
            const chipCtx = { id: 'ctxVectors', color: C.sectionTitle, bg: 'rgba(150,150,150,0.18)', border: C.sectionTitle };
            const chipResE = { id: 'resE', color: C.embedPos };

            // --- Matrix dimensions ---
            function matDims(rows, cols) {
                return { w: cols * matCellW + matPad * 2, h: rows * matCellH + matPad * 2 };
            }
            const matNxN = matDims(N, N);
            const matNxD = matDims(N, d);

            // --- Chip text widths ---
            const smTextW = smallCharW * 7.5 + chipPadX * 2;
            const qTextW = charW * 1.8 + chipPadX * 2;
            const kTextW = charW * 2.8 + chipPadX * 2;
            const dotTextW = charW * 1.5;
            const scTextW = charW * 3.8 + chipPadX * 2;
            const vTextW = charW * 2 + chipPadX * 2;
            const lparenW = charW * 1.8;
            const rparenW = charW * 1.8;
            const eChipTextW = charW * 1.8 + chipPadX * 2;
            const attnScoresTextW = charW * 10 + chipPadX * 2;  // "Attn Scores" label

            // --- Vertical flow uses stage Y positions from computeLayout ---
            const tokenStr = activeData.tokens.join(' ');
            const tokChipW = Math.min(charW * (tokenStr.length + 1) + chipPadX * 2, 180 * scale);
            const arrowH = B_eq.arrowH;
            const branchH = B_eq.branchH;
            const showEMatrix = et.has('E');
            const eChipW = showEMatrix ? matNxD.w : eChipTextW;
            const eChipH = showEMatrix ? matNxD.h : defaultChipH;
            const availableW = canvasW;
            const flowCenterX = availableW / 2;

            // ============ STAGE 1: TOKENS CHIP ============
            const tokChipY = B_eq.stage1Y;
            const tokChipX = flowCenterX - tokChipW / 2;
            const tokChipG = makeChip(tokChipX, tokChipY, tokChipW, defaultChipH, chipTokens, { noBrackets: true, nonExpandable: true });
            const tokFontSize = Math.min(smallSize * 0.7, tokChipW / (tokenStr.length + 1) * 1.4);
            tokChipG.append('text')
                .attr('x', tokChipX + tokChipW / 2).attr('y', tokChipY + defaultChipH / 2 - 6 * scale)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                .attr('font-family', MONO).attr('font-size', tokFontSize)
                .attr('fill', phaseIdx < PHASES.indexOf('SHOW_TOKENS') ? C.textMuted : C.tokenText)
                .text(tokenStr);
            chipDimLabel(tokChipG, tokChipX + tokChipW / 2, tokChipY + defaultChipH / 2 + 7 * scale, `N=${N}`);

            // Helper: draw a filled triangular arrowhead at (ax, ay) pointing down
            const arrowTipH = 6 * scale;
            const arrowTipHalfW = 3.5 * scale;
            function arrowHead(ax, ay, color) {
                g.append('path')
                    .attr('d', `M${ax},${ay} L${ax - arrowTipHalfW},${ay - arrowTipH} L${ax + arrowTipHalfW},${ay - arrowTipH} Z`)
                    .attr('fill', color).attr('stroke', 'none');
            }
            // Helper: draw a vertical arrow line from y1 to y2, stopping short for arrowhead
            function arrowLine(ax, y1, y2, color, dashed) {
                g.append('line')
                    .attr('x1', ax).attr('y1', y1)
                    .attr('x2', ax).attr('y2', y2 - arrowTipH)
                    .attr('stroke', color).attr('stroke-width', 1.2 * scale)
                    .attr('stroke-dasharray', dashed ? `${3 * scale},${3 * scale}` : 'none');
                arrowHead(ax, y2, color);
            }
            // Helper: draw a curved arrow from (x1,y1) to (x2,y2) with bezier control points
            function curvedArrow(x1, y1, x2, y2, color, dashed) {
                const midY = y1 + (y2 - y1) * 0.5;
                g.append('path')
                    .attr('d', `M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2 - arrowTipH}`)
                    .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.2 * scale)
                    .attr('stroke-dasharray', dashed ? `${3 * scale},${3 * scale}` : 'none');
                arrowHead(x2, y2, color);
            }

            // ============ STAGE 2: ARROW tokens → E ============
            const arrow1Color = phaseIdx >= PHASES.indexOf('SHOW_EMBEDDINGS') ? C.canvasText : C.textMuted;
            arrowLine(flowCenterX, B_eq.stage2Y, B_eq.stage2Y + arrowH, arrow1Color);

            // ============ STAGE 3: EMBEDDING CHIP [E] ============
            const eChipY = B_eq.stage3Y;
            const eChipX = flowCenterX - eChipW / 2;
            if (showEMatrix) {
                const eChipG = makeChip(eChipX, eChipY, matNxD.w, matNxD.h, chipE, { isMatrix: true });
                drawMatrixGrid(eChipG, eChipX, eChipY, activeData.embeddings, N, d, C.embedPos);
                dimLabel(eChipX + matNxD.w / 2, eChipY + matNxD.h, `<${N}, ${d}>`);
            } else {
                const eChipG = makeChip(eChipX, eChipY, eChipTextW, defaultChipH, chipE);
                eChipG.append('text')
                    .attr('x', eChipX + eChipTextW / 2).attr('y', eChipY + defaultChipH / 2 - 6 * scale)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                    .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                    .attr('font-style', 'italic').attr('font-weight', 'bold')
                    .attr('fill', phaseIdx < PHASES.indexOf('SHOW_EMBEDDINGS') ? C.textMuted : chipE.color)
                    .text('E');
                chipDimLabel(eChipG, eChipX + eChipTextW / 2, eChipY + defaultChipH / 2 + 7 * scale, `<${N}, ${d}>`);
            }

            // ============ STAGE 4-5: BRANCHING & PROJECTION ROW ============
            const wChipW = charW * 5 + chipPadX * 2;
            const wSmallGap = gap * 1.2;   // tight gap between E·Wq and E·Wk
            const wLargeGap = gap * 6;     // large gap separating E·Wv on the right
            const branchTopY = B_eq.stage4Y;
            const projRowY = B_eq.stage5Y;

            const wChips = [
                { label: 'E\u00b7W\u0071', result: 'Q', color: C.qText, bg: C.qBg, border: C.qBorder, fill: C.qFill, phase: 'PROJECT_Q' },
                { label: 'E\u00b7W\u2096', result: 'K', color: C.kText, bg: C.kBg, border: C.kBorder, fill: C.kFill, phase: 'PROJECT_K' },
                { label: 'E\u00b7W\u1d65', result: 'V', color: C.vText, bg: C.vBg, border: C.vBorder, fill: C.vFill, phase: 'PROJECT_V' },
            ];

            // Layout: [E·Wq][small gap][E·Wk]  [large gap]  [E·Wv]
            const leftGroupW = wChipW * 2 + wSmallGap;
            const projTotalW = leftGroupW + wLargeGap + wChipW;
            const projStartX = flowCenterX - projTotalW / 2;

            // X positions for each chip
            const projChipXs = [
                projStartX,                                    // E·Wq
                projStartX + wChipW + wSmallGap,               // E·Wk
                projStartX + leftGroupW + wLargeGap,           // E·Wv (isolated right)
            ];
            const projChipCenters = projChipXs.map(x => x + wChipW / 2);

            for (let wi = 0; wi < 3; wi++) {
                const wc = wChips[wi];
                const wx = projChipXs[wi];
                const wCenterX = projChipCenters[wi];
                const wLocked = phaseIdx < PHASES.indexOf(wc.phase);
                const wColor = wLocked ? C.textMuted : wc.color;
                const wBorderColor = wLocked ? C.textMuted : wc.border;
                const wArrowColor = wLocked ? C.textMuted : wc.fill;

                // Branching curve: E bottom center → chip top center
                // Arrows are always the E embedding blue since it's the embeddings flowing in
                const branchColor = wLocked ? C.textMuted : C.embedPos;
                const bStartY = branchTopY;
                const bEndY = projRowY;
                const bLineEndY = bEndY - arrowTipH;
                const bMidY = bStartY + (bLineEndY - bStartY) * 0.5;
                g.append('path')
                    .attr('d', `M${flowCenterX},${bStartY} C${flowCenterX},${bMidY} ${wCenterX},${bMidY} ${wCenterX},${bLineEndY}`)
                    .attr('fill', 'none').attr('stroke', branchColor)
                    .attr('stroke-width', 1.2 * scale);
                arrowHead(wCenterX, bEndY, branchColor);

                // Projection chip — uses dedicated bg/border CSS colors
                const wChipG = g.append('g');
                wChipG.append('rect')
                    .attr('x', wx).attr('y', projRowY).attr('width', wChipW).attr('height', defaultChipH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', wLocked ? C.textMuted : wc.bg).attr('opacity', wLocked ? 0.06 : 1);
                wChipG.append('rect')
                    .attr('x', wx).attr('y', projRowY).attr('width', wChipW).attr('height', defaultChipH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', 'none').attr('stroke', wBorderColor).attr('stroke-width', 2);
                // Split label so "E·" is embedding blue and "W_" keeps chip color
                const eLabelColor = wLocked ? C.textMuted : C.embedPos;
                const wLabelText = wChipG.append('text')
                    .attr('x', wCenterX).attr('y', projRowY + defaultChipH / 2 - 6 * scale)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                    .attr('font-family', SERIF).attr('font-size', mathSize * 0.65)
                    .attr('font-style', 'italic').attr('font-weight', 'bold');
                wLabelText.append('tspan').attr('fill', eLabelColor).text('E\u00b7');
                wLabelText.append('tspan').attr('fill', wColor).text(wc.label.replace('E\u00b7', ''));
                chipDimLabel(wChipG, wCenterX, projRowY + defaultChipH / 2 + 7 * scale, `<${d}, ${d}>`);
            }

            // ============ SHARED EQUATION DRAWING HELPER ============
            // Draws softmax(Q·K^T/√dk)·V + E for a given head's data.
            // Returns { ctxUnderlayX, ctxUnderlayY, ctxUnderlayW, ctxUnderlayH, qTargetCenterX, qTargetTopY, kTargetCenterX, kTargetTopY, vChipCenterX, vChipTopY, resECenterX, resEChipY, smBoxCenterY }
            function drawEquationForHead(centerX, topY, headQ, headK, headV, headRawScores, headScaledScores, headAttnWeights, headLabel) {
                const softmaxBoxPadY_h = B_eq.softmaxBoxPadY;
                const softmaxBoxPadX_h = Math.round(12 * scale);
                const softmaxBoxH_h = B_eq.softmaxBoxH;

                // Compute inner content width
                let softmaxInnerW;
                if (fracMode === 'softmax') {
                    softmaxInnerW = matNxN.w;
                } else if (fracMode === 'scaled') {
                    softmaxInnerW = smTextW + gap + lparenW + matNxN.w + rparenW;
                } else if (fracMode === 'QKT') {
                    const fracW2 = Math.max(matNxN.w, scTextW);
                    softmaxInnerW = smTextW + gap + lparenW + fracW2 + rparenW;
                } else {
                    const showQMat = fracMode === 'normal' && et.has('Q');
                    const showKMat = fracMode === 'normal' && et.has('K');
                    const qW2 = showQMat ? matNxD.w : qTextW;
                    const kW2 = showKMat ? matNxD.w : kTextW;
                    const numW2 = qW2 + dotTextW + kW2;
                    const fracW2 = Math.max(numW2, scTextW);
                    softmaxInnerW = smTextW + gap + lparenW + fracW2 + rparenW;
                }
                const softmaxBoxW_h = softmaxInnerW + softmaxBoxPadX_h * 2;
                const softmaxBoxX_h = centerX - softmaxBoxW_h / 2;
                const softmaxBoxY_h = topY + Math.round(8 * scale) + Math.round(4 * scale);

                // V chip position
                const showVMatrix = et.has('V');
                const vW_h = showVMatrix ? matNxD.w : vTextW;
                const smBoxRightEdge_h = softmaxBoxX_h + softmaxBoxW_h;
                const dotVGap_h = gap * 1.5;
                const dotVX_h = smBoxRightEdge_h + dotVGap_h;
                const vChipX_h = dotVX_h + charW * 2 + gap;
                const vChipCenterX_h = vChipX_h + vW_h / 2;

                // Context Vectors underlay
                const ctxUnderlayPad_h = Math.round(8 * scale);
                const ctxUnderlayX_h = softmaxBoxX_h - ctxUnderlayPad_h;
                const ctxUnderlayY_h = softmaxBoxY_h - ctxUnderlayPad_h;
                const ctxUnderlayW_h = (vChipX_h + vW_h) - softmaxBoxX_h + ctxUnderlayPad_h * 2;
                const ctxUnderlayH_h = softmaxBoxH_h + ctxUnderlayPad_h * 2;
                const ctxChipG_h = makeChip(ctxUnderlayX_h, ctxUnderlayY_h, ctxUnderlayW_h, ctxUnderlayH_h, chipCtx);
                // Dimension label
                const ctxLabelY_h = ctxUnderlayY_h + ctxUnderlayH_h - 4 * scale;
                g.append('text')
                    .attr('x', ctxUnderlayX_h + ctxUnderlayW_h / 2).attr('y', ctxLabelY_h)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'alphabetic')
                    .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                    .attr('fill', C.textMuted).attr('opacity', 0.7)
                    .text(headLabel ? `Head ${headLabel} — <${N}, ${d}>` : `<${N}, ${d}>`);

                // Softmax chip
                const smBoxChipG_h = makeChip(softmaxBoxX_h, softmaxBoxY_h, softmaxBoxW_h, softmaxBoxH_h, chipSm);
                const smBoxCenterY_h = softmaxBoxY_h + softmaxBoxH_h / 2;
                let sx_h = softmaxBoxX_h + softmaxBoxPadX_h;

                // Track Q/K^T positions
                let qTCX = centerX - 20 * scale, kTCX = centerX + 20 * scale;
                let qTTY = softmaxBoxY_h, kTTY = softmaxBoxY_h;

                if (fracMode === 'softmax') {
                    const matY = smBoxCenterY_h - matNxN.h / 2;
                    drawMatrixGrid(smBoxChipG_h, sx_h, matY, headAttnWeights, N, N, C.sectionTitle, 2);
                    dimLabel(sx_h + matNxN.w / 2, matY + matNxN.h, `<${N}, ${N}>`);
                } else {
                    g.append('text')
                        .attr('x', sx_h + smTextW / 2).attr('y', smBoxCenterY_h - 6 * scale)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-family', SERIF).attr('font-size', smallSize).attr('font-weight', 'bold')
                        .attr('fill', phaseIdx < PHASES.indexOf('APPLY_SOFTMAX') ? C.textMuted : chipSm.color)
                        .text('softmax');
                    g.append('text')
                        .attr('x', sx_h + smTextW / 2).attr('y', smBoxCenterY_h + 7 * scale)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                        .attr('fill', C.textMuted).attr('opacity', 0.7)
                        .text(`<${N}, ${N}>`);
                    sx_h += smTextW + gap;
                    staticText(sx_h + lparenW / 2, smBoxCenterY_h, '(', mathSize + 10);
                    sx_h += lparenW;

                    if (fracMode === 'scaled') {
                        const matY = smBoxCenterY_h - matNxN.h / 2;
                        const chipG2 = makeChip(sx_h, matY, matNxN.w, matNxN.h, chipSc, { isMatrix: true });
                        drawMatrixGrid(chipG2, sx_h, matY, headScaledScores, N, N, C.scaleLabel, 2);
                        dimLabel(sx_h + matNxN.w / 2, matY + matNxN.h, `<${N}, ${N}>`);
                        sx_h += matNxN.w;
                    } else {
                        const showQMatrix = fracMode === 'normal' && et.has('Q');
                        const showKMatrix = fracMode === 'normal' && et.has('K');
                        let numW, numH;
                        if (fracMode === 'QKT') {
                            numW = matNxN.w; numH = matNxN.h;
                        } else if (showQMatrix && showKMatrix) {
                            numW = matNxD.w + dotTextW + matNxD.w; numH = matNxD.h;
                        } else if (showQMatrix) {
                            numW = matNxD.w + dotTextW + kTextW; numH = Math.max(matNxD.h, defaultChipH);
                        } else if (showKMatrix) {
                            numW = qTextW + dotTextW + matNxD.w; numH = Math.max(matNxD.h, defaultChipH);
                        } else {
                            numW = qTextW + dotTextW + kTextW; numH = defaultChipH;
                        }
                        const fracW2 = Math.max(numW, scTextW);
                        const fracCenterX2 = sx_h + fracW2 / 2;
                        const numY = smBoxCenterY_h - fracGap / 2 - numH;

                        if (fracMode === 'QKT') {
                            const numStartX = fracCenterX2 - matNxN.w / 2;
                            const chipG2 = makeChip(numStartX, numY, matNxN.w, matNxN.h, chipQKT, { isMatrix: true });
                            drawMatrixGrid(chipG2, numStartX, numY, headRawScores, N, N, C.scaleLabel);
                            g.append('text').attr('x', numStartX + matNxN.w / 2).attr('y', numY - 4)
                                .attr('text-anchor', 'middle').attr('dominant-baseline', 'alphabetic')
                                .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                                .attr('fill', C.textMuted).attr('opacity', 0.7).text(`<${N}, ${N}>`);
                        } else {
                            const numStartX = fracCenterX2 - numW / 2;
                            let nx = numStartX;
                            if (showQMatrix) {
                                const qY = numY + (numH - matNxD.h) / 2;
                                const chipG2 = makeChip(nx, qY, matNxD.w, matNxD.h, chipQ, { isMatrix: true });
                                drawMatrixGrid(chipG2, nx, qY, headQ, N, d, C.qText);
                                g.append('text').attr('x', nx + matNxD.w / 2).attr('y', qY - 4)
                                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'alphabetic')
                                    .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                                    .attr('fill', C.textMuted).attr('opacity', 0.7).text(`<${N}, ${d}>`);
                                qTCX = nx + matNxD.w / 2; qTTY = qY;
                                nx += matNxD.w;
                            } else {
                                const qY = numY + (numH - defaultChipH) / 2;
                                const qChipG = makeChip(nx, qY, qTextW, defaultChipH, chipQ);
                                qChipG.append('text').attr('x', nx + qTextW / 2).attr('y', qY + defaultChipH / 2 - 6 * scale)
                                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                                    .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                                    .attr('font-style', 'italic').attr('font-weight', 'bold')
                                    .attr('fill', phaseIdx < PHASES.indexOf('PROJECT_Q') ? C.textMuted : chipQ.color).text('Q');
                                chipDimLabel(qChipG, nx + qTextW / 2, qY + defaultChipH / 2 + 7 * scale, `<${N}, ${d}>`);
                                qTCX = nx + qTextW / 2; qTTY = qY;
                                nx += qTextW;
                            }
                            const dotY = numY + (numH - defaultChipH) / 2;
                            const dotChipG2 = makeChip(nx, dotY, dotTextW, defaultChipH, chipQKT, { noBrackets: true });
                            dotChipG2.append('text').attr('x', nx + dotTextW / 2).attr('y', dotY + defaultChipH / 2)
                                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                                .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                                .attr('fill', phaseIdx < PHASES.indexOf('COMPUTE_SCORES') ? C.textMuted : chipQKT.color).text('\u00b7');
                            nx += dotTextW;
                            if (showKMatrix) {
                                const kY = numY + (numH - matNxD.h) / 2;
                                const chipG2 = makeChip(nx, kY, matNxD.w, matNxD.h, chipK, { isMatrix: true });
                                drawMatrixGrid(chipG2, nx, kY, headK, N, d, C.kText);
                                g.append('text').attr('x', nx + matNxD.w / 2).attr('y', kY - 4)
                                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'alphabetic')
                                    .attr('font-family', MONO).attr('font-size', 6.3 * scale)
                                    .attr('fill', C.textMuted).attr('opacity', 0.7).text(`<${d}, ${N}>`);
                                kTCX = nx + matNxD.w / 2; kTTY = kY;
                                nx += matNxD.w;
                            } else {
                                const kY = numY + (numH - defaultChipH) / 2;
                                const kChipG = makeChip(nx, kY, kTextW, defaultChipH, chipK);
                                kChipG.append('text').attr('x', nx + kTextW / 2).attr('y', kY + defaultChipH / 2 - 6 * scale)
                                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                                    .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                                    .attr('font-style', 'italic').attr('font-weight', 'bold')
                                    .attr('fill', phaseIdx < PHASES.indexOf('PROJECT_K') ? C.textMuted : chipK.color).text('K\u1d40');
                                chipDimLabel(kChipG, nx + kTextW / 2, kY + defaultChipH / 2 + 7 * scale, `<${d}, ${N}>`);
                                kTCX = nx + kTextW / 2; kTTY = kY;
                                nx += kTextW;
                            }
                        }
                        // Fraction line
                        g.append('line').attr('x1', fracCenterX2 - fracW2 / 2 + 2).attr('y1', smBoxCenterY_h)
                            .attr('x2', fracCenterX2 + fracW2 / 2 - 2).attr('y2', smBoxCenterY_h)
                            .attr('stroke', C.canvasText).attr('stroke-width', 1.2);
                        // √dk
                        const denY = smBoxCenterY_h + fracGap / 2;
                        const denX = fracCenterX2 - scTextW / 2;
                        const scChipG2 = makeChip(denX, denY, scTextW, defaultChipH, chipSc);
                        scChipG2.append('text').attr('x', denX + scTextW / 2).attr('y', denY + defaultChipH / 2 - 6 * scale)
                            .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                            .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                            .attr('font-style', 'italic').attr('font-weight', 'bold')
                            .attr('fill', phaseIdx < PHASES.indexOf('SCALE_SCORES') ? C.textMuted : chipSc.color).text('\u221ad\u2096');
                        chipDimLabel(scChipG2, denX + scTextW / 2, denY + defaultChipH / 2 + 7 * scale, `= ${Math.round(Math.sqrt(d))}`);
                        sx_h += fracW2;
                    }
                    staticText(sx_h + rparenW / 2, smBoxCenterY_h, ')', mathSize + 10);
                }

                // "· V"
                const vLocked_h = phaseIdx < PHASES.indexOf('PROJECT_V');
                staticText(dotVX_h + charW, smBoxCenterY_h, '\u00b7', mathSize);
                if (showVMatrix) {
                    const matY = smBoxCenterY_h - matNxD.h / 2;
                    const chipG2 = makeChip(vChipX_h, matY, matNxD.w, matNxD.h, chipV, { isMatrix: true });
                    drawMatrixGrid(chipG2, vChipX_h, matY, headV, N, d, C.vText);
                    dimLabel(vChipX_h + matNxD.w / 2, matY + matNxD.h, `<${N}, ${d}>`);
                } else {
                    const vChipY = smBoxCenterY_h - defaultChipH / 2;
                    const vChipG2 = makeChip(vChipX_h, vChipY, vTextW, defaultChipH, chipV);
                    vChipG2.append('text').attr('x', vChipX_h + vTextW / 2).attr('y', vChipY + defaultChipH / 2 - 6 * scale)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-family', SERIF).attr('font-size', mathSize)
                        .attr('font-style', 'italic').attr('font-weight', 'bold')
                        .attr('fill', vLocked_h ? C.textMuted : chipV.color).text('V');
                    chipDimLabel(vChipG2, vChipX_h + vTextW / 2, vChipY + defaultChipH / 2 + 7 * scale, `<${N}, ${d}>`);
                }

                // "+ E"
                const plusW_h = charW * 2.5;
                const resETextW_h = eChipTextW;
                const plusX_h = ctxUnderlayX_h + ctxUnderlayW_h + gap;
                const plusColor_h = phaseIdx >= PHASES.indexOf('SHOW_OUTPUT') ? C.canvasText : C.textMuted;
                staticText(plusX_h + plusW_h / 2, smBoxCenterY_h, '+', mathSize, plusColor_h);
                const resEChipX_h = plusX_h + plusW_h;
                const resEChipY_h = smBoxCenterY_h - defaultChipH / 2;
                const resEChipG_h = makeChip(resEChipX_h, resEChipY_h, resETextW_h, defaultChipH, chipResE);
                resEChipG_h.append('text').attr('x', resEChipX_h + resETextW_h / 2).attr('y', resEChipY_h + defaultChipH / 2 - 6 * scale)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                    .attr('font-family', SERIF).attr('font-size', mathSize - 1)
                    .attr('font-style', 'italic').attr('font-weight', 'bold')
                    .attr('fill', phaseIdx < PHASES.indexOf('SHOW_OUTPUT') ? C.textMuted : chipResE.color).text('E');
                chipDimLabel(resEChipG_h, resEChipX_h + resETextW_h / 2, resEChipY_h + defaultChipH / 2 + 7 * scale, `<${N}, ${d}>`);

                const vChipTopY_h = smBoxCenterY_h - (showVMatrix ? matNxD.h : defaultChipH) / 2;
                return {
                    ctxUnderlayX: ctxUnderlayX_h, ctxUnderlayY: ctxUnderlayY_h,
                    ctxUnderlayW: ctxUnderlayW_h, ctxUnderlayH: ctxUnderlayH_h,
                    qTargetCenterX: qTCX, qTargetTopY: qTTY,
                    kTargetCenterX: kTCX, kTargetTopY: kTTY,
                    vChipCenterX: vChipCenterX_h, vChipTopY: vChipTopY_h,
                    resECenterX: resEChipX_h + resETextW_h / 2, resEChipY: resEChipY_h,
                    smBoxCenterY: smBoxCenterY_h,
                    resETextW: resETextW_h, plusW: plusW_h,
                };
            }

            // ============ Shared arrow start positions ============
            const leftGroupCenterX = (projChipCenters[0] + projChipCenters[1]) / 2;
            const vProjCenterX = projChipCenters[2];
            const vLocked = phaseIdx < PHASES.indexOf('PROJECT_V');
            const vArrowColor = vLocked ? C.textMuted : C.vFill;

            // Variable to hold the equation result (for arrow routing and residual)
            let eqResult;

            if (B_eq.multiHead && activeData.heads && activeData.heads.length > 1) {
                // ============ MULTI-HEAD CAROUSEL ============
                // Active card shows full equation; inactive cards show mini thumbnails
                const nHeads = activeData.heads.length;
                const galleryY = B_eq.stageGalleryY;
                const titleH = B_eq.galleryTitleH;
                const arrowFromProjH = B_eq.galleryArrowFromProjH;
                const navH = B_eq.galleryNavH;

                // Active card's equation starts after title + arrows
                const eqTopY = galleryY + titleH + arrowFromProjH;

                // Arrows from projections into the active card's equation
                const arrowStartY = B_eq.stage5Y + B_eq.defaultChipH;
                const activeHead = activeData.heads[this.modelHead];

                // Draw the full equation for the active head (before the box so we know its bounds)
                eqResult = drawEquationForHead(
                    leftGroupCenterX, eqTopY,
                    activeHead.Q, activeHead.K, activeHead.V,
                    activeHead.rawScores || activeData.rawScores,
                    activeHead.scaledScores || activeData.scaledScores,
                    activeHead.attentionWeights || activeHead.attention_weights,
                    null  // label drawn separately on the box
                );

                // ---- Head container box: wraps title + equation ----
                const boxPad = Math.round(10 * scale);
                // Horizontal: span from context underlay left edge to residual E right edge + padding
                const boxX = eqResult.ctxUnderlayX - boxPad;
                const boxY = galleryY;
                const boxW = (eqResult.resECenterX + eqResult.resETextW / 2 + boxPad) - boxX + boxPad;
                const boxBottomY = eqResult.ctxUnderlayY + eqResult.ctxUnderlayH + boxPad;
                const boxH = boxBottomY - boxY;

                // Draw the box behind everything — insert at the start of gB
                g.insert('rect', ':first-child')
                    .attr('x', boxX).attr('y', boxY)
                    .attr('width', boxW).attr('height', boxH)
                    .attr('rx', 6).attr('ry', 6)
                    .attr('fill', C.activeBorder).attr('fill-opacity', 0.04)
                    .attr('stroke', C.activeBorder).attr('stroke-width', 1.5)
                    .attr('stroke-dasharray', `${4 * scale},${3 * scale}`);

                // Title label on the box top edge
                const titleLabelW = Math.round(charW * (`Head ${this.modelHead} / ${nHeads}`.length + 2));
                const titleLabelH = Math.round(titleH * 0.8);
                const titleLabelX = boxX + boxW / 2 - titleLabelW / 2;
                const titleLabelY = boxY - titleLabelH / 2;
                g.append('rect')
                    .attr('x', titleLabelX).attr('y', titleLabelY)
                    .attr('width', titleLabelW).attr('height', titleLabelH)
                    .attr('rx', 3).attr('ry', 3)
                    .attr('fill', C.canvasBg);
                g.append('text')
                    .attr('x', boxX + boxW / 2).attr('y', boxY)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                    .attr('font-family', MONO).attr('font-size', 9 * scale).attr('font-weight', '700')
                    .attr('fill', C.activeBorder)
                    .text(`Head ${this.modelHead} / ${nHeads}`);

                // Deferred arrows: projections → Q, K^T, V inside the active card
                curvedArrow(projChipCenters[0], arrowStartY, eqResult.qTargetCenterX, eqResult.qTargetTopY,
                    phaseIdx < PHASES.indexOf('PROJECT_Q') ? C.textMuted : wChips[0].fill,
                    phaseIdx < PHASES.indexOf('PROJECT_Q'));
                curvedArrow(projChipCenters[1], arrowStartY, eqResult.kTargetCenterX, eqResult.kTargetTopY,
                    phaseIdx < PHASES.indexOf('PROJECT_K') ? C.textMuted : wChips[1].fill,
                    phaseIdx < PHASES.indexOf('PROJECT_K'));
                curvedArrow(vProjCenterX, arrowStartY, eqResult.vChipCenterX, eqResult.vChipTopY, vArrowColor, vLocked);

                // Nav dots + prev/next below the equation
                const dotsY = galleryY + B_eq.headCardEquationH + navH / 2;
                const dotR = 3;
                const dotGap = 8;
                const dotsW = nHeads * dotGap;
                const dotsStartX = canvasW / 2 - dotsW / 2;
                for (let h = 0; h < nHeads; h++) {
                    const isActiveH = h === this.modelHead;
                    const dotG = g.append('g').style('cursor', 'pointer');
                    dotG.append('circle')
                        .attr('cx', dotsStartX + h * dotGap + dotR).attr('cy', dotsY)
                        .attr('r', isActiveH ? dotR + 1 : dotR)
                        .attr('fill', isActiveH ? C.activeBorder : C.cellBorder);
                    (function(idx, viz) {
                        dotG.on('click', function() { viz.modelHead = idx; viz.computeLayout(); viz.draw(); });
                    })(h, self);
                }
                // Prev/next buttons
                const navBtnR = Math.round(10 * scale);
                if (this.modelHead > 0) {
                    const prevG = g.append('g').style('cursor', 'pointer');
                    prevG.append('circle').attr('cx', dotsStartX - 20).attr('cy', dotsY).attr('r', navBtnR)
                        .attr('fill', C.cellBg).attr('stroke', C.cellBorder);
                    prevG.append('text').attr('x', dotsStartX - 20).attr('y', dotsY)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-family', MONO).attr('font-size', 10).attr('fill', C.canvasText).text('\u25C0');
                    prevG.on('click', function() { self.modelHead = Math.max(0, self.modelHead - 1); self.computeLayout(); self.draw(); });
                }
                if (this.modelHead < nHeads - 1) {
                    const nextG = g.append('g').style('cursor', 'pointer');
                    nextG.append('circle').attr('cx', dotsStartX + dotsW + 20).attr('cy', dotsY).attr('r', navBtnR)
                        .attr('fill', C.cellBg).attr('stroke', C.cellBorder);
                    nextG.append('text').attr('x', dotsStartX + dotsW + 20).attr('y', dotsY)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-family', MONO).attr('font-size', 10).attr('fill', C.canvasText).text('\u25B6');
                    nextG.on('click', function() { self.modelHead = Math.min(nHeads - 1, self.modelHead + 1); self.computeLayout(); self.draw(); });
                }

            } else {
                // ============ SINGLE-HEAD: STAGES 6+7 (original layout) ============
                const stage6TopY = B_eq.stage6Y;
                const stage7TopY = B_eq.stage7Y;

                eqResult = drawEquationForHead(
                    leftGroupCenterX, stage7TopY,
                    activeData.Q, activeData.K, activeData.V,
                    activeData.rawScores, activeData.scaledScores, activeData.attentionWeights,
                    null
                );

                // Arrows from projections
                curvedArrow(vProjCenterX, stage6TopY, eqResult.vChipCenterX, eqResult.vChipTopY, vArrowColor, vLocked);
                curvedArrow(projChipCenters[0], stage6TopY, eqResult.qTargetCenterX, eqResult.qTargetTopY,
                    phaseIdx < PHASES.indexOf('PROJECT_Q') ? C.textMuted : wChips[0].fill,
                    phaseIdx < PHASES.indexOf('PROJECT_Q'));
                curvedArrow(projChipCenters[1], stage6TopY, eqResult.kTargetCenterX, eqResult.kTargetTopY,
                    phaseIdx < PHASES.indexOf('PROJECT_K') ? C.textMuted : wChips[1].fill,
                    phaseIdx < PHASES.indexOf('PROJECT_K'));
            }

            // ============ RESIDUAL SKIP CONNECTION ============
            const residualColor = phaseIdx >= PHASES.indexOf('SHOW_OUTPUT') ? C.embedPos : C.textMuted;
            const skipLineX = Math.max(
                eqResult.resECenterX + eqResult.resETextW / 2 + 10 * scale,
                eqResult.ctxUnderlayX + eqResult.ctxUnderlayW + eqResult.plusW + eqResult.resETextW + 15 * scale
            );
            g.append('path')
                .attr('d', `M${flowCenterX + (showEMatrix ? matNxD.w/2 : eChipTextW/2)},${B_eq.stage3Y + (showEMatrix ? matNxD.h : defaultChipH) / 2} L${skipLineX},${B_eq.stage3Y + (showEMatrix ? matNxD.h : defaultChipH) / 2} L${skipLineX},${eqResult.resEChipY + defaultChipH / 2} L${eqResult.resECenterX + eqResult.resETextW / 2},${eqResult.resEChipY + defaultChipH / 2}`)
                .attr('fill', 'none').attr('stroke', residualColor)
                .attr('stroke-width', 1.2 * scale)
                .attr('stroke-dasharray', phaseIdx < PHASES.indexOf('SHOW_OUTPUT') ? `${3 * scale},${3 * scale}` : 'none');

            // ============ ARROW TO HEATMAP ============
            const smArrowColor = phaseIdx >= PHASES.indexOf('SHOW_OUTPUT') ? C.canvasText : C.textMuted;
            const heatmapArrowX = eqResult.ctxUnderlayX + eqResult.ctxUnderlayW / 2;
            arrowLine(heatmapArrowX, eqResult.ctxUnderlayY + eqResult.ctxUnderlayH, B_eq.stage9Y, smArrowColor);

            // ============ STAGE 9: BOTTOM PANEL (heatmap/output matrix) ============
            const bottomPanelY = B_eq.stage9Y;
            const bottomPanelH = B_eq.stage9H;
            if (this.eqRightMode === 'matrix') {
                this.drawInlineOutputMatrix(C, 0, bottomPanelY, bottomPanelH);
            } else {
                this.drawInlineAttentionHeatmap(C, 0, bottomPanelY, bottomPanelH);
            }
        }

        // ---- Attention heatmap drawn below the equation ----
        drawInlineAttentionHeatmap(C, startX, zy, barH) {
            const g = this.gB;
            const phaseIdx = PHASES.indexOf(this.phase);
            const softmaxIdx = PHASES.indexOf('APPLY_SOFTMAX');
            const activeData = this.getActiveData();
            const canvasW = this.layout.canvasW;
            if (phaseIdx < softmaxIdx || !activeData) return;

            const N = activeData.tokens.length;
            const heads = activeData.heads;
            const self = this;

            // Clickable background to toggle to matrix view
            g.append('rect')
                .attr('x', 0).attr('y', zy)
                .attr('width', canvasW).attr('height', barH)
                .attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('click', function() { self.eqRightMode = 'matrix'; self.draw(); });

            // Aggregate weights across visible heads
            const aggWeights = [];
            let visibleCount = 0;
            for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                if (this.visibleHeads[h]) visibleCount++;
            }
            for (let i = 0; i < N; i++) {
                const row = [];
                for (let j = 0; j < N; j++) {
                    let sum = 0;
                    for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                        if (!this.visibleHeads[h]) continue;
                        sum += heads[h].attentionWeights[i][j];
                    }
                    row.push(visibleCount > 0 ? sum / visibleCount : 0);
                }
                aggWeights.push(row);
            }

            // Layout calculations - center grid in full canvas width
            const padding = 4;
            const labelPad = 50;  // space for token labels on left
            const topLabelPad = 35;  // space for token labels on top
            const bottomPad = 16;  // space for hint at bottom
            const availW = canvasW - padding * 2;
            const availH = barH - topLabelPad - bottomPad;
            const cellSize = Math.min(
                availH / N,
                (availW - labelPad - padding) / N
            );
            const gridW = cellSize * N;
            const gridH = cellSize * N;
            // Center the grid horizontally in the canvas
            const gridX = (canvasW - gridW) / 2 + labelPad / 2;
            const gridY = zy + topLabelPad;

            // Draw row labels (queries - left)
            for (let i = 0; i < N; i++) {
                const y = gridY + i * cellSize + cellSize / 2;
                const label = activeData.tokens[i].length > 5
                    ? activeData.tokens[i].slice(0, 4) + '…'
                    : activeData.tokens[i];
                g.append('text')
                    .attr('x', gridX - 4).attr('y', y)
                    .attr('text-anchor', 'end').attr('dominant-baseline', 'central')
                    .attr('font-family', MONO).attr('font-size', 10)
                    .attr('fill', C.qText)
                    .text(label);
            }

            // Draw column labels (keys - top, rotated)
            for (let j = 0; j < N; j++) {
                const x = gridX + j * cellSize + cellSize / 2;
                const label = activeData.tokens[j].length > 5
                    ? activeData.tokens[j].slice(0, 4) + '…'
                    : activeData.tokens[j];
                g.append('text')
                    .attr('x', x).attr('y', gridY - 4)
                    .attr('text-anchor', 'start').attr('dominant-baseline', 'alphabetic')
                    .attr('font-family', MONO).attr('font-size', 10)
                    .attr('fill', C.kText)
                    .attr('transform', `rotate(-45, ${x}, ${gridY - 4})`)
                    .text(label);
            }

            // Draw heatmap cells with numbers
            const fontSize = Math.max(9, Math.min(14, cellSize * 0.35));
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const x = gridX + j * cellSize;
                    const y = gridY + i * cellSize;
                    const w = aggWeights[i][j];
                    const cellColor = interpolateHeatColor(w, C.heatCool, C.heatMid, C.heatWarm);
                    g.append('rect')
                        .attr('x', x).attr('y', y)
                        .attr('width', cellSize - 1).attr('height', cellSize - 1)
                        .attr('fill', cellColor)
                        .attr('rx', 1);

                    // Add weight value as text (use contrasting color)
                    const textColor = w > 0.5 ? '#fff' : C.text;
                    const displayVal = w < 0.01 ? '' : (w < 0.1 ? w.toFixed(2).slice(1) : w.toFixed(1));
                    g.append('text')
                        .attr('x', x + (cellSize - 1) / 2)
                        .attr('y', y + (cellSize - 1) / 2)
                        .attr('text-anchor', 'middle')
                        .attr('dominant-baseline', 'central')
                        .attr('font-family', MONO)
                        .attr('font-size', fontSize)
                        .attr('fill', textColor)
                        .text(displayVal);
                }
            }

            // Axis labels
            g.append('text')
                .attr('x', gridX - 32).attr('y', gridY + gridH / 2)
                .attr('transform', `rotate(-90, ${gridX - 32}, ${gridY + gridH / 2})`)
                .attr('text-anchor', 'middle').attr('font-size', 9)
                .attr('fill', C.textMuted)
                .text('Query');

            g.append('text')
                .attr('x', gridX + gridW / 2).attr('y', zy + 10)
                .attr('text-anchor', 'middle').attr('font-size', 9)
                .attr('fill', C.textMuted)
                .text('Key');

            // Dimension label
            g.append('text')
                .attr('x', gridX + gridW + 8).attr('y', gridY + gridH / 2)
                .attr('text-anchor', 'start').attr('dominant-baseline', 'central')
                .attr('font-family', MONO).attr('font-size', 7)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text(`<${N}, ${N}>`);

            // Hint label with toggle icon
            const hintX = gridX + gridW / 2;
            const hintY = zy + barH - 4;
            const hintG = g.append('g')
                .attr('class', 'toggle-hint')
                .attr('cursor', 'pointer')
                .on('click', function() { self.eqRightMode = 'matrix'; self.draw(); });

            hintG.append('text')
                .attr('x', hintX - 8).attr('y', hintY)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
                .attr('font-family', MONO).attr('font-size', 10)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text('⇄');

            hintG.append('text')
                .attr('x', hintX + 30).attr('y', hintY)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
                .attr('font-family', MONO).attr('font-size', 10)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text('matrix');
        }


        // ---- Output matrix drawn below the equation (toggle from BertViz) ----
        drawInlineOutputMatrix(C, startX, zy, barH) {
            const g = this.gB;
            const phaseIdx = PHASES.indexOf(this.phase);
            const softmaxIdx = PHASES.indexOf('APPLY_SOFTMAX');
            const activeData = this.getActiveData();
            const canvasW = this.layout.canvasW;
            const isModel = this.layout.isModel;
            if (phaseIdx < softmaxIdx || !activeData) return;

            const self = this;
            const N = activeData.tokens.length;
            const d = isModel ? activeData.Q[0].length : this.embedDim;
            const outputs = activeData.outputs;
            const matCellW = this.layout.B_eq.matCellW;
            const matCellH = this.layout.B_eq.matCellH;
            const matPad = 6;

            // Clickable background to toggle back to BertViz
            g.append('rect')
                .attr('x', 0).attr('y', zy)
                .attr('width', canvasW).attr('height', barH)
                .attr('fill', 'transparent').attr('cursor', 'pointer')
                .on('click', function() { self.eqRightMode = 'bertviz'; self.draw(); });

            // Center the output matrix in the full canvas
            const matW = d * matCellW + matPad * 2;
            const matH = N * matCellH + matPad * 2;
            const mx = (canvasW - matW) / 2;
            const my = zy + (barH - matH) / 2;

            // Border + brackets
            const outputColor = C.outputText || '#2e7d32';
            g.append('rect')
                .attr('x', mx).attr('y', my).attr('width', matW).attr('height', matH)
                .attr('rx', 3).attr('ry', 3)
                .attr('fill', outputColor).attr('opacity', 0.08);
            g.append('rect')
                .attr('x', mx).attr('y', my).attr('width', matW).attr('height', matH)
                .attr('rx', 3).attr('ry', 3)
                .attr('fill', 'none').attr('stroke', outputColor).attr('stroke-width', 1.2);

            // Bracket decorations
            const cap = 4, inset = 1.5;
            g.append('path')
                .attr('d', `M${mx+cap+inset},${my+inset} L${mx+inset},${my+inset} L${mx+inset},${my+matH-inset} L${mx+cap+inset},${my+matH-inset}`)
                .attr('fill', 'none').attr('stroke', outputColor).attr('stroke-width', 1.2);
            g.append('path')
                .attr('d', `M${mx+matW-cap-inset},${my+inset} L${mx+matW-inset},${my+inset} L${mx+matW-inset},${my+matH-inset} L${mx+matW-cap-inset},${my+matH-inset}`)
                .attr('fill', 'none').attr('stroke', outputColor).attr('stroke-width', 1.2);

            // Matrix values
            if (isModel) {
                // Heatmap mode for large matrices
                let minVal = Infinity, maxVal = -Infinity;
                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < d; j++) {
                        const v = outputs[i][j];
                        if (v < minVal) minVal = v;
                        if (v > maxVal) maxVal = v;
                    }
                }
                const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 0.001);
                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < d; j++) {
                        const cx = mx + matPad + j * matCellW;
                        const cy = my + matPad + i * matCellH;
                        const v = outputs[i][j];
                        const t = (v / absMax + 1) / 2;
                        g.append('rect')
                            .attr('x', cx).attr('y', cy)
                            .attr('width', matCellW).attr('height', matCellH)
                            .attr('fill', interpolateHeatColor(t, '#4575b4', '#f7f7f7', '#d73027'));
                    }
                }
            } else {
                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < d; j++) {
                        const cx = mx + matPad + j * matCellW;
                        const cy = my + matPad + i * matCellH;
                        g.append('text')
                            .attr('x', cx + matCellW / 2).attr('y', cy + matCellH / 2)
                            .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                            .attr('font-family', MONO).attr('font-size', 8)
                            .attr('fill', outputColor)
                            .text(outputs[i][j].toFixed(1));
                    }
                }
            }

            // Dimension label
            g.append('text')
                .attr('x', mx + matW + 8).attr('y', my + matH / 2)
                .attr('text-anchor', 'start').attr('dominant-baseline', 'central')
                .attr('font-family', MONO).attr('font-size', 7)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text(`<${N}, ${d}>`);

            // Label below with toggle hint
            const hintG = g.append('g')
                .attr('class', 'toggle-hint')
                .attr('cursor', 'pointer')
                .on('click', function() { self.eqRightMode = 'bertviz'; self.draw(); });

            hintG.append('text')
                .attr('x', mx + matW / 2 - 30).attr('y', my + matH + 10)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
                .attr('font-family', MONO).attr('font-size', 10)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text('⇄');

            hintG.append('text')
                .attr('x', mx + matW / 2 + 12).attr('y', my + matH + 10)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
                .attr('font-family', MONO).attr('font-size', 10)
                .attr('fill', C.textMuted).attr('opacity', 0.7)
                .text('heatmap');
        }

        // ============================================
        // Zone C: Head-Lines View or Heatmap View
        // ============================================
        drawZoneC(C) {
            const phaseIdx = PHASES.indexOf(this.phase);
            const softmaxIdx = PHASES.indexOf('APPLY_SOFTMAX');
            const isModel = this.dataMode !== 'synthetic';

            // Show/hide view mode bar based on phase and mode
            // In model mode, hide Lines button since BertViz is inline; only show Heatmap and Compare Layers
            if (phaseIdx >= softmaxIdx && this.getActiveData()) {
                this.viewModeBar.classList.add('visible');
                this.btnLayerCompare.style.display = isModel ? '' : 'none';
                // Hide Lines/Heatmap toggle in model mode - only show Compare Layers
                if (this.btnViewLines) this.btnViewLines.style.display = isModel ? 'none' : '';
                if (this.btnViewHeatmap) this.btnViewHeatmap.style.display = isModel ? 'none' : '';
            } else {
                this.viewModeBar.classList.remove('visible');
            }

            if (phaseIdx < softmaxIdx) return;
            if (!this.getActiveData()) return;

            // Layer comparison mode takes priority
            if (this.layerCompareMode && isModel && this.modelData) {
                this.drawLayerComparisonView(C);
            } else if (this.zoneViewMode === 'heatmap') {
                this.drawHeatmapView(C);
            } else if (!isModel) {
                // Only show head lines view in synthetic mode (redundant with inline BertViz in model mode)
                this.drawHeadLinesView(C);
            }
            // In model mode with lines view (default), Zone C is empty - BertViz is shown inline
        }

        // ============================================
        // Head-Lines View (BertViz-style)
        // ============================================
        drawHeadLinesView(C) {
            const g = this.gC;
            const activeData = this.getActiveData();
            const canvasW = this.layout.canvasW;
            const N = activeData.tokens.length;
            const zy = this.layout.C.y;
            const zh = this.layout.C.h;
            const heads = activeData.heads;
            const colors = getHeadColors();

            const progress = 1;

            // Title
            g.append('text')
                .attr('x', 15).attr('y', zy + 4)
                .attr('dominant-baseline', 'hanging')
                .attr('font-family', MONO).attr('font-size', 10).attr('font-weight', 'bold')
                .attr('fill', C.sectionTitle)
                .text('ATTENTION HEAD VIEW');

            // Layout
            const padding = 20;
            const leftX = 80;
            const rightX = canvasW / 2 - 20;
            const lineLeftX = leftX + 10;
            const lineRightX = rightX - 10;
            const topY = zy + padding + 6;
            const usableH = zh - padding * 2 - 6;
            const tokenSpacing = N > 1 ? usableH / (N - 1) : 0;

            const self = this;

            // Draw left token column (queries) — interactive
            for (let i = 0; i < N; i++) {
                const y = topY + i * tokenSpacing;
                const isHovered = (this.hoveredToken === i && this.hoveredSide === 'left');
                const tokenText = activeData.tokens[i];
                const textW = tokenText.length * 7 + 8;

                // Background highlight for hovered token
                if (isHovered) {
                    g.append('rect')
                        .attr('x', leftX - textW - 2).attr('y', y - 10)
                        .attr('width', textW + 4).attr('height', 20)
                        .attr('rx', 4).attr('ry', 4)
                        .attr('fill', C.qBg)
                        .attr('filter', 'url(#token-glow-q)');
                }

                const tokenG = g.append('g')
                    .attr('transform', isHovered ? `translate(${leftX}, ${y}) scale(1.1) translate(${-leftX}, ${-y})` : '');

                tokenG.append('text')
                    .attr('x', leftX).attr('y', y)
                    .attr('text-anchor', 'end').attr('dominant-baseline', 'central')
                    .attr('font-family', MONO)
                    .attr('font-size', isHovered ? 11 : 10)
                    .attr('font-weight', 'bold')
                    .attr('fill', isHovered ? C.activeBorder : C.qText)
                    .attr('cursor', 'pointer')
                    .datum({ idx: i, side: 'left' })
                    .on('mouseenter', function(event, d) {
                        self.hoveredToken = d.idx;
                        self.hoveredSide = d.side;
                        self.draw();
                    })
                    .on('mouseleave', function() {
                        self.hoveredToken = null;
                        self.hoveredSide = null;
                        self.draw();
                    })
                    .text(tokenText);

                // Dot at line anchor
                g.append('circle')
                    .attr('cx', lineLeftX).attr('cy', y).attr('r', isHovered ? 3 : 2)
                    .attr('fill', C.qText);
            }

            // Draw right token column (keys) — interactive
            for (let i = 0; i < N; i++) {
                const y = topY + i * tokenSpacing;
                const isHovered = (this.hoveredToken === i && this.hoveredSide === 'right');
                const tokenText = activeData.tokens[i];
                const textW = tokenText.length * 7 + 8;

                // Background highlight for hovered token
                if (isHovered) {
                    g.append('rect')
                        .attr('x', rightX - 2).attr('y', y - 10)
                        .attr('width', textW + 4).attr('height', 20)
                        .attr('rx', 4).attr('ry', 4)
                        .attr('fill', C.kBg)
                        .attr('filter', 'url(#token-glow-k)');
                }

                const tokenG = g.append('g')
                    .attr('transform', isHovered ? `translate(${rightX}, ${y}) scale(1.1) translate(${-rightX}, ${-y})` : '');

                tokenG.append('text')
                    .attr('x', rightX).attr('y', y)
                    .attr('dominant-baseline', 'central')
                    .attr('font-family', MONO)
                    .attr('font-size', isHovered ? 11 : 10)
                    .attr('font-weight', 'bold')
                    .attr('fill', isHovered ? C.activeBorder : C.kText)
                    .attr('cursor', 'pointer')
                    .datum({ idx: i, side: 'right' })
                    .on('mouseenter', function(event, d) {
                        self.hoveredToken = d.idx;
                        self.hoveredSide = d.side;
                        self.draw();
                    })
                    .on('mouseleave', function() {
                        self.hoveredToken = null;
                        self.hoveredSide = null;
                        self.draw();
                    })
                    .text(tokenText);

                g.append('circle')
                    .attr('cx', lineRightX).attr('cy', y).attr('r', isHovered ? 3 : 2)
                    .attr('fill', C.kText);
            }

            // Draw attention lines for each visible head
            for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                if (!this.visibleHeads[h]) continue;

                const weights = heads[h].attentionWeights;
                const color = colors[h % colors.length];

                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < N; j++) {
                        const w = weights[i][j];
                        if (w < 0.01) continue;

                        if (this.hoveredToken !== null) {
                            if (this.hoveredSide === 'left' && i !== this.hoveredToken) continue;
                            if (this.hoveredSide === 'right' && j !== this.hoveredToken) continue;
                        }

                        const y1 = topY + i * tokenSpacing;
                        const y2 = topY + j * tokenSpacing;

                        g.append('line')
                            .attr('x1', lineLeftX).attr('y1', y1)
                            .attr('x2', lineRightX).attr('y2', y2)
                            .attr('stroke', color)
                            .attr('stroke-width', 1 + w * 3)
                            .attr('opacity', progress * w);
                    }
                }
            }
        }

        // ============================================
        // Layer Comparison View (small multiples)
        // ============================================
        drawLayerComparisonView(C) {
            const g = this.gC;
            const canvasW = this.layout.canvasW;
            const zy = this.layout.C.y;
            const zh = this.layout.C.h;
            const tokens = this.modelData.tokens;
            const N = tokens.length;
            const numLayers = this.modelData.layers.length;
            const self = this;

            // Title
            g.append('text')
                .attr('x', 15).attr('y', zy + 4)
                .attr('dominant-baseline', 'hanging')
                .attr('font-family', MONO).attr('font-size', 10).attr('font-weight', 'bold')
                .attr('fill', C.sectionTitle)
                .text('LAYER COMPARISON (click to expand)');

            // Layout: 4 columns x 3 rows for 12 layers
            const cols = 4;
            const rows = 3;
            const padding = 20;
            const topY = zy + padding + 8;
            const usableW = canvasW - padding * 2;
            const usableH = zh - padding - 12;
            const cellW = usableW / cols - 8;
            const cellH = usableH / rows - 8;
            const gridSize = Math.min(cellW - 30, cellH - 16);
            const cellPixelSize = gridSize / N;

            for (let layer = 0; layer < numLayers; layer++) {
                const col = layer % cols;
                const row = Math.floor(layer / cols);
                const x = padding + col * (cellW + 8);
                const y = topY + row * (cellH + 8);
                const isSelected = this.modelLayer === layer;

                const layerData = this.modelData.layers[layer];

                // Aggregate attention weights across all heads for this layer
                const aggWeights = [];
                const numHeads = layerData.heads.length;
                for (let i = 0; i < N; i++) {
                    const rowW = [];
                    for (let j = 0; j < N; j++) {
                        let sum = 0;
                        for (let h = 0; h < numHeads; h++) {
                            sum += layerData.heads[h].attention_weights[i][j];
                        }
                        rowW.push(sum / numHeads);
                    }
                    aggWeights.push(rowW);
                }

                // Background for selected layer
                const bgG = g.append('g')
                    .attr('class', 'layer-cell')
                    .attr('cursor', 'pointer')
                    .datum({ layer })
                    .on('click', function(event, d) {
                        self.modelLayer = d.layer;
                        self.layerValue.textContent = d.layer;
                        self.layerCompareMode = false;
                        self.btnLayerCompare.classList.remove('active');
                        self.reset();
                    });

                bgG.append('rect')
                    .attr('x', x - 2).attr('y', y - 2)
                    .attr('width', cellW + 4).attr('height', cellH + 4)
                    .attr('rx', 4).attr('ry', 4)
                    .attr('fill', isSelected ? C.activeBorder : 'transparent')
                    .attr('opacity', isSelected ? 0.15 : 0);

                bgG.append('rect')
                    .attr('x', x - 2).attr('y', y - 2)
                    .attr('width', cellW + 4).attr('height', cellH + 4)
                    .attr('rx', 4).attr('ry', 4)
                    .attr('fill', 'none')
                    .attr('stroke', isSelected ? C.activeBorder : C.cellBorder)
                    .attr('stroke-width', isSelected ? 2 : 1);

                // Layer label
                g.append('text')
                    .attr('x', x + 2).attr('y', y + 10)
                    .attr('font-family', MONO).attr('font-size', 9).attr('font-weight', 'bold')
                    .attr('fill', isSelected ? C.activeBorder : C.labelColor)
                    .text(`L${layer}`);

                // Mini heatmap
                const gridX = x + 24;
                const gridY = y + 2;

                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < N; j++) {
                        const cx = gridX + j * cellPixelSize;
                        const cy = gridY + i * cellPixelSize;
                        const w = aggWeights[i][j];
                        const cellColor = interpolateHeatColor(w, C.heatCool, C.heatMid, C.heatWarm);

                        g.append('rect')
                            .attr('x', cx).attr('y', cy)
                            .attr('width', cellPixelSize - 0.5).attr('height', cellPixelSize - 0.5)
                            .attr('fill', cellColor)
                            .attr('pointer-events', 'none');
                    }
                }

                // Stats: max attention
                let maxW = 0;
                for (const row of aggWeights) {
                    for (const w of row) {
                        if (w > maxW) maxW = w;
                    }
                }

                g.append('text')
                    .attr('x', x + 2).attr('y', y + cellH - 4)
                    .attr('font-family', MONO).attr('font-size', 7)
                    .attr('fill', C.textMuted)
                    .text(`max: ${maxW.toFixed(2)}`);
            }
        }

        // ============================================
        // Heatmap View (NxN grid)
        // ============================================
        drawHeatmapView(C) {
            const g = this.gC;
            const activeData = this.getActiveData();
            const canvasW = this.layout.canvasW;
            const N = activeData.tokens.length;
            const zy = this.layout.C.y;
            const zh = this.layout.C.h;
            const heads = activeData.heads;
            const colors = getHeadColors();
            const self = this;

            // Title
            g.append('text')
                .attr('x', 15).attr('y', zy + 4)
                .attr('dominant-baseline', 'hanging')
                .attr('font-family', MONO).attr('font-size', 10).attr('font-weight', 'bold')
                .attr('fill', C.sectionTitle)
                .text('ATTENTION HEATMAP');

            // Compute aggregated attention weights (average across visible heads)
            const aggWeights = [];
            let visibleCount = 0;
            for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                if (!this.visibleHeads[h]) continue;
                visibleCount++;
            }

            for (let i = 0; i < N; i++) {
                const row = [];
                for (let j = 0; j < N; j++) {
                    let sum = 0;
                    for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                        if (!this.visibleHeads[h]) continue;
                        sum += heads[h].attentionWeights[i][j];
                    }
                    row.push(visibleCount > 0 ? sum / visibleCount : 0);
                }
                aggWeights.push(row);
            }

            // Layout
            const padding = 24;
            const labelPad = 50;  // space for token labels
            const topY = zy + padding + 6;
            const usableH = zh - padding * 2 - 6;
            const cellSize = Math.min(
                (usableH - 10) / N,
                (canvasW / 2 - labelPad * 2) / N
            );
            const gridW = cellSize * N;
            const gridH = cellSize * N;
            const gridX = (canvasW / 2 - gridW) / 2 + labelPad;
            const gridY = topY + 10;

            // Draw row labels (queries - left side)
            for (let i = 0; i < N; i++) {
                const y = gridY + i * cellSize + cellSize / 2;
                const isSelected = this.selectedCell && this.selectedCell.i === i;
                g.append('text')
                    .attr('x', gridX - 6).attr('y', y)
                    .attr('text-anchor', 'end').attr('dominant-baseline', 'central')
                    .attr('font-family', MONO).attr('font-size', 9)
                    .attr('font-weight', isSelected ? 'bold' : 'normal')
                    .attr('fill', isSelected ? C.activeBorder : C.qText)
                    .text(activeData.tokens[i]);
            }

            // Draw column labels (keys - top side)
            for (let j = 0; j < N; j++) {
                const x = gridX + j * cellSize + cellSize / 2;
                const isSelected = this.selectedCell && this.selectedCell.j === j;
                g.append('text')
                    .attr('x', x).attr('y', gridY - 6)
                    .attr('text-anchor', 'middle').attr('dominant-baseline', 'alphabetic')
                    .attr('font-family', MONO).attr('font-size', 9)
                    .attr('font-weight', isSelected ? 'bold' : 'normal')
                    .attr('fill', isSelected ? C.activeBorder : C.kText)
                    .text(activeData.tokens[j]);
            }

            // Draw heatmap cells
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const x = gridX + j * cellSize;
                    const y = gridY + i * cellSize;
                    const w = aggWeights[i][j];
                    const isSelected = this.selectedCell && this.selectedCell.i === i && this.selectedCell.j === j;

                    // Cell background
                    const cellColor = interpolateHeatColor(w, C.heatCool, C.heatMid, C.heatWarm);
                    g.append('rect')
                        .attr('x', x).attr('y', y)
                        .attr('width', cellSize - 1).attr('height', cellSize - 1)
                        .attr('fill', cellColor)
                        .attr('stroke', isSelected ? C.activeBorder : 'none')
                        .attr('stroke-width', isSelected ? 2 : 0)
                        .attr('cursor', 'pointer')
                        .datum({ i, j, w })
                        .on('click', function(event, d) {
                            if (self.selectedCell && self.selectedCell.i === d.i && self.selectedCell.j === d.j) {
                                self.selectedCell = null;
                            } else {
                                self.selectedCell = { i: d.i, j: d.j };
                            }
                            self.draw();
                        });

                    // Show weight value for larger cells
                    if (cellSize >= 20) {
                        g.append('text')
                            .attr('x', x + (cellSize - 1) / 2).attr('y', y + (cellSize - 1) / 2)
                            .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                            .attr('font-family', MONO).attr('font-size', 8)
                            .attr('fill', w > 0.5 ? '#fff' : C.cellText)
                            .attr('pointer-events', 'none')
                            .text(w.toFixed(2));
                    }
                }
            }

            // Draw color scale legend
            const scaleX = gridX + gridW + 20;
            const scaleY = gridY;
            const scaleW = 12;
            const scaleH = gridH;
            const gradSteps = 20;

            for (let s = 0; s < gradSteps; s++) {
                const t = 1 - s / (gradSteps - 1);  // reverse so high values at top
                const sy = scaleY + s * (scaleH / gradSteps);
                const sh = scaleH / gradSteps + 1;
                g.append('rect')
                    .attr('x', scaleX).attr('y', sy)
                    .attr('width', scaleW).attr('height', sh)
                    .attr('fill', interpolateHeatColor(t, C.heatCool, C.heatMid, C.heatWarm));
            }

            g.append('text')
                .attr('x', scaleX + scaleW + 4).attr('y', scaleY + 4)
                .attr('font-family', MONO).attr('font-size', 8)
                .attr('fill', C.labelColor)
                .text('1.0');

            g.append('text')
                .attr('x', scaleX + scaleW + 4).attr('y', scaleY + scaleH - 2)
                .attr('font-family', MONO).attr('font-size', 8)
                .attr('fill', C.labelColor)
                .text('0.0');

            // If a cell is selected, show details
            if (this.selectedCell) {
                const { i, j } = this.selectedCell;
                const detailX = canvasW / 2 + 30;
                const detailY = topY + 10;

                g.append('text')
                    .attr('x', detailX).attr('y', detailY)
                    .attr('font-family', MONO).attr('font-size', 11).attr('font-weight', 'bold')
                    .attr('fill', C.sectionTitle)
                    .text(`"${activeData.tokens[i]}" → "${activeData.tokens[j]}"`);

                g.append('text')
                    .attr('x', detailX).attr('y', detailY + 18)
                    .attr('font-family', MONO).attr('font-size', 10)
                    .attr('fill', C.labelColor)
                    .text(`Avg weight: ${aggWeights[i][j].toFixed(4)}`);

                // Show per-head weights
                let headY = detailY + 40;
                for (let h = 0; h < this.numHeads && h < heads.length; h++) {
                    if (!this.visibleHeads[h]) continue;
                    const hw = heads[h].attentionWeights[i][j];
                    const color = colors[h % colors.length];

                    g.append('circle')
                        .attr('cx', detailX + 6).attr('cy', headY - 3)
                        .attr('r', 5).attr('fill', color);

                    g.append('text')
                        .attr('x', detailX + 16).attr('y', headY)
                        .attr('font-family', MONO).attr('font-size', 9)
                        .attr('fill', C.cellText)
                        .text(`Head ${h}: ${hw.toFixed(4)}`);

                    headY += 16;
                }
            }
        }

        // ============================================
        // Head color bar
        // ============================================
        buildHeadColorBar() {
            const bar = this.headColorBar;
            const label = bar.querySelector('.attn-head-color-bar-label');
            bar.innerHTML = '';
            bar.appendChild(label);

            const colors = getHeadColors();
            const self = this;

            for (let h = 0; h < this.numHeads; h++) {
                const btn = document.createElement('button');
                btn.className = 'attn-head-btn' + (this.visibleHeads[h] ? ' active' : '');
                btn.style.backgroundColor = colors[h % colors.length];
                btn.dataset.head = h;

                // Create tooltip element
                const tooltip = document.createElement('div');
                tooltip.className = 'attn-head-tooltip';
                tooltip.innerHTML = `<strong>Head ${h}</strong><br><span class="tooltip-stats">-</span>`;
                btn.appendChild(tooltip);

                btn.addEventListener('mouseenter', () => {
                    const activeData = self.getActiveData();
                    if (activeData && activeData.heads && activeData.heads[h]) {
                        const weights = activeData.heads[h].attentionWeights;
                        let maxW = 0, sumW = 0, count = 0;
                        for (const row of weights) {
                            for (const w of row) {
                                if (w > maxW) maxW = w;
                                sumW += w;
                                count++;
                            }
                        }
                        const avgW = count > 0 ? sumW / count : 0;
                        tooltip.querySelector('.tooltip-stats').innerHTML =
                            `Max: ${maxW.toFixed(3)}<br>Avg: ${avgW.toFixed(3)}`;
                    }
                });

                btn.addEventListener('click', () => {
                    const idx = parseInt(btn.dataset.head);
                    const allVisible = this.visibleHeads.every(v => v);
                    const soloVisible = this.visibleHeads.filter(v => v).length === 1 && this.visibleHeads[idx];

                    if (allVisible) {
                        this.visibleHeads = this.visibleHeads.map((_, i) => i === idx);
                    } else if (soloVisible) {
                        this.visibleHeads = this.visibleHeads.map(() => true);
                    } else {
                        this.visibleHeads[idx] = !this.visibleHeads[idx];
                    }
                    this.updateHeadButtons();
                    this.draw();
                });

                bar.appendChild(btn);
            }

            const allBtn = document.createElement('button');
            allBtn.className = 'attn-head-btn-all';
            allBtn.textContent = 'All';
            allBtn.addEventListener('click', () => {
                this.visibleHeads = this.visibleHeads.map(() => true);
                this.updateHeadButtons();
                this.draw();
            });
            bar.appendChild(allBtn);
        }

        updateHeadButtons() {
            const btns = this.headColorBar.querySelectorAll('.attn-head-btn');
            btns.forEach((btn, i) => {
                if (this.visibleHeads[i]) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        updateRunButtonState() {
            if (this.phase === 'IDLE') {
                this.btnRun.classList.add('pulse-ready');
            } else {
                this.btnRun.classList.remove('pulse-ready');
            }
        }
    }

    // ============================================
    // Bootstrap
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new AttentionVisualizer();

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
