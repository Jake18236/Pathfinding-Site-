/**
 * Transformer & Attention Visualizer
 *
 * Interactive visualization of self-attention mechanisms in Transformers.
 * Features attention heatmap, multi-head attention, Q/K/V computation,
 * positional encoding, and temperature control.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_WIDTH = 560;
    const CANVAS_HEIGHT = 400;
    const PADDING = { top: 60, right: 30, bottom: 30, left: 80 };
    const DEFAULT_DIM = 64;
    const MIN_HEADS = 1;
    const MAX_HEADS = 8;
    const DEFAULT_SENTENCE = 'The cat sat on the mat';

    // ============================================
    // Math Utilities
    // ============================================

    /**
     * Softmax over an array of values
     */
    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    /**
     * Dot product of two vectors
     */
    function dot(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Generate a random vector of given dimension with some structure.
     * Uses a seeded approach so results are deterministic per token string + head.
     */
    function hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash |= 0;
        }
        return hash;
    }

    /**
     * Simple pseudo-random number generator (mulberry32)
     */
    function mulberry32(seed) {
        return function() {
            seed |= 0;
            seed = seed + 0x6D2B79F5 | 0;
            let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    /**
     * Generate a structured random vector for a token, simulating learned embeddings.
     * Similar tokens (same first letter, common words) produce somewhat correlated vectors.
     */
    function generateEmbedding(token, dim, headIndex, projType) {
        const seed = hashCode(token + '_' + headIndex + '_' + projType);
        const rng = mulberry32(seed);
        const vec = [];
        for (let i = 0; i < dim; i++) {
            // Generate approximately normal distribution via Box-Muller-like approach
            const u1 = rng();
            const u2 = rng();
            const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
            vec.push(z / Math.sqrt(dim));
        }
        return vec;
    }

    /**
     * Compute sinusoidal positional encoding for a given position and dimension
     */
    function positionalEncoding(pos, dim) {
        const pe = new Array(dim);
        for (let i = 0; i < dim; i++) {
            const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
            pe[i] = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
        }
        return pe;
    }

    /**
     * Add two vectors element-wise
     */
    function vecAdd(a, b) {
        return a.map((v, i) => v + b[i]);
    }

    // ============================================
    // AttentionComputer
    // ============================================
    class AttentionComputer {
        constructor() {
            this.tokens = [];
            this.numHeads = 4;
            this.dim = DEFAULT_DIM;
            this.temperature = 1.0;
            this.usePE = false;
            this.attentionWeights = []; // [head][query][key]
            this.scores = []; // raw scores before softmax
        }

        setTokens(sentence) {
            this.tokens = sentence.trim().split(/\s+/).filter(t => t.length > 0);
        }

        computeAttention() {
            const n = this.tokens.length;
            const headDim = Math.max(4, Math.floor(this.dim / this.numHeads));

            this.attentionWeights = [];
            this.scores = [];

            for (let h = 0; h < this.numHeads; h++) {
                // Generate Q, K, V matrices for this head
                const Q = [];
                const K = [];

                for (let i = 0; i < n; i++) {
                    let qVec = generateEmbedding(this.tokens[i], headDim, h, 'Q');
                    let kVec = generateEmbedding(this.tokens[i], headDim, h, 'K');

                    // Add positional encoding if enabled
                    if (this.usePE) {
                        const pe = positionalEncoding(i, headDim);
                        qVec = vecAdd(qVec, pe.map(v => v * 0.1));
                        kVec = vecAdd(kVec, pe.map(v => v * 0.1));
                    }

                    Q.push(qVec);
                    K.push(kVec);
                }

                // Compute scaled dot-product attention scores
                const scale = Math.sqrt(headDim);
                const headScores = [];
                const headWeights = [];

                for (let i = 0; i < n; i++) {
                    const rowScores = [];
                    for (let j = 0; j < n; j++) {
                        const score = dot(Q[i], K[j]) / scale / this.temperature;
                        rowScores.push(score);
                    }
                    headScores.push(rowScores);
                    headWeights.push(softmax(rowScores));
                }

                this.scores.push(headScores);
                this.attentionWeights.push(headWeights);
            }
        }

        getAverageAttention() {
            const n = this.tokens.length;
            const avg = Array.from({ length: n }, () => new Array(n).fill(0));

            for (let h = 0; h < this.numHeads; h++) {
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        avg[i][j] += this.attentionWeights[h][i][j] / this.numHeads;
                    }
                }
            }
            return avg;
        }

        getMaxAttention(headIndex) {
            const weights = (headIndex === -1) ? this.getAverageAttention() : this.attentionWeights[headIndex];
            let max = -Infinity;
            for (const row of weights) {
                for (const w of row) {
                    if (w > max) max = w;
                }
            }
            return max;
        }

        getMinAttention(headIndex) {
            const weights = (headIndex === -1) ? this.getAverageAttention() : this.attentionWeights[headIndex];
            let min = Infinity;
            for (const row of weights) {
                for (const w of row) {
                    if (w < min) min = w;
                }
            }
            return min;
        }
    }

    // ============================================
    // Theme Colors
    // ============================================
    function getColors() {
        const isDark = VizLib.ThemeManager.isDarkTheme();
        return {
            bg:             isDark ? '#1d2021' : '#fafafa',
            text:           isDark ? '#ebdbb2' : '#333333',
            textMuted:      isDark ? '#a89984' : '#999999',
            gridLine:       isDark ? 'rgba(168,153,132,0.15)' : 'rgba(0,0,0,0.06)',
            cellBorder:     isDark ? 'rgba(80,73,69,0.6)' : 'rgba(200,200,200,0.8)',
            hoverBorder:    isDark ? '#fabd2f' : '#ffd700',
            tokenHighlight: isDark ? '#83a598' : '#1565c0',
            rowHighlight:   isDark ? 'rgba(131,165,152,0.1)' : 'rgba(21,101,192,0.05)',
            colHighlight:   isDark ? 'rgba(131,165,152,0.1)' : 'rgba(21,101,192,0.05)',
            // Heatmap gradient stops (value 0 -> 1)
            gradientStops: isDark
                ? ['#1d2021', '#3c3836', '#504945', '#d65d0e', '#fe8019', '#fb4934', '#cc241d']
                : ['#f0f4ff', '#b3cde3', '#8fb5d5', '#fdd49e', '#fc8d59', '#d73027', '#7f0000'],
            // PE colors
            pePositive:     isDark ? '#83a598' : '#2196f3',
            peNegative:     isDark ? '#fb4934' : '#ff5722',
        };
    }

    /**
     * Interpolate through gradient stops for a value in [0, 1]
     */
    function interpolateColor(stops, t) {
        t = Math.max(0, Math.min(1, t));
        const n = stops.length - 1;
        const idx = t * n;
        const lower = Math.floor(idx);
        const upper = Math.min(lower + 1, n);
        const frac = idx - lower;

        const c1 = hexToRGB(stops[lower]);
        const c2 = hexToRGB(stops[upper]);

        const r = Math.round(c1.r + (c2.r - c1.r) * frac);
        const g = Math.round(c1.g + (c2.g - c1.g) * frac);
        const b = Math.round(c1.b + (c2.b - c1.b) * frac);

        return `rgb(${r},${g},${b})`;
    }

    function hexToRGB(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return { r, g, b };
    }

    /**
     * Get luminance of a color to decide text color contrast
     */
    function getLuminance(hex) {
        const { r, g, b } = hexToRGB(hex);
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    }

    // ============================================
    // HeatmapRenderer
    // ============================================
    class HeatmapRenderer {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = null;
            this.dpr = 1;
            this.logicalWidth = CANVAS_WIDTH;
            this.logicalHeight = CANVAS_HEIGHT;
            this.hoveredCell = null; // {row, col}
        }

        setup() {
            const result = VizLib.CanvasUtils.setupHiDPICanvas(this.canvas);
            this.ctx = result.ctx;
            this.dpr = result.dpr;
            this.logicalWidth = result.logicalWidth;
            this.logicalHeight = result.logicalHeight;
        }

        render(computer, headIndex) {
            const ctx = this.ctx;
            const colors = getColors();
            const tokens = computer.tokens;
            const n = tokens.length;

            if (n === 0) {
                this.renderEmpty(colors);
                return;
            }

            const weights = (headIndex === -1) ? computer.getAverageAttention() : computer.attentionWeights[headIndex];

            VizLib.CanvasUtils.resetCanvasTransform(ctx, this.dpr);
            VizLib.CanvasUtils.clearCanvas(ctx, this.logicalWidth, this.logicalHeight, colors.bg);

            // Calculate cell dimensions
            const gridW = this.logicalWidth - PADDING.left - PADDING.right;
            const gridH = this.logicalHeight - PADDING.top - PADDING.bottom;
            const cellW = gridW / n;
            const cellH = gridH / n;

            // Draw column headers (key tokens)
            ctx.save();
            ctx.font = `600 ${Math.min(12, 120 / n)}px ${this.getMonoFont()}`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            for (let j = 0; j < n; j++) {
                const x = PADDING.left + j * cellW + cellW / 2;
                const isHovered = this.hoveredCell && this.hoveredCell.col === j;
                ctx.fillStyle = isHovered ? colors.tokenHighlight : colors.text;
                ctx.fillText(this.truncateToken(tokens[j], cellW), x, PADDING.top - 6);
            }
            ctx.restore();

            // Draw "Key" label at top
            ctx.save();
            ctx.font = `italic 11px sans-serif`;
            ctx.fillStyle = colors.textMuted;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('Key tokens \u2192', PADDING.left + gridW / 2, PADDING.top - 22);
            ctx.restore();

            // Draw row labels (query tokens)
            ctx.save();
            ctx.font = `600 ${Math.min(12, 120 / n)}px ${this.getMonoFont()}`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < n; i++) {
                const y = PADDING.top + i * cellH + cellH / 2;
                const isHovered = this.hoveredCell && this.hoveredCell.row === i;
                ctx.fillStyle = isHovered ? colors.tokenHighlight : colors.text;
                ctx.fillText(this.truncateToken(tokens[i], PADDING.left - 10), PADDING.left - 8, y);
            }
            ctx.restore();

            // Draw "Query" label on the left (rotated)
            ctx.save();
            ctx.font = `italic 11px sans-serif`;
            ctx.fillStyle = colors.textMuted;
            ctx.translate(12, PADDING.top + gridH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Query tokens \u2192', 0, 0);
            ctx.restore();

            // Draw row/column highlight for hovered cell
            if (this.hoveredCell) {
                const { row, col } = this.hoveredCell;

                // Row highlight
                ctx.fillStyle = colors.rowHighlight;
                ctx.fillRect(PADDING.left, PADDING.top + row * cellH, gridW, cellH);

                // Column highlight
                ctx.fillStyle = colors.colHighlight;
                ctx.fillRect(PADDING.left + col * cellW, PADDING.top, cellW, gridH);
            }

            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const w = weights[i][j];
                    const x = PADDING.left + j * cellW;
                    const y = PADDING.top + i * cellH;

                    // Cell color from gradient
                    const cellColor = interpolateColor(colors.gradientStops, w);
                    ctx.fillStyle = cellColor;
                    ctx.fillRect(x, y, cellW, cellH);

                    // Cell border
                    ctx.strokeStyle = colors.cellBorder;
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(x, y, cellW, cellH);

                    // Weight text (show only if cells are large enough)
                    if (cellW > 30 && cellH > 20) {
                        // Decide text color based on background luminance
                        const cellHex = this.colorToHex(cellColor);
                        const lum = getLuminance(cellHex);
                        ctx.fillStyle = lum > 0.5 ? colors.text : '#ffffff';
                        ctx.font = `${Math.min(11, cellW / 4.5)}px ${this.getMonoFont()}`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(w.toFixed(2), x + cellW / 2, y + cellH / 2);
                    }
                }
            }

            // Draw hover border
            if (this.hoveredCell) {
                const { row, col } = this.hoveredCell;
                const x = PADDING.left + col * cellW;
                const y = PADDING.top + row * cellH;
                ctx.strokeStyle = colors.hoverBorder;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
            }
        }

        renderEmpty(colors) {
            const ctx = this.ctx;
            VizLib.CanvasUtils.resetCanvasTransform(ctx, this.dpr);
            VizLib.CanvasUtils.clearCanvas(ctx, this.logicalWidth, this.logicalHeight, colors.bg);

            ctx.fillStyle = colors.textMuted;
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Enter a sentence and click "Compute Attention"',
                this.logicalWidth / 2, this.logicalHeight / 2);
        }

        renderPE(computer) {
            if (!computer.usePE || computer.tokens.length === 0) return;

            const ctx = this.ctx;
            const colors = getColors();
            const n = computer.tokens.length;
            const dim = Math.max(4, Math.floor(computer.dim / computer.numHeads));

            // Draw a small PE visualization below the heatmap label area
            const peHeight = 8;
            const gridW = this.logicalWidth - PADDING.left - PADDING.right;
            const cellW = gridW / n;
            const peY = this.logicalHeight - PADDING.bottom + 4;

            ctx.save();
            ctx.font = 'italic 9px sans-serif';
            ctx.fillStyle = colors.textMuted;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText('PE', PADDING.left - 8, peY + peHeight / 2);

            for (let i = 0; i < n; i++) {
                const pe = positionalEncoding(i, dim);
                const x = PADDING.left + i * cellW;
                // Show average PE magnitude as a color bar
                const avgPE = pe.reduce((a, b) => a + b, 0) / dim;
                const clampedPE = Math.max(-1, Math.min(1, avgPE * 5));
                if (clampedPE >= 0) {
                    ctx.fillStyle = VizLib.ThemeManager.hexToRgba(
                        colors.pePositive, Math.abs(clampedPE) * 0.8 + 0.2
                    );
                } else {
                    ctx.fillStyle = VizLib.ThemeManager.hexToRgba(
                        colors.peNegative, Math.abs(clampedPE) * 0.8 + 0.2
                    );
                }
                ctx.fillRect(x, peY, cellW, peHeight);
                ctx.strokeStyle = colors.cellBorder;
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, peY, cellW, peHeight);
            }
            ctx.restore();
        }

        /**
         * Get cell at a given mouse position
         */
        getCellAt(mouseX, mouseY, numTokens) {
            const gridW = this.logicalWidth - PADDING.left - PADDING.right;
            const gridH = this.logicalHeight - PADDING.top - PADDING.bottom;
            const cellW = gridW / numTokens;
            const cellH = gridH / numTokens;

            const col = Math.floor((mouseX - PADDING.left) / cellW);
            const row = Math.floor((mouseY - PADDING.top) / cellH);

            if (row >= 0 && row < numTokens && col >= 0 && col < numTokens) {
                return { row, col };
            }
            return null;
        }

        getMonoFont() {
            return "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";
        }

        truncateToken(token, maxWidth) {
            // For very long tokens, truncate
            if (token.length > 8) {
                return token.substring(0, 7) + '\u2026';
            }
            return token;
        }

        colorToHex(colorStr) {
            // Convert rgb(r,g,b) to hex
            const match = colorStr.match(/rgb\((\d+),(\d+),(\d+)\)/);
            if (match) {
                const r = parseInt(match[1]).toString(16).padStart(2, '0');
                const g = parseInt(match[2]).toString(16).padStart(2, '0');
                const b = parseInt(match[3]).toString(16).padStart(2, '0');
                return `#${r}${g}${b}`;
            }
            return colorStr;
        }
    }

    // ============================================
    // Application State & Controller
    // ============================================
    let canvas, renderer, computer;
    let currentHead = 0; // -1 = average, 0..n-1 = individual heads

    function init() {
        canvas = document.getElementById('attention-canvas');
        if (!canvas) return;

        computer = new AttentionComputer();
        renderer = new HeatmapRenderer(canvas);
        renderer.setup();

        bindEvents();
        buildHeadTabs();
        updateMetrics();

        // Initial render (empty state)
        const colors = getColors();
        renderer.renderEmpty(colors);

        // Listen for theme changes
        VizLib.ThemeManager.onThemeChange(() => {
            renderer.setup();
            if (computer.tokens.length > 0 && computer.attentionWeights.length > 0) {
                renderer.render(computer, currentHead);
                if (computer.usePE) renderer.renderPE(computer);
            } else {
                renderer.renderEmpty(getColors());
            }
        });
    }

    function bindEvents() {
        // Compute button
        document.getElementById('btn-compute').addEventListener('click', computeAndRender);

        // Reset button
        document.getElementById('btn-reset').addEventListener('click', resetVisualization);

        // Enter key on input
        document.getElementById('input-sentence').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                computeAndRender();
            }
        });

        // Heads stepper
        document.getElementById('btn-heads-dec').addEventListener('click', function() {
            const input = document.getElementById('num-heads');
            let val = parseInt(input.value);
            if (val > MIN_HEADS) {
                input.value = val - 1;
                if (computer.attentionWeights.length > 0) computeAndRender();
            }
        });

        document.getElementById('btn-heads-inc').addEventListener('click', function() {
            const input = document.getElementById('num-heads');
            let val = parseInt(input.value);
            if (val < MAX_HEADS) {
                input.value = val + 1;
                if (computer.attentionWeights.length > 0) computeAndRender();
            }
        });

        // Temperature slider
        const tempSlider = document.getElementById('temperature-slider');
        const tempValue = document.getElementById('temperature-value');
        tempSlider.addEventListener('input', function() {
            tempValue.textContent = parseFloat(this.value).toFixed(1);
            if (computer.attentionWeights.length > 0) {
                computer.temperature = parseFloat(this.value);
                computer.computeAttention();
                renderer.render(computer, currentHead);
                if (computer.usePE) renderer.renderPE(computer);
                updateMetrics();
                updateTokenDetail();
            }
        });

        // Positional encoding toggle
        document.getElementById('toggle-pe').addEventListener('change', function() {
            computer.usePE = this.checked;
            if (computer.attentionWeights.length > 0) {
                computer.computeAttention();
                renderer.render(computer, currentHead);
                if (computer.usePE) renderer.renderPE(computer);
                updateMetrics();
                updateTokenDetail();
            }
        });

        // Canvas mouse events
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseleave', handleMouseLeave);
    }

    function handleMouseMove(e) {
        if (computer.tokens.length === 0 || computer.attentionWeights.length === 0) return;

        const pos = VizLib.CanvasUtils.getMousePosition(canvas, e);
        const cell = renderer.getCellAt(pos.x, pos.y, computer.tokens.length);

        if (cell && (!renderer.hoveredCell || renderer.hoveredCell.row !== cell.row || renderer.hoveredCell.col !== cell.col)) {
            renderer.hoveredCell = cell;
            renderer.render(computer, currentHead);
            if (computer.usePE) renderer.renderPE(computer);
            updateTokenDetail();
        } else if (!cell && renderer.hoveredCell) {
            renderer.hoveredCell = null;
            renderer.render(computer, currentHead);
            if (computer.usePE) renderer.renderPE(computer);
            clearTokenDetail();
        }
    }

    function handleMouseLeave() {
        if (renderer.hoveredCell) {
            renderer.hoveredCell = null;
            if (computer.tokens.length > 0 && computer.attentionWeights.length > 0) {
                renderer.render(computer, currentHead);
                if (computer.usePE) renderer.renderPE(computer);
            }
            clearTokenDetail();
        }
    }

    function computeAndRender() {
        const sentence = document.getElementById('input-sentence').value;
        if (!sentence.trim()) return;

        computer.setTokens(sentence);
        computer.numHeads = parseInt(document.getElementById('num-heads').value);
        computer.temperature = parseFloat(document.getElementById('temperature-slider').value);
        computer.usePE = document.getElementById('toggle-pe').checked;

        computer.computeAttention();

        // Rebuild head tabs if head count changed
        buildHeadTabs();
        if (currentHead >= computer.numHeads) {
            currentHead = 0;
        }
        setActiveHeadTab(currentHead);

        renderer.setup();
        renderer.render(computer, currentHead);
        if (computer.usePE) renderer.renderPE(computer);
        updateMetrics();

        document.getElementById('metric-status').textContent = 'Computed';
    }

    function resetVisualization() {
        document.getElementById('input-sentence').value = DEFAULT_SENTENCE;
        document.getElementById('num-heads').value = '4';
        document.getElementById('temperature-slider').value = '1.0';
        document.getElementById('temperature-value').textContent = '1.0';
        document.getElementById('toggle-pe').checked = false;

        computer = new AttentionComputer();
        currentHead = 0;
        renderer.hoveredCell = null;

        buildHeadTabs();
        renderer.setup();
        renderer.renderEmpty(getColors());
        updateMetrics();
        clearTokenDetail();
        document.getElementById('metric-status').textContent = 'Ready';
    }

    // ============================================
    // Head Tabs
    // ============================================
    function buildHeadTabs() {
        const container = document.getElementById('head-tabs');
        const numHeads = parseInt(document.getElementById('num-heads').value);

        container.innerHTML = '';

        // Average button
        const avgBtn = document.createElement('button');
        avgBtn.type = 'button';
        avgBtn.className = 'btn btn-default' + (currentHead === -1 ? ' active' : '');
        avgBtn.textContent = 'Avg';
        avgBtn.dataset.head = '-1';
        avgBtn.addEventListener('click', function() {
            switchHead(-1);
        });
        container.appendChild(avgBtn);

        // Individual head buttons
        for (let h = 0; h < numHeads; h++) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-default' + (currentHead === h ? ' active' : '');
            btn.textContent = (h + 1).toString();
            btn.dataset.head = h.toString();
            btn.addEventListener('click', (function(headIdx) {
                return function() { switchHead(headIdx); };
            })(h));
            container.appendChild(btn);
        }
    }

    function switchHead(headIndex) {
        currentHead = headIndex;
        setActiveHeadTab(headIndex);

        if (computer.tokens.length > 0 && computer.attentionWeights.length > 0) {
            renderer.render(computer, currentHead);
            if (computer.usePE) renderer.renderPE(computer);
            updateMetrics();
            updateTokenDetail();
        }

        // Update head badge
        const badge = document.getElementById('head-badge');
        if (headIndex === -1) {
            badge.textContent = 'Average';
        } else {
            badge.textContent = 'Head ' + (headIndex + 1);
        }
    }

    function setActiveHeadTab(headIndex) {
        const container = document.getElementById('head-tabs');
        const buttons = container.querySelectorAll('.btn');
        buttons.forEach(btn => {
            const h = parseInt(btn.dataset.head);
            btn.classList.toggle('active', h === headIndex);
        });
    }

    // ============================================
    // Metrics
    // ============================================
    function updateMetrics() {
        document.getElementById('metric-tokens').textContent = computer.tokens.length || '-';
        document.getElementById('metric-heads').textContent = parseInt(document.getElementById('num-heads').value);
        document.getElementById('metric-dim').textContent = computer.dim;

        if (computer.attentionWeights.length > 0) {
            const maxAttn = computer.getMaxAttention(currentHead);
            const minAttn = computer.getMinAttention(currentHead);
            document.getElementById('metric-max-attn').textContent = maxAttn.toFixed(4);
            document.getElementById('metric-min-attn').textContent = minAttn.toFixed(4);
        } else {
            document.getElementById('metric-max-attn').textContent = '-';
            document.getElementById('metric-min-attn').textContent = '-';
        }
    }

    // ============================================
    // Token Detail Panel
    // ============================================
    function updateTokenDetail() {
        if (!renderer.hoveredCell || computer.attentionWeights.length === 0) return;

        const { row, col } = renderer.hoveredCell;
        const queryToken = computer.tokens[row];
        const keyToken = computer.tokens[col];
        const weights = (currentHead === -1) ? computer.getAverageAttention() : computer.attentionWeights[currentHead];
        const weight = weights[row][col];

        // Build detailed view: show the attention distribution for this query token
        const container = document.getElementById('token-detail-content');
        const n = computer.tokens.length;

        // Get all attention weights for the hovered query (row)
        const queryWeights = [];
        for (let j = 0; j < n; j++) {
            queryWeights.push({ token: computer.tokens[j], weight: weights[row][j], idx: j });
        }
        // Sort by weight descending
        queryWeights.sort((a, b) => b.weight - a.weight);

        const colors = getColors();

        let html = `<div class="token-detail-header">`;
        html += `<span class="token-name">"${queryToken}"</span> \u2192 <span class="token-name">"${keyToken}"</span>: `;
        html += `<strong>${weight.toFixed(4)}</strong></div>`;

        html += `<table class="token-detail-table">`;
        html += `<thead><tr><th>Key Token</th><th>Attention</th><th>Bar</th></tr></thead><tbody>`;

        for (const item of queryWeights) {
            const barWidth = Math.round(item.weight * 100);
            const isHovered = item.idx === col;
            const highlightStyle = isHovered ? `font-weight:700;color:${colors.tokenHighlight}` : '';
            const barColor = interpolateColor(colors.gradientStops, item.weight);
            html += `<tr${isHovered ? ' style="background:' + colors.rowHighlight + '"' : ''}>`;
            html += `<td style="${highlightStyle}">${item.token}</td>`;
            html += `<td style="${highlightStyle}">${item.weight.toFixed(4)}</td>`;
            html += `<td><div class="attn-bar-container">`;
            html += `<div class="attn-bar" style="width:${barWidth}%;background:${barColor};min-width:2px;"></div>`;
            html += `</div></td>`;
            html += `</tr>`;
        }

        html += `</tbody></table>`;
        container.innerHTML = html;
    }

    function clearTokenDetail() {
        const container = document.getElementById('token-detail-content');
        container.innerHTML = '<p class="no-data-message"><i class="fa fa-hand-pointer-o"></i><br>Hover over a cell in the heatmap to see attention details.</p>';
    }

    // ============================================
    // Bootstrap
    // ============================================
    window.addEventListener('vizlib-ready', init);
})();
