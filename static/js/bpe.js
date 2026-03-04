/**
 * BPE (Byte Pair Encoding) Visualizer
 *
 * Animated visualization of the BPE subword tokenization algorithm:
 * split text into characters → count pair frequencies → merge most frequent →
 * grow vocabulary. Pre-computes all merge steps, then reveals them via animation.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_W = 720;
    const CANVAS_H = 700;
    const MONO = "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";
    const EOW = '_';

    // Zone Y positions
    const ZONE = {
        A: { y: 20,  h: 200 }, // Current Tokenization
        B: { y: 240, h: 180 }, // Top Pair Frequencies
        C: { y: 440, h: 90  }, // Merge Operation
        D: { y: 545, h: 140 }, // Merge Rules + Vocabulary
    };

    // Preset corpora
    const PRESETS = {
        'default': ['low', 'low', 'lowest', 'newer', 'wider'],
        'hello':   ['hello', 'world', 'hello', 'there', 'help', 'held'],
    };

    // ============================================
    // BPE Algorithm Functions
    // ============================================
    function splitIntoChars(words) {
        // Returns array of token arrays, one per word. Each token is a string.
        // Appends EOW marker to end of each word.
        return words.map(w => {
            const chars = w.split('');
            chars.push(EOW);
            return chars;
        });
    }

    function countPairs(tokenizedWords) {
        const freq = {};
        for (const tokens of tokenizedWords) {
            for (let i = 0; i < tokens.length - 1; i++) {
                const key = tokens[i] + '\t' + tokens[i + 1];
                freq[key] = (freq[key] || 0) + 1;
            }
        }
        return freq;
    }

    function findBestPair(pairFreqs) {
        let bestKey = null;
        let bestCount = 0;
        for (const key in pairFreqs) {
            if (pairFreqs[key] > bestCount) {
                bestCount = pairFreqs[key];
                bestKey = key;
            }
        }
        if (!bestKey) return null;
        const parts = bestKey.split('\t');
        return { a: parts[0], b: parts[1], count: bestCount };
    }

    function applyMerge(tokenizedWords, pair) {
        const merged = pair.a + pair.b;
        return tokenizedWords.map(tokens => {
            const result = [];
            let i = 0;
            while (i < tokens.length) {
                if (i < tokens.length - 1 && tokens[i] === pair.a && tokens[i + 1] === pair.b) {
                    result.push(merged);
                    i += 2;
                } else {
                    result.push(tokens[i]);
                    i++;
                }
            }
            return result;
        });
    }

    function countTotalTokens(tokenizedWords) {
        let total = 0;
        for (const tokens of tokenizedWords) total += tokens.length;
        return total;
    }

    function getVocabulary(tokenizedWords) {
        const vocab = new Set();
        for (const tokens of tokenizedWords) {
            for (const t of tokens) vocab.add(t);
        }
        return Array.from(vocab).sort();
    }

    // ============================================
    // Pre-compute all merge steps
    // ============================================
    function precomputeMerges(words, maxMerges) {
        let tokenized = splitIntoChars(words);
        const initialTokens = countTotalTokens(tokenized);

        const history = [];

        // Step 0: initial state
        history.push({
            step: 0,
            tokens: tokenized.map(t => t.slice()),
            pairFreqs: countPairs(tokenized),
            bestPair: null,
            rule: null,
            vocabulary: getVocabulary(tokenized),
            totalTokens: initialTokens,
        });

        for (let i = 0; i < maxMerges; i++) {
            const pairFreqs = countPairs(tokenized);
            const best = findBestPair(pairFreqs);
            if (!best) break;

            const merged = best.a + best.b;
            tokenized = applyMerge(tokenized, best);

            history.push({
                step: i + 1,
                tokens: tokenized.map(t => t.slice()),
                pairFreqs: pairFreqs,
                bestPair: best,
                rule: { from: [best.a, best.b], to: merged, num: i + 1 },
                vocabulary: getVocabulary(tokenized),
                totalTokens: countTotalTokens(tokenized),
            });
        }

        return { history, initialTokens };
    }

    // ============================================
    // Theme-aware color helper
    // ============================================
    function getThemeColors() {
        const s = getComputedStyle(document.documentElement);
        const g = name => s.getPropertyValue(name).trim();
        return {
            charBg:       g('--bpe-char-bg'),
            charBorder:   g('--bpe-char-border'),
            charText:     g('--bpe-char-text'),
            mergedBg:     g('--bpe-merged-bg'),
            mergedBorder: g('--bpe-merged-border'),
            mergedText:   g('--bpe-merged-text'),
            activeBg:     g('--bpe-active-bg'),
            activeBorder: g('--bpe-active-border'),
            activeText:   g('--bpe-active-text'),
            activeGlow:   g('--bpe-active-glow'),
            eowBg:        g('--bpe-eow-bg'),
            eowBorder:    g('--bpe-eow-border'),
            eowText:      g('--bpe-eow-text'),
            barFill:      g('--bpe-bar-fill'),
            barHighlight: g('--bpe-bar-highlight'),
            barBg:        g('--bpe-bar-bg'),
            arrowColor:   g('--bpe-arrow-color'),
            arrowGlow:    g('--bpe-arrow-glow'),
            vocabPillBg:  g('--bpe-vocab-pill-bg'),
            vocabPillBorder: g('--bpe-vocab-pill-border'),
            vocabPillText:   g('--bpe-vocab-pill-text'),
            vocabNewBg:      g('--bpe-vocab-new-bg'),
            vocabNewBorder:  g('--bpe-vocab-new-border'),
            vocabNewText:    g('--bpe-vocab-new-text'),
            ruleText:        g('--bpe-rule-text'),
            ruleHighlight:   g('--bpe-rule-highlight'),
            sectionTitle:    g('--bpe-section-title'),
            labelColor:      g('--bpe-label-color'),
            canvasText:      g('--bpe-canvas-text'),
            canvasBg:        g('--viz-canvas-bg'),
            textMuted:       g('--viz-text-muted'),
        };
    }

    // ============================================
    // Main Visualizer Class
    // ============================================
    let clamp;

    class BPEVisualizer {
        constructor() {
            this.canvas = document.getElementById('bpe-canvas');
            this.ctx = this.canvas.getContext('2d');

            // HiDPI setup
            if (window.VizLib && window.VizLib.CanvasUtils) {
                window.VizLib.CanvasUtils.setupHiDPICanvas(this.canvas);
            }

            // State
            this.words = PRESETS['default'].slice();
            this.maxMerges = 10;
            this.speed = 5;
            this.mergeHistory = [];
            this.initialTokens = 0;

            // Animation state machine
            this.phase = 'IDLE';
            this.currentMergeStep = 0; // which merge we're animating (1-indexed in history)
            this.phaseProgress = 0;
            this.phaseStartTime = 0;
            this.isProcessing = false;
            this.animId = null;

            // DOM elements
            this.corpusSelect = document.getElementById('corpus-select');
            this.customCorpusRow = document.getElementById('custom-corpus-row');
            this.customCorpusInput = document.getElementById('custom-corpus');
            this.maxMergesValue = document.getElementById('max-merges-value');
            this.btnMergesDown = document.getElementById('btn-merges-down');
            this.btnMergesUp = document.getElementById('btn-merges-up');
            this.btnRun = document.getElementById('btn-run');
            this.btnStep = document.getElementById('btn-step');
            this.btnReset = document.getElementById('btn-reset');
            this.speedSlider = document.getElementById('speed-slider');
            this.speedValue = document.getElementById('speed-value');

            this.bindEvents();
            this.reset();
        }

        bindEvents() {
            this.btnRun.addEventListener('click', () => this.runAll());
            this.btnStep.addEventListener('click', () => this.stepOnce());
            this.btnReset.addEventListener('click', () => this.reset());

            this.speedSlider.addEventListener('input', () => {
                this.speed = parseInt(this.speedSlider.value);
                this.speedValue.textContent = this.speed;
            });

            this.corpusSelect.addEventListener('change', () => {
                if (this.corpusSelect.value === 'custom') {
                    this.customCorpusRow.classList.add('visible');
                } else {
                    this.customCorpusRow.classList.remove('visible');
                }
                this.reset();
            });

            this.customCorpusInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') this.reset();
            });

            this.btnMergesDown.addEventListener('click', () => {
                if (this.maxMerges > 1) {
                    this.maxMerges--;
                    this.maxMergesValue.textContent = this.maxMerges;
                    this.reset();
                }
            });

            this.btnMergesUp.addEventListener('click', () => {
                if (this.maxMerges < 20) {
                    this.maxMerges++;
                    this.maxMergesValue.textContent = this.maxMerges;
                    this.reset();
                }
            });

            document.addEventListener('themechange', () => this.draw());
        }

        // ============================================
        // Corpus helpers
        // ============================================
        getCorpusWords() {
            const sel = this.corpusSelect.value;
            if (sel === 'custom') {
                const text = this.customCorpusInput.value.trim();
                if (!text) return PRESETS['default'].slice();
                return text.split(/[,\s]+/).filter(w => w.length > 0).map(w => w.toLowerCase());
            }
            return (PRESETS[sel] || PRESETS['default']).slice();
        }

        // ============================================
        // Reset — pre-compute and show initial state
        // ============================================
        reset() {
            this.stopAnimation();
            this.words = this.getCorpusWords();

            const { history, initialTokens } = precomputeMerges(this.words, this.maxMerges);
            this.mergeHistory = history;
            this.initialTokens = initialTokens;

            this.phase = 'IDLE';
            this.currentMergeStep = 0;
            this.phaseProgress = 1;
            this.isProcessing = false;

            this.updateMetrics();
            this.draw();
        }

        // ============================================
        // Phase management
        // ============================================
        getPhaseDuration() {
            return Math.max(200, 1400 - this.speed * 120);
        }

        getNextPhase() {
            switch (this.phase) {
                case 'IDLE':           return 'SHOW_INPUT';
                case 'SHOW_INPUT':     return 'SPLIT_CHARS';
                case 'SPLIT_CHARS':
                    this.currentMergeStep = 0;
                    if (this.mergeHistory.length > 1) return 'SCAN_PAIRS';
                    return 'COMPLETE';
                case 'SCAN_PAIRS':     return 'HIGHLIGHT_PAIR';
                case 'HIGHLIGHT_PAIR': return 'ANIMATE_MERGE';
                case 'ANIMATE_MERGE':  return 'UPDATE_TOKENS';
                case 'UPDATE_TOKENS':
                    this.currentMergeStep++;
                    if (this.currentMergeStep < this.mergeHistory.length - 1) return 'SCAN_PAIRS';
                    return 'COMPLETE';
                default:               return 'COMPLETE';
            }
        }

        advancePhase() {
            const next = this.getNextPhase();
            if (next === this.phase && next === 'COMPLETE') return false;

            this.phase = next;
            this.phaseProgress = 0;
            this.phaseStartTime = performance.now();

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
                this.draw();
            }
        }

        runAll() {
            if (this.isProcessing) return;
            if (this.phase === 'COMPLETE') {
                // Re-run: reset to IDLE but keep pre-computed data
                this.phase = 'IDLE';
                this.currentMergeStep = 0;
                this.phaseProgress = 1;
            }

            this.isProcessing = true;
            this.btnRun.disabled = true;

            const animate = (now) => {
                if (!this.isProcessing) return;

                const elapsed = now - this.phaseStartTime;
                const duration = this.getPhaseDuration();
                this.phaseProgress = clamp(elapsed / duration, 0, 1);

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

                this.animId = requestAnimationFrame(animate);
            };

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
            this.btnRun.disabled = false;
        }

        // ============================================
        // Metrics
        // ============================================
        updateMetrics() {
            const phaseNames = {
                'IDLE': 'Idle', 'SHOW_INPUT': 'Show Input', 'SPLIT_CHARS': 'Split Characters',
                'SCAN_PAIRS': 'Scanning Pairs', 'HIGHLIGHT_PAIR': 'Best Pair',
                'ANIMATE_MERGE': 'Merging', 'UPDATE_TOKENS': 'Updating',
                'COMPLETE': 'Complete'
            };

            document.getElementById('metric-phase').textContent = phaseNames[this.phase] || this.phase;

            const totalMerges = this.mergeHistory.length - 1;
            const displayStep = this.getCurrentDisplayStep();
            document.getElementById('metric-merge-step').textContent = displayStep + ' / ' + totalMerges;

            const state = this.getCurrentState();
            document.getElementById('metric-vocab-size').textContent = state.vocabulary.length;
            document.getElementById('metric-total-tokens').textContent = state.totalTokens;

            const compression = this.initialTokens / Math.max(1, state.totalTokens);
            document.getElementById('metric-compression').textContent = compression.toFixed(2) + 'x';

            // Best pair
            const mergeState = this.getMergeState();
            if (mergeState && mergeState.bestPair) {
                const bp = mergeState.bestPair;
                document.getElementById('metric-best-pair').textContent =
                    '("' + bp.a + '", "' + bp.b + '") × ' + bp.count;
            } else {
                document.getElementById('metric-best-pair').textContent = '-';
            }
        }

        getCurrentDisplayStep() {
            if (this.phase === 'IDLE' || this.phase === 'SHOW_INPUT' || this.phase === 'SPLIT_CHARS') return 0;
            if (this.phase === 'COMPLETE') return this.mergeHistory.length - 1;
            // During merge loop, currentMergeStep is 0-indexed but points to the merge we're doing
            if (this.phase === 'UPDATE_TOKENS') return this.currentMergeStep + 1;
            return this.currentMergeStep;
        }

        // Get the token state to display (before or after current merge)
        getCurrentState() {
            if (this.phase === 'IDLE') {
                return { tokens: [], vocabulary: [], totalTokens: 0 };
            }
            if (this.phase === 'SHOW_INPUT') {
                return { tokens: [], vocabulary: [], totalTokens: 0 };
            }

            // Index into history: after UPDATE_TOKENS for step N, show history[N+1]
            let idx = 0;
            if (this.phase === 'SPLIT_CHARS') {
                idx = 0;
            } else if (this.phase === 'SCAN_PAIRS' || this.phase === 'HIGHLIGHT_PAIR' || this.phase === 'ANIMATE_MERGE') {
                idx = this.currentMergeStep;
            } else if (this.phase === 'UPDATE_TOKENS') {
                idx = this.currentMergeStep + 1;
            } else if (this.phase === 'COMPLETE') {
                idx = this.mergeHistory.length - 1;
            }

            idx = Math.min(idx, this.mergeHistory.length - 1);
            return this.mergeHistory[idx];
        }

        // Get the merge-in-progress state (the step we're about to apply or just applied)
        getMergeState() {
            if (this.phase === 'SCAN_PAIRS' || this.phase === 'HIGHLIGHT_PAIR' || this.phase === 'ANIMATE_MERGE') {
                const nextIdx = this.currentMergeStep + 1;
                if (nextIdx < this.mergeHistory.length) return this.mergeHistory[nextIdx];
            }
            if (this.phase === 'UPDATE_TOKENS') {
                const idx = this.currentMergeStep + 1;
                if (idx < this.mergeHistory.length) return this.mergeHistory[idx];
            }
            return null;
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

            this.drawZoneA(ctx, C); // Current Tokenization
            this.drawZoneB(ctx, C); // Pair Frequencies
            this.drawZoneC(ctx, C); // Merge Operation
            this.drawZoneD(ctx, C); // Rules + Vocabulary
        }

        // ============================================
        // Zone A: Current Tokenization
        // ============================================
        drawZoneA(ctx, C) {
            const { y: zy, h: zh } = ZONE.A;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('CURRENT TOKENIZATION', 15, zy + 2);

            if (this.phase === 'IDLE') return;

            // Show input phase: fade in the raw text
            if (this.phase === 'SHOW_INPUT') {
                ctx.save();
                ctx.globalAlpha = this.phaseProgress;
                ctx.font = '14px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Corpus: ' + this.words.join(', '), CANVAS_W / 2, zy + zh / 2);
                ctx.restore();
                return;
            }

            // Draw token boxes for each word
            const state = this.getCurrentState();
            if (!state.tokens || state.tokens.length === 0) return;

            const mergeState = this.getMergeState();
            const bestPair = mergeState ? mergeState.bestPair : null;
            const isHighlighting = (this.phase === 'HIGHLIGHT_PAIR' || this.phase === 'ANIMATE_MERGE');

            const padX = 15;
            const padY = zy + 20;
            const boxH = 26;
            const boxGap = 3;
            const wordGap = 14;
            const maxRowW = CANVAS_W - padX * 2;

            let curY = padY;

            for (let wi = 0; wi < state.tokens.length; wi++) {
                const tokens = state.tokens[wi];
                // Measure row width
                ctx.font = '12px ' + MONO;
                const boxWidths = tokens.map(t => ctx.measureText(t).width + 16);
                const rowW = boxWidths.reduce((a, b) => a + b, 0) + (tokens.length - 1) * boxGap;
                const scale = rowW > maxRowW ? maxRowW / rowW : 1;

                // Word label
                ctx.font = '10px ' + MONO;
                ctx.fillStyle = C.labelColor;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(this.words[wi] + ':', padX, curY + boxH / 2);
                const labelW = ctx.measureText(this.words[wi] + ':').width + 10;

                let curX = padX + labelW;

                for (let ti = 0; ti < tokens.length; ti++) {
                    const token = tokens[ti];
                    const bw = boxWidths[ti] * scale;

                    // Determine token type and colors
                    let bg, border, text;
                    const isEOW = token === EOW;
                    const isMerged = token.length > 1 && !isEOW;
                    const isActive = isHighlighting && bestPair &&
                        ti < tokens.length - 1 &&
                        tokens[ti] === bestPair.a && tokens[ti + 1] === bestPair.b;
                    const isActiveSecond = isHighlighting && bestPair &&
                        ti > 0 &&
                        tokens[ti - 1] === bestPair.a && tokens[ti] === bestPair.b;

                    if (isActive || isActiveSecond) {
                        bg = C.activeBg;
                        border = C.activeBorder;
                        text = C.activeText;
                    } else if (isEOW) {
                        bg = C.eowBg;
                        border = C.eowBorder;
                        text = C.eowText;
                    } else if (isMerged) {
                        bg = C.mergedBg;
                        border = C.mergedBorder;
                        text = C.mergedText;
                    } else {
                        bg = C.charBg;
                        border = C.charBorder;
                        text = C.charText;
                    }

                    // Animation: merge sliding
                    let drawX = curX;
                    if (this.phase === 'ANIMATE_MERGE' && bestPair) {
                        if (isActive) {
                            // Slide slightly right
                            drawX = curX + (bw * 0.15) * this.phaseProgress;
                        } else if (isActiveSecond) {
                            // Slide slightly left
                            drawX = curX - (bw * 0.15) * this.phaseProgress;
                        }
                    }

                    // Glow for active
                    if ((isActive || isActiveSecond) && this.phase === 'HIGHLIGHT_PAIR') {
                        ctx.save();
                        ctx.shadowColor = C.activeGlow;
                        ctx.shadowBlur = 8 + 4 * Math.sin(this.phaseProgress * Math.PI);
                        ctx.fillStyle = bg;
                        this.roundRect(ctx, drawX, curY, bw, boxH, 4);
                        ctx.fill();
                        ctx.restore();
                    }

                    // Box
                    ctx.fillStyle = bg;
                    ctx.strokeStyle = border;
                    ctx.lineWidth = (isActive || isActiveSecond) ? 2 : 1;
                    this.roundRect(ctx, drawX, curY, bw, boxH, 4);
                    ctx.fill();
                    ctx.stroke();

                    // Token text
                    ctx.fillStyle = text;
                    ctx.font = '12px ' + MONO;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(token, drawX + bw / 2, curY + boxH / 2);

                    curX += bw + boxGap * scale;
                }

                curY += boxH + wordGap;
                if (curY > zy + zh - 10) break; // Don't overflow zone
            }
        }

        // ============================================
        // Zone B: Top Pair Frequencies
        // ============================================
        drawZoneB(ctx, C) {
            const { y: zy, h: zh } = ZONE.B;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('PAIR FREQUENCIES', 15, zy + 2);

            // Only show during merge loop or complete
            const showPhases = ['SCAN_PAIRS', 'HIGHLIGHT_PAIR', 'ANIMATE_MERGE', 'UPDATE_TOKENS', 'COMPLETE'];
            if (!showPhases.includes(this.phase)) return;

            // Get pair freqs from current state (before merge)
            let pairFreqs;
            if (this.phase === 'COMPLETE') {
                // Show last step's freqs
                const lastIdx = this.mergeHistory.length - 1;
                pairFreqs = countPairs(this.mergeHistory[lastIdx].tokens);
            } else {
                const state = this.getCurrentState();
                pairFreqs = state.pairFreqs || {};
            }

            // Sort and take top 8
            const sorted = Object.entries(pairFreqs)
                .map(([key, count]) => {
                    const parts = key.split('\t');
                    return { a: parts[0], b: parts[1], count };
                })
                .sort((a, b) => b.count - a.count)
                .slice(0, 8);

            if (sorted.length === 0) return;

            const mergeState = this.getMergeState();
            const bestPair = mergeState ? mergeState.bestPair : null;

            const barAreaX = 140;
            const barAreaW = CANVAS_W - barAreaX - 30;
            const barH = 16;
            const barGap = 4;
            const startY = zy + 20;
            const maxCount = sorted[0].count;

            for (let i = 0; i < sorted.length; i++) {
                const pair = sorted[i];
                const by = startY + i * (barH + barGap);
                if (by + barH > zy + zh) break;

                const isBest = bestPair && pair.a === bestPair.a && pair.b === bestPair.b;

                // Pair label
                ctx.font = '11px ' + MONO;
                ctx.fillStyle = isBest ? C.arrowColor : C.labelColor;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText('("' + pair.a + '", "' + pair.b + '")', barAreaX - 10, by + barH / 2);

                // Bar background
                ctx.fillStyle = C.barBg;
                this.roundRect(ctx, barAreaX, by, barAreaW, barH, 3);
                ctx.fill();

                // Bar fill
                let barProgress = pair.count / maxCount;
                if (this.phase === 'SCAN_PAIRS') {
                    barProgress *= this.phaseProgress;
                }
                const barW = Math.max(4, barAreaW * barProgress);
                ctx.fillStyle = isBest ? C.barHighlight : C.barFill;
                this.roundRect(ctx, barAreaX, by, barW, barH, 3);
                ctx.fill();

                // Glow for best pair
                if (isBest && this.phase === 'HIGHLIGHT_PAIR') {
                    ctx.save();
                    ctx.shadowColor = C.activeGlow;
                    ctx.shadowBlur = 6 + 3 * Math.sin(this.phaseProgress * Math.PI);
                    ctx.fillStyle = C.barHighlight;
                    this.roundRect(ctx, barAreaX, by, barW, barH, 3);
                    ctx.fill();
                    ctx.restore();
                }

                // Count label
                ctx.fillStyle = isBest ? C.arrowColor : C.canvasText;
                ctx.font = '10px ' + MONO;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText('×' + pair.count, barAreaX + barW + 6, by + barH / 2);
            }
        }

        // ============================================
        // Zone C: Merge Operation
        // ============================================
        drawZoneC(ctx, C) {
            const { y: zy, h: zh } = ZONE.C;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('MERGE OPERATION', 15, zy + 2);

            const mergeState = this.getMergeState();
            if (!mergeState || !mergeState.bestPair) return;

            const showPhases = ['HIGHLIGHT_PAIR', 'ANIMATE_MERGE', 'UPDATE_TOKENS'];
            if (!showPhases.includes(this.phase)) return;

            const bp = mergeState.bestPair;
            const merged = bp.a + bp.b;
            const centerX = CANVAS_W / 2;
            const centerY = zy + zh / 2 + 5;

            // Left side: ("a", "b")
            const leftText = '("' + bp.a + '", "' + bp.b + '")';
            ctx.font = 'bold 16px ' + MONO;
            const leftW = ctx.measureText(leftText).width;
            const rightText = '"' + merged + '"';
            const rightW = ctx.measureText(rightText).width;
            const arrowW = 60;
            const totalW = leftW + arrowW + rightW;
            const startX = centerX - totalW / 2;

            // Draw left tokens
            ctx.fillStyle = C.activeText;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(leftText, startX, centerY);

            // Arrow
            const arrowStartX = startX + leftW + 10;
            const arrowEndX = arrowStartX + arrowW - 20;

            let arrowProgress = 1;
            if (this.phase === 'ANIMATE_MERGE') {
                arrowProgress = this.phaseProgress;
            } else if (this.phase === 'HIGHLIGHT_PAIR') {
                arrowProgress = 0.3;
            }

            // Arrow glow
            ctx.save();
            ctx.shadowColor = C.arrowGlow;
            ctx.shadowBlur = 6;
            ctx.strokeStyle = C.arrowColor;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(arrowStartX, centerY);
            const curArrowEnd = arrowStartX + (arrowEndX - arrowStartX) * arrowProgress;
            ctx.lineTo(curArrowEnd, centerY);
            ctx.stroke();

            // Arrow head
            if (arrowProgress > 0.5) {
                ctx.fillStyle = C.arrowColor;
                ctx.beginPath();
                ctx.moveTo(curArrowEnd + 8, centerY);
                ctx.lineTo(curArrowEnd - 2, centerY - 5);
                ctx.lineTo(curArrowEnd - 2, centerY + 5);
                ctx.closePath();
                ctx.fill();
            }
            ctx.restore();

            // Right side: merged result
            const mergedAlpha = this.phase === 'ANIMATE_MERGE' ? this.phaseProgress : (this.phase === 'UPDATE_TOKENS' ? 1 : 0.3);
            ctx.save();
            ctx.globalAlpha = mergedAlpha;
            ctx.font = 'bold 16px ' + MONO;
            ctx.fillStyle = C.mergedText;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(rightText, arrowEndX + 20, centerY);
            ctx.restore();

            // Rule number
            if (mergeState.rule) {
                ctx.font = '10px ' + MONO;
                ctx.fillStyle = C.labelColor;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText('Rule #' + mergeState.rule.num, centerX, centerY + 20);
            }
        }

        // ============================================
        // Zone D: Merge Rules + Vocabulary
        // ============================================
        drawZoneD(ctx, C) {
            const { y: zy, h: zh } = ZONE.D;
            const halfW = CANVAS_W / 2;

            // ---- Left half: Merge Rules ----
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('MERGE RULES', 15, zy + 2);

            const ruleStartY = zy + 20;
            const ruleLineH = 16;
            const currentDisplayStep = this.getCurrentDisplayStep();

            // Collect all rules up to current step
            const rules = [];
            for (let i = 1; i < this.mergeHistory.length; i++) {
                const h = this.mergeHistory[i];
                if (h.rule && h.step <= currentDisplayStep) {
                    rules.push(h.rule);
                }
                // Also include rule being animated
                if (h.rule && h.step === currentDisplayStep + 1 &&
                    (this.phase === 'ANIMATE_MERGE' || this.phase === 'UPDATE_TOKENS')) {
                    rules.push(h.rule);
                }
            }

            for (let i = 0; i < rules.length; i++) {
                const rule = rules[i];
                const ry = ruleStartY + i * ruleLineH;
                if (ry + ruleLineH > zy + zh) break;

                const isLatest = i === rules.length - 1 &&
                    (this.phase === 'UPDATE_TOKENS' || this.phase === 'ANIMATE_MERGE');

                ctx.font = '11px ' + MONO;
                ctx.fillStyle = isLatest ? C.ruleHighlight : C.ruleText;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const ruleText = rule.num + '. ("' + rule.from[0] + '","' + rule.from[1] + '") \u2192 "' + rule.to + '"';
                ctx.fillText(ruleText, 20, ry + ruleLineH / 2);
            }

            if (rules.length === 0) {
                ctx.font = '11px ' + MONO;
                ctx.fillStyle = C.labelColor;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText('(no merges yet)', 20, ruleStartY + ruleLineH / 2);
            }

            // ---- Right half: Vocabulary ----
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('VOCABULARY', halfW + 15, zy + 2);

            const state = this.getCurrentState();
            const vocab = state.vocabulary || [];
            if (vocab.length === 0) return;

            // Determine which tokens are new (added in latest merge)
            const mergeState = this.getMergeState();
            const newToken = (mergeState && mergeState.rule &&
                (this.phase === 'UPDATE_TOKENS' || this.phase === 'ANIMATE_MERGE'))
                ? mergeState.rule.to : null;

            const pillH = 20;
            const pillGap = 4;
            const pillPadX = 8;
            const vocabStartY = zy + 20;
            const vocabAreaW = halfW - 30;

            ctx.font = '10px ' + MONO;

            let px = halfW + 15;
            let py = vocabStartY;

            for (let i = 0; i < vocab.length; i++) {
                const token = vocab[i];
                const tw = ctx.measureText(token).width + pillPadX * 2;
                const pw = Math.max(tw, 24);

                // Wrap to next line
                if (px + pw > halfW + 15 + vocabAreaW) {
                    px = halfW + 15;
                    py += pillH + pillGap;
                }

                if (py + pillH > zy + zh) break;

                const isNew = token === newToken;

                // Pill background
                ctx.fillStyle = isNew ? C.vocabNewBg : C.vocabPillBg;
                ctx.strokeStyle = isNew ? C.vocabNewBorder : C.vocabPillBorder;
                ctx.lineWidth = isNew ? 1.5 : 1;
                this.roundRect(ctx, px, py, pw, pillH, pillH / 2);
                ctx.fill();
                ctx.stroke();

                // Pill text
                ctx.fillStyle = isNew ? C.vocabNewText : C.vocabPillText;
                ctx.font = '10px ' + MONO;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(token, px + pw / 2, py + pillH / 2);

                px += pw + pillGap;
            }
        }

        // ============================================
        // Drawing helpers
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
    }

    // ============================================
    // Bootstrap
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new BPEVisualizer();

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
