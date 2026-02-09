/**
 * Naive Bayes Classifier Visualizer
 *
 * Interactive text classification using Naive Bayes with spam/ham detection.
 * Features Laplace smoothing, word probability tables, prior/posterior
 * visualization, and step-by-step probability display.
 */
(function() {
    'use strict';

    // ============================================
    // Constants & Preset Training Data
    // ============================================
    const CANVAS_HEIGHT = 100;
    let CANVAS_WIDTH = 500;

    const DEFAULT_TRAINING_DATA = [
        // Spam messages
        { text: 'Win a free iPhone now click here', label: 'spam' },
        { text: 'Congratulations you won a million dollars', label: 'spam' },
        { text: 'Free money no credit check required', label: 'spam' },
        { text: 'Buy cheap pills online discount pharmacy', label: 'spam' },
        { text: 'You have been selected for a cash prize', label: 'spam' },
        { text: 'Claim your free reward today limited offer', label: 'spam' },
        { text: 'Make money fast work from home opportunity', label: 'spam' },
        { text: 'Urgent act now to claim your prize winner', label: 'spam' },
        { text: 'Get rich quick guaranteed income easy money', label: 'spam' },
        { text: 'Special promotion buy one get one free deal', label: 'spam' },
        { text: 'Exclusive offer just for you free gift card', label: 'spam' },
        { text: 'Click here to win big cash prize now', label: 'spam' },
        // Ham messages
        { text: 'Hey are you coming to the meeting tomorrow', label: 'ham' },
        { text: 'Can you pick up groceries on your way home', label: 'ham' },
        { text: 'The project deadline has been moved to Friday', label: 'ham' },
        { text: 'Let me know when you are available to talk', label: 'ham' },
        { text: 'Thanks for sending the report it looks good', label: 'ham' },
        { text: 'Are we still on for lunch today at noon', label: 'ham' },
        { text: 'I will be working from home tomorrow morning', label: 'ham' },
        { text: 'Please review the attached document and reply', label: 'ham' },
        { text: 'See you at the conference next week', label: 'ham' },
        { text: 'Happy birthday hope you have a great day', label: 'ham' },
        { text: 'Can we reschedule our appointment to next week', label: 'ham' },
        { text: 'The team meeting is at three this afternoon', label: 'ham' }
    ];

    // ============================================
    // NaiveBayesClassifier Class
    // ============================================
    class NaiveBayesClassifier {
        constructor() {
            this.alpha = 1.0;           // Laplace smoothing parameter
            this.classCounts = {};       // {spam: n, ham: n}
            this.classWordCounts = {};   // {spam: {word: count}, ham: {word: count}}
            this.classTotalWords = {};   // {spam: total, ham: total}
            this.vocabulary = new Set();
            this.totalDocs = 0;
            this.trainingData = [];
        }

        /**
         * Tokenize text into lowercase words, removing punctuation
         */
        tokenize(text) {
            return text.toLowerCase()
                .replace(/[^a-z0-9\s]/g, '')
                .split(/\s+/)
                .filter(w => w.length > 0);
        }

        /**
         * Train the classifier on a dataset
         */
        train(data) {
            this.trainingData = [...data];
            this.classCounts = {};
            this.classWordCounts = {};
            this.classTotalWords = {};
            this.vocabulary = new Set();
            this.totalDocs = 0;

            for (const item of data) {
                const label = item.label;
                const words = this.tokenize(item.text);

                // Count documents per class
                this.classCounts[label] = (this.classCounts[label] || 0) + 1;
                this.totalDocs++;

                // Initialize class word counts
                if (!this.classWordCounts[label]) {
                    this.classWordCounts[label] = {};
                    this.classTotalWords[label] = 0;
                }

                // Count words per class
                for (const word of words) {
                    this.vocabulary.add(word);
                    this.classWordCounts[label][word] = (this.classWordCounts[label][word] || 0) + 1;
                    this.classTotalWords[label]++;
                }
            }
        }

        /**
         * Get prior probability P(class)
         */
        getPrior(label) {
            return (this.classCounts[label] || 0) / this.totalDocs;
        }

        /**
         * Get likelihood P(word|class) with Laplace smoothing
         */
        getWordProbability(word, label) {
            const wordCount = (this.classWordCounts[label] && this.classWordCounts[label][word]) || 0;
            const totalWords = this.classTotalWords[label] || 0;
            const vocabSize = this.vocabulary.size;

            return (wordCount + this.alpha) / (totalWords + this.alpha * vocabSize);
        }

        /**
         * Get log-likelihood for a word given class
         */
        getLogWordProbability(word, label) {
            return Math.log(this.getWordProbability(word, label));
        }

        /**
         * Classify a message. Returns detailed breakdown.
         */
        classify(text) {
            const words = this.tokenize(text);
            const classes = Object.keys(this.classCounts);

            if (classes.length === 0 || words.length === 0) {
                return null;
            }

            const logScores = {};
            const wordDetails = [];

            // Compute log P(class) + sum log P(word|class) for each class
            for (const cls of classes) {
                logScores[cls] = Math.log(this.getPrior(cls));

                for (const word of words) {
                    logScores[cls] += this.getLogWordProbability(word, cls);
                }
            }

            // Compute per-word details
            for (const word of words) {
                const pSpam = this.getWordProbability(word, 'spam');
                const pHam = this.getWordProbability(word, 'ham');
                const logLR = Math.log(pSpam) - Math.log(pHam);

                wordDetails.push({
                    word,
                    pSpam,
                    pHam,
                    logLR,
                    inVocab: this.vocabulary.has(word)
                });
            }

            // Convert log-scores to probabilities via log-sum-exp (softmax)
            const maxLog = Math.max(...Object.values(logScores));
            let sumExp = 0;
            const expScores = {};
            for (const cls of classes) {
                expScores[cls] = Math.exp(logScores[cls] - maxLog);
                sumExp += expScores[cls];
            }

            const probabilities = {};
            for (const cls of classes) {
                probabilities[cls] = expScores[cls] / sumExp;
            }

            // Determine predicted class
            let predicted = classes[0];
            for (const cls of classes) {
                if (probabilities[cls] > probabilities[predicted]) {
                    predicted = cls;
                }
            }

            return {
                predicted,
                probabilities,
                logScores,
                wordDetails,
                words,
                priors: {
                    spam: this.getPrior('spam'),
                    ham: this.getPrior('ham')
                }
            };
        }

        /**
         * Get all vocabulary word probabilities for display
         */
        getVocabularyProbabilities() {
            const result = [];
            for (const word of this.vocabulary) {
                const pSpam = this.getWordProbability(word, 'spam');
                const pHam = this.getWordProbability(word, 'ham');
                result.push({
                    word,
                    pSpam,
                    pHam,
                    logLR: Math.log(pSpam) - Math.log(pHam)
                });
            }
            return result;
        }
    }

    // ============================================
    // PriorPosteriorRenderer Class
    // ============================================
    class PriorPosteriorRenderer {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = null;
            this.dpr = 1;
        }

        setup() {
            if (window.VizLib) {
                const result = VizLib.setupHiDPICanvas(this.canvas);
                this.ctx = result.ctx;
                this.dpr = result.dpr;
                CANVAS_WIDTH = result.logicalWidth;
            } else {
                this.ctx = this.canvas.getContext('2d');
            }
        }

        render(priors, posteriors) {
            const ctx = this.ctx;
            if (!ctx) return;

            const dpr = this.dpr;
            const isDark = document.documentElement.getAttribute('data-theme') === 'gruvbox-dark';

            // Clear
            if (window.VizLib) {
                VizLib.resetCanvasTransform(ctx, dpr);
                const bg = isDark ? '#1d2021' : '#fafafa';
                VizLib.clearCanvas(ctx, CANVAS_WIDTH, CANVAS_HEIGHT, bg);
            } else {
                ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            }

            const textColor = isDark ? '#ebdbb2' : '#333333';
            const mutedColor = isDark ? '#a89984' : '#6c757d';
            const spamColor = isDark ? '#fb4934' : '#dc3545';
            const hamColor = isDark ? '#b8bb26' : '#28a745';
            const trackColor = isDark ? '#3c3836' : '#e9ecef';
            const arrowColor = isDark ? '#a89984' : '#999';

            const barHeight = 18;
            const sectionWidth = (CANVAS_WIDTH - 80) / 2; // space for arrow
            const leftX = 20;
            const rightX = CANVAS_WIDTH - sectionWidth - 20;
            const arrowCenterX = CANVAS_WIDTH / 2;

            // --- Prior section ---
            const priorY = 10;
            ctx.fillStyle = mutedColor;
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Prior', leftX + sectionWidth / 2, priorY);

            // Prior spam bar
            const priorBarY = priorY + 18;
            this._drawBar(ctx, leftX, priorBarY, sectionWidth, barHeight, priors.spam, spamColor, trackColor);
            ctx.fillStyle = textColor;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Spam ' + (priors.spam * 100).toFixed(1) + '%', leftX + 4, priorBarY + barHeight / 2);

            // Prior ham bar
            const priorHamY = priorBarY + barHeight + 4;
            this._drawBar(ctx, leftX, priorHamY, sectionWidth, barHeight, priors.ham, hamColor, trackColor);
            ctx.fillStyle = textColor;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Ham ' + (priors.ham * 100).toFixed(1) + '%', leftX + 4, priorHamY + barHeight / 2);

            // --- Arrow ---
            const arrowY = priorBarY + barHeight + 2;
            ctx.strokeStyle = arrowColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(arrowCenterX - 18, arrowY);
            ctx.lineTo(arrowCenterX + 12, arrowY);
            ctx.stroke();
            // Arrowhead
            ctx.beginPath();
            ctx.moveTo(arrowCenterX + 12, arrowY);
            ctx.lineTo(arrowCenterX + 6, arrowY - 4);
            ctx.lineTo(arrowCenterX + 6, arrowY + 4);
            ctx.closePath();
            ctx.fillStyle = arrowColor;
            ctx.fill();

            ctx.fillStyle = mutedColor;
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('words', arrowCenterX, arrowY + 6);

            // --- Posterior section ---
            ctx.fillStyle = mutedColor;
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Posterior', rightX + sectionWidth / 2, priorY);

            // Posterior spam bar
            this._drawBar(ctx, rightX, priorBarY, sectionWidth, barHeight, posteriors.spam, spamColor, trackColor);
            ctx.fillStyle = textColor;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Spam ' + (posteriors.spam * 100).toFixed(1) + '%', rightX + 4, priorBarY + barHeight / 2);

            // Posterior ham bar
            this._drawBar(ctx, rightX, priorHamY, sectionWidth, barHeight, posteriors.ham, hamColor, trackColor);
            ctx.fillStyle = textColor;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Ham ' + (posteriors.ham * 100).toFixed(1) + '%', rightX + 4, priorHamY + barHeight / 2);
        }

        _drawBar(ctx, x, y, width, height, value, color, trackColor) {
            // Track
            ctx.fillStyle = trackColor;
            ctx.beginPath();
            ctx.moveTo(x + 3, y);
            ctx.lineTo(x + width - 3, y);
            ctx.quadraticCurveTo(x + width, y, x + width, y + 3);
            ctx.lineTo(x + width, y + height - 3);
            ctx.quadraticCurveTo(x + width, y + height, x + width - 3, y + height);
            ctx.lineTo(x + 3, y + height);
            ctx.quadraticCurveTo(x, y + height, x, y + height - 3);
            ctx.lineTo(x, y + 3);
            ctx.quadraticCurveTo(x, y, x + 3, y);
            ctx.fill();

            // Filled portion
            const filledWidth = Math.max(0, width * value);
            if (filledWidth > 0) {
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.7;
                ctx.beginPath();
                const fw = Math.min(filledWidth, width);
                ctx.moveTo(x + 3, y);
                ctx.lineTo(x + fw - (fw >= width ? 3 : 0), y);
                if (fw >= width) {
                    ctx.quadraticCurveTo(x + fw, y, x + fw, y + 3);
                    ctx.lineTo(x + fw, y + height - 3);
                    ctx.quadraticCurveTo(x + fw, y + height, x + fw - 3, y + height);
                } else {
                    ctx.lineTo(x + fw, y);
                    ctx.lineTo(x + fw, y + height);
                }
                ctx.lineTo(x + 3, y + height);
                ctx.quadraticCurveTo(x, y + height, x, y + height - 3);
                ctx.lineTo(x, y + 3);
                ctx.quadraticCurveTo(x, y, x + 3, y);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
        }

        resize() {
            if (window.VizLib) {
                const result = VizLib.setupHiDPICanvas(this.canvas);
                this.ctx = result.ctx;
                this.dpr = result.dpr;
                CANVAS_WIDTH = result.logicalWidth;
            }
        }
    }

    // ============================================
    // NaiveBayesVisualizer (Main Controller)
    // ============================================
    class NaiveBayesVisualizer {
        constructor() {
            this.classifier = new NaiveBayesClassifier();
            this.priorPosteriorRenderer = null;
            this.lastResult = null;
            this.showLogProbs = false;
            this.sortColumn = 'ratio';
            this.sortAscending = false;
        }

        init() {
            if (window.VizLib) {
                this._setup();
            } else {
                window.addEventListener('vizlib-ready', () => this._setup());
            }
        }

        _setup() {
            // Train on default data
            this.classifier.train(DEFAULT_TRAINING_DATA);

            // Setup prior/posterior canvas
            const ppCanvas = document.getElementById('prior-posterior-canvas');
            if (ppCanvas) {
                this.priorPosteriorRenderer = new PriorPosteriorRenderer(ppCanvas);
                this.priorPosteriorRenderer.setup();
            }

            // Bind events
            this._bindEvents();

            // Update metrics
            this._updateMetrics();

            // Render training data list
            this._renderTrainingData();
        }

        _bindEvents() {
            // Classify button
            const classifyBtn = document.getElementById('btn-classify');
            if (classifyBtn) {
                classifyBtn.addEventListener('click', () => this._onClassify());
            }

            // Enter key in textarea
            const input = document.getElementById('message-input');
            if (input) {
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this._onClassify();
                    }
                });
            }

            // Alpha slider
            const alphaSlider = document.getElementById('alpha-slider');
            if (alphaSlider) {
                alphaSlider.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    this.classifier.alpha = val;
                    document.getElementById('alpha-value').textContent = val.toFixed(1);
                    document.getElementById('metric-alpha').textContent = val.toFixed(1);
                    // Re-classify if there is a previous result
                    if (this.lastResult) {
                        this._onClassify();
                    }
                });
            }

            // Log probabilities toggle
            const logToggle = document.getElementById('show-log-probs');
            if (logToggle) {
                logToggle.addEventListener('change', (e) => {
                    this.showLogProbs = e.target.checked;
                    if (this.lastResult) {
                        this._renderWordTable(this.lastResult);
                    }
                });
            }

            // Sortable table headers
            document.querySelectorAll('.word-prob-table .sortable').forEach(th => {
                th.addEventListener('click', () => {
                    const col = th.getAttribute('data-sort');
                    if (this.sortColumn === col) {
                        this.sortAscending = !this.sortAscending;
                    } else {
                        this.sortColumn = col;
                        this.sortAscending = col === 'word';
                    }
                    // Update sort indicators
                    document.querySelectorAll('.word-prob-table .sortable').forEach(h => {
                        h.classList.remove('sort-active');
                    });
                    th.classList.add('sort-active');

                    if (this.lastResult) {
                        this._renderWordTable(this.lastResult);
                    }
                });
            });

            // Toggle training data panel
            const toggleBtn = document.getElementById('btn-toggle-training');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => {
                    const panel = document.getElementById('training-panel');
                    const icon = document.getElementById('training-toggle-icon');
                    if (panel.style.display === 'none') {
                        panel.style.display = '';
                        icon.className = 'fa fa-chevron-up';
                    } else {
                        panel.style.display = 'none';
                        icon.className = 'fa fa-chevron-down';
                    }
                });
            }

            // Add training example
            const addBtn = document.getElementById('btn-add-training');
            if (addBtn) {
                addBtn.addEventListener('click', () => this._onAddTraining());
            }

            // Enter key in new training text
            const newText = document.getElementById('new-training-text');
            if (newText) {
                newText.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this._onAddTraining();
                    }
                });
            }

            // Reset button
            const resetBtn = document.getElementById('btn-reset');
            if (resetBtn) {
                resetBtn.addEventListener('click', () => this._onReset());
            }

            // Resize handler
            let resizeTimeout;
            window.addEventListener('resize', () => {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(() => {
                    if (this.priorPosteriorRenderer) {
                        this.priorPosteriorRenderer.resize();
                        if (this.lastResult) {
                            this.priorPosteriorRenderer.render(
                                this.lastResult.priors,
                                this.lastResult.probabilities
                            );
                        }
                    }
                }, 100);
            });

            // Theme change
            document.addEventListener('themechange', () => {
                if (this.lastResult && this.priorPosteriorRenderer) {
                    this.priorPosteriorRenderer.render(
                        this.lastResult.priors,
                        this.lastResult.probabilities
                    );
                }
            });

            // Info tab button switching (for btn-group variant)
            document.querySelectorAll('.info-panel-tabs .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tabId = btn.getAttribute('data-tab');
                    // Deactivate all
                    btn.closest('.info-panel-tabs').querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    // Show target tab
                    const panel = btn.closest('.panel');
                    panel.querySelectorAll('.info-tab-content').forEach(t => t.classList.remove('active'));
                    const target = panel.querySelector('#tab-' + tabId);
                    if (target) target.classList.add('active');
                });
            });
        }

        _onClassify() {
            const input = document.getElementById('message-input');
            const text = input ? input.value.trim() : '';

            if (!text) {
                this._updateStatus('Enter a message to classify');
                return;
            }

            const result = this.classifier.classify(text);
            if (!result) {
                this._updateStatus('No training data');
                return;
            }

            this.lastResult = result;

            // Update result display
            this._renderResult(result);

            // Update word table
            this._renderWordTable(result);

            // Update prior/posterior canvas
            const ppSection = document.getElementById('prior-posterior-section');
            if (ppSection) {
                ppSection.style.display = '';
            }
            if (this.priorPosteriorRenderer) {
                this.priorPosteriorRenderer.render(
                    result.priors,
                    result.probabilities
                );
            }

            // Update metrics
            this._updateMetrics(result);
        }

        _renderResult(result) {
            const display = document.getElementById('result-display');
            if (display) {
                display.style.display = '';
            }

            // Update badge
            const badge = document.getElementById('result-badge');
            if (badge) {
                badge.className = 'viz-badge result-badge ' + result.predicted;
                badge.textContent = result.predicted === 'spam' ? 'SPAM' : 'HAM';
            }

            // Update probability bars
            const spamProb = result.probabilities.spam || 0;
            const hamProb = result.probabilities.ham || 0;

            const spamBar = document.getElementById('spam-bar');
            const hamBar = document.getElementById('ham-bar');
            const spamProbEl = document.getElementById('spam-prob');
            const hamProbEl = document.getElementById('ham-prob');

            if (spamBar) spamBar.style.width = (spamProb * 100).toFixed(1) + '%';
            if (hamBar) hamBar.style.width = (hamProb * 100).toFixed(1) + '%';

            if (this.showLogProbs) {
                if (spamProbEl) spamProbEl.textContent = 'log: ' + (result.logScores.spam || 0).toFixed(4);
                if (hamProbEl) hamProbEl.textContent = 'log: ' + (result.logScores.ham || 0).toFixed(4);
            } else {
                if (spamProbEl) spamProbEl.textContent = (spamProb * 100).toFixed(1) + '%';
                if (hamProbEl) hamProbEl.textContent = (hamProb * 100).toFixed(1) + '%';
            }
        }

        _renderWordTable(result) {
            const tbody = document.getElementById('word-prob-tbody');
            if (!tbody) return;

            const details = [...result.wordDetails];

            // Sort
            details.sort((a, b) => {
                let valA, valB;
                switch (this.sortColumn) {
                    case 'word': valA = a.word; valB = b.word; break;
                    case 'pSpam': valA = a.pSpam; valB = b.pSpam; break;
                    case 'pHam': valA = a.pHam; valB = b.pHam; break;
                    case 'ratio': valA = a.logLR; valB = b.logLR; break;
                    default: valA = a.logLR; valB = b.logLR;
                }
                if (typeof valA === 'string') {
                    return this.sortAscending ? valA.localeCompare(valB) : valB.localeCompare(valA);
                }
                return this.sortAscending ? valA - valB : valB - valA;
            });

            let html = '';
            for (const d of details) {
                const ratioClass = d.logLR > 0.1 ? 'ratio-positive' :
                                   d.logLR < -0.1 ? 'ratio-negative' : 'ratio-neutral';
                const vocabClass = d.inVocab ? '' : ' word-highlight';
                const ratioSign = d.logLR > 0 ? '+' : '';

                html += '<tr class="' + vocabClass + '">';
                html += '<td>' + this._escapeHtml(d.word) + (d.inVocab ? '' : ' <small title="Not in training vocabulary">(new)</small>') + '</td>';
                html += '<td>' + d.pSpam.toFixed(6) + '</td>';
                html += '<td>' + d.pHam.toFixed(6) + '</td>';
                html += '<td class="' + ratioClass + '">' + ratioSign + d.logLR.toFixed(4) + '</td>';
                html += '</tr>';
            }

            // Log probability breakdown
            if (this.showLogProbs && result) {
                html += '<tr><td colspan="4" class="log-prob-section">';
                html += '<div class="log-prob-header">Log-Probability Breakdown</div>';

                // Spam
                html += '<div class="log-prob-term"><span>log P(Spam)</span><span>' + Math.log(result.priors.spam).toFixed(4) + '</span></div>';
                for (const d of result.wordDetails) {
                    html += '<div class="log-prob-term"><span>+ log P(' + this._escapeHtml(d.word) + '|Spam)</span><span>' + Math.log(d.pSpam).toFixed(4) + '</span></div>';
                }
                html += '<div class="log-prob-term total"><span>Total log P(Spam|msg)</span><span>' + (result.logScores.spam || 0).toFixed(4) + '</span></div>';

                html += '<br>';

                // Ham
                html += '<div class="log-prob-term"><span>log P(Ham)</span><span>' + Math.log(result.priors.ham).toFixed(4) + '</span></div>';
                for (const d of result.wordDetails) {
                    html += '<div class="log-prob-term"><span>+ log P(' + this._escapeHtml(d.word) + '|Ham)</span><span>' + Math.log(d.pHam).toFixed(4) + '</span></div>';
                }
                html += '<div class="log-prob-term total"><span>Total log P(Ham|msg)</span><span>' + (result.logScores.ham || 0).toFixed(4) + '</span></div>';

                html += '</td></tr>';
            }

            tbody.innerHTML = html;

            // Update word count badge
            const badge = document.getElementById('word-count-badge');
            if (badge) {
                badge.textContent = details.length + ' words';
            }
        }

        _renderTrainingData() {
            const list = document.getElementById('training-list');
            if (!list) return;

            let html = '';
            for (let i = 0; i < this.classifier.trainingData.length; i++) {
                const item = this.classifier.trainingData[i];
                html += '<div class="training-item ' + item.label + '">';
                html += '<span class="training-item-label ' + item.label + '">' + item.label + '</span>';
                html += '<span class="training-item-text">' + this._escapeHtml(item.text) + '</span>';
                html += '<span class="training-item-remove" data-index="' + i + '" title="Remove"><i class="fa fa-times"></i></span>';
                html += '</div>';
            }

            list.innerHTML = html;

            // Bind remove buttons
            list.querySelectorAll('.training-item-remove').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const idx = parseInt(e.currentTarget.getAttribute('data-index'));
                    this._onRemoveTraining(idx);
                });
            });
        }

        _onAddTraining() {
            const textEl = document.getElementById('new-training-text');
            const classEl = document.getElementById('new-training-class');
            const text = textEl ? textEl.value.trim() : '';
            const label = classEl ? classEl.value : 'spam';

            if (!text) return;

            // Add to training data and retrain
            const newData = [...this.classifier.trainingData, { text, label }];
            this.classifier.train(newData);

            // Clear input
            if (textEl) textEl.value = '';

            // Re-render
            this._renderTrainingData();
            this._updateMetrics();
            this._updateStatus('Added training example');

            // Re-classify if there is a previous result
            if (this.lastResult) {
                this._onClassify();
            }
        }

        _onRemoveTraining(index) {
            const data = [...this.classifier.trainingData];
            if (index >= 0 && index < data.length) {
                data.splice(index, 1);
                this.classifier.train(data);
                this._renderTrainingData();
                this._updateMetrics();
                this._updateStatus('Removed training example');

                if (this.lastResult) {
                    this._onClassify();
                }
            }
        }

        _onReset() {
            this.classifier.train(DEFAULT_TRAINING_DATA);
            this.lastResult = null;

            // Reset UI
            const input = document.getElementById('message-input');
            if (input) input.value = '';

            const display = document.getElementById('result-display');
            if (display) display.style.display = 'none';

            const ppSection = document.getElementById('prior-posterior-section');
            if (ppSection) ppSection.style.display = 'none';

            const badge = document.getElementById('result-badge');
            if (badge) {
                badge.className = 'viz-badge result-badge';
                badge.textContent = 'Ready';
            }

            const tbody = document.getElementById('word-prob-tbody');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Classify a message to see word probabilities</td></tr>';
            }

            const wordBadge = document.getElementById('word-count-badge');
            if (wordBadge) wordBadge.textContent = '0 words';

            // Reset slider
            const alphaSlider = document.getElementById('alpha-slider');
            if (alphaSlider) {
                alphaSlider.value = 1;
                this.classifier.alpha = 1.0;
                document.getElementById('alpha-value').textContent = '1.0';
            }

            // Reset log probs toggle
            const logToggle = document.getElementById('show-log-probs');
            if (logToggle) {
                logToggle.checked = false;
                this.showLogProbs = false;
            }

            this._renderTrainingData();
            this._updateMetrics();
            this._updateStatus('Reset to defaults');
        }

        _updateMetrics(result) {
            const vocabEl = document.getElementById('metric-vocab');
            const spamCountEl = document.getElementById('metric-spam-count');
            const hamCountEl = document.getElementById('metric-ham-count');
            const classEl = document.getElementById('metric-classification');
            const confEl = document.getElementById('metric-confidence');
            const alphaEl = document.getElementById('metric-alpha');

            if (vocabEl) vocabEl.textContent = this.classifier.vocabulary.size;
            if (spamCountEl) spamCountEl.textContent = this.classifier.classCounts.spam || 0;
            if (hamCountEl) hamCountEl.textContent = this.classifier.classCounts.ham || 0;
            if (alphaEl) alphaEl.textContent = this.classifier.alpha.toFixed(1);

            if (result) {
                if (classEl) {
                    classEl.innerHTML = result.predicted === 'spam'
                        ? '<span style="color: var(--nb-spam-color);">Spam</span>'
                        : '<span style="color: var(--nb-ham-color);">Ham</span>';
                }
                if (confEl) {
                    const conf = Math.max(result.probabilities.spam || 0, result.probabilities.ham || 0);
                    confEl.textContent = (conf * 100).toFixed(1) + '%';
                }
            } else {
                if (classEl) classEl.textContent = '-';
                if (confEl) confEl.textContent = '-';
            }
        }

        _updateStatus(text) {
            const el = document.getElementById('metric-status');
            if (el) el.textContent = text;
        }

        _escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    }

    // ============================================
    // Initialize on DOM ready
    // ============================================
    document.addEventListener('DOMContentLoaded', () => {
        const visualizer = new NaiveBayesVisualizer();
        visualizer.init();
    });

})();
