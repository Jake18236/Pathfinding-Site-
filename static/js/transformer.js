/**
 * Transformer & Attention Architecture Diagram Visualizer
 *
 * DOM-based interactive block diagram with self-attention.
 * Shows full sequence tokenization + embedding table, then
 * per-head attention heatmaps in a scrollable carousel.
 *
 * RESPONSIVE: No hardcoded px in inline styles. All sizing via CSS custom
 * properties or relative units (em, %). CSS handles responsive scaling.
 */
(function () {
    'use strict';

    var CU, DU;
    function resolveLibs() {
        CU = window.VizLib.CanvasUtils;
        DU = window.VizLib.DomUtils;
    }

    // ─── Seeded PRNG (mulberry32) ──────────────────────────────────────
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = seed + 0x6D2B79F5 | 0;
            var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    // ─── Math Utilities ───────────────────────────────────────────────
    function makeWeights(rows, cols, rng, scale) {
        scale = scale || 1;
        var m = [];
        for (var i = 0; i < rows; i++) {
            var row = [];
            for (var j = 0; j < cols; j++) {
                row.push((rng() * 2 - 1) * scale);
            }
            m.push(row);
        }
        return m;
    }

    function makeVec(n, rngFn, scale) {
        scale = scale || 0.1;
        var v = [];
        for (var i = 0; i < n; i++) v.push((typeof rngFn === 'function' ? rngFn() : 0) * scale);
        return v;
    }

    function matVecMul(M, v) {
        var out = [];
        for (var i = 0; i < M.length; i++) {
            var sum = 0;
            for (var j = 0; j < v.length; j++) sum += M[i][j] * v[j];
            out.push(sum);
        }
        return out;
    }

    function vecAdd(a, b) {
        var out = [];
        for (var i = 0; i < a.length; i++) out.push(a[i] + b[i]);
        return out;
    }

    function dot(a, b) {
        var sum = 0;
        for (var i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    function softmax(v) {
        var maxV = Math.max.apply(null, v);
        var exps = v.map(function (x) { return Math.exp(x - maxV); });
        var sum = exps.reduce(function (a, b) { return a + b; }, 0);
        return exps.map(function (e) { return e / sum; });
    }

    // ─── Constants: Pre-tokenized Examples ────────────────────────────
    var EXAMPLES = [
        {
            label: 'The cat sat on the mat',
            tokens: [
                { word: 'The', id: 464 },
                { word: ' cat', id: 3797 },
                { word: ' sat', id: 3290 },
                { word: ' on', id: 319 },
                { word: ' the', id: 262 },
                { word: ' mat', id: 2603 }
            ]
        },
        {
            label: 'I like cute kitties and',
            tokens: [
                { word: 'I', id: 40 },
                { word: ' like', id: 1299 },
                { word: ' cute', id: 23172 },
                { word: ' k', id: 372 },
                { word: 'itt', id: 1387 },
                { word: 'ies', id: 566 },
                { word: ' and', id: 326 }
            ]
        },
        {
            label: 'King is to queen as man',
            tokens: [
                { word: 'King', id: 15839 },
                { word: ' is', id: 318 },
                { word: ' to', id: 284 },
                { word: ' queen', id: 16599 },
                { word: ' as', id: 355 },
                { word: ' man', id: 582 }
            ]
        }
    ];

    // ─── Color Helpers ───────────────────────────────────────────────
    function getCSS(prop) {
        return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
    }

    function heatmapColor(value) {
        // value in [0, 1] — interpolate from low attention to high attention
        var lo = CU.hexToRgb(getCSS('--tf-attn-zero') || '#f0f4ff');
        var hi = CU.hexToRgb(getCSS('--tf-attn-high') || '#d73027');
        var rgb = CU.colorLerp(lo, hi, value);
        return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    }

    function heatmapTextColor(value) {
        return value > 0.45
            ? (getCSS('--tf-weight-text-light') || '#ffffff')
            : (getCSS('--tf-weight-text-dark') || '#333333');
    }

    function displayWord(w) {
        return w.replace(/^ /, '\u00b7');
    }

    function absMaxOf(values) {
        var m = 0;
        for (var i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]));
        return m < 0.001 ? 1 : m;
    }

    function vecHeatmapColor(value, absMax) {
        var lo = CU.hexToRgb(getCSS('--tf-attn-zero') || '#f0f4ff');
        var hi = CU.hexToRgb(getCSS('--tf-pe-positive') || '#2196f3');
        var t = (value + absMax) / (2 * absMax);
        var rgb = CU.colorLerp(lo, hi, t);
        return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    }

    // ─── DOM element helper ──────────────────────────────────────────
    function el(tag, attrs, children) {
        var node = document.createElement(tag);
        if (attrs) {
            for (var k in attrs) {
                if (k === 'className') node.className = attrs[k];
                else if (k === 'textContent') node.textContent = attrs[k];
                else if (k === 'innerHTML') node.innerHTML = attrs[k];
                else node.setAttribute(k, attrs[k]);
            }
        }
        if (children) {
            if (!Array.isArray(children)) children = [children];
            for (var i = 0; i < children.length; i++) {
                if (typeof children[i] === 'string') {
                    node.appendChild(document.createTextNode(children[i]));
                } else if (children[i]) {
                    node.appendChild(children[i]);
                }
            }
        }
        return node;
    }

    // ─── Embedding lookup (hash-based, deterministic per token ID) ──
    function getEmbedding(tokenId, dim, seed) {
        var rng = mulberry32(seed + tokenId * 7919);
        var emb = [];
        for (var i = 0; i < dim; i++) {
            emb.push((rng() * 2 - 1) * 0.5);
        }
        return emb;
    }

    function positionalEncoding(pos, dim) {
        var pe = [];
        for (var i = 0; i < dim; i++) {
            var angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
            pe.push((i % 2 === 0) ? Math.sin(angle) : Math.cos(angle));
        }
        return pe;
    }

    // ─── Model ───────────────────────────────────────────────────────
    function buildModel(embedDim, numHeads) {
        var rng = mulberry32(42);
        var headDim = Math.floor(embedDim / numHeads);
        var heads = [];
        for (var h = 0; h < numHeads; h++) {
            heads.push({
                W_Q: makeWeights(headDim, embedDim, rng, Math.sqrt(2.0 / (embedDim + headDim))),
                W_K: makeWeights(headDim, embedDim, rng, Math.sqrt(2.0 / (embedDim + headDim))),
                W_V: makeWeights(headDim, embedDim, rng, Math.sqrt(2.0 / (embedDim + headDim)))
            });
        }
        return {
            embSeed: 12345,
            embedDim: embedDim,
            numHeads: numHeads,
            headDim: headDim,
            heads: heads
        };
    }

    function runForward(model, tokens, temperature, usePE) {
        var n = tokens.length;

        // 1. Embeddings
        var embeddings = [];
        for (var i = 0; i < n; i++) {
            embeddings.push(getEmbedding(tokens[i].id, model.embedDim, model.embSeed));
        }

        // 2. Positional encoding
        var pe = [];
        var embedded = [];
        for (var i = 0; i < n; i++) {
            var p = positionalEncoding(i, model.embedDim);
            pe.push(p);
            embedded.push(usePE ? vecAdd(embeddings[i], p) : embeddings[i].slice());
        }

        // 3. Per-head attention
        var heads = [];
        for (var h = 0; h < model.numHeads; h++) {
            var Q = [], K = [], V = [];
            for (var i = 0; i < n; i++) {
                Q.push(matVecMul(model.heads[h].W_Q, embedded[i]));
                K.push(matVecMul(model.heads[h].W_K, embedded[i]));
                V.push(matVecMul(model.heads[h].W_V, embedded[i]));
            }

            var scale = Math.sqrt(model.headDim);
            var scores = [];
            var weights = [];
            for (var i = 0; i < n; i++) {
                var row = [];
                for (var j = 0; j < n; j++) {
                    var rawScore = dot(Q[i], K[j]) / scale / temperature;
                    // Add position-based bias per head for visually distinct patterns
                    var bias = 0;
                    var headType = h % 4;
                    if (headType === 0) {
                        // "Local" head: attend to nearby tokens
                        bias = -0.5 * Math.abs(i - j);
                    } else if (headType === 1) {
                        // "Self" head: strongly attend to own position
                        bias = (i === j) ? 2.0 : -0.3;
                    } else if (headType === 2) {
                        // "Beginning" head: attend to early positions
                        bias = -0.4 * j;
                    } else {
                        // "Relative" head: attend to previous token
                        bias = (j === Math.max(0, i - 1)) ? 1.8 : -0.2;
                    }
                    row.push(rawScore + bias);
                }
                scores.push(row);
                weights.push(softmax(row));
            }

            // Attended output: weights @ V
            var output = [];
            for (var i = 0; i < n; i++) {
                var out = [];
                for (var d = 0; d < model.headDim; d++) out.push(0);
                for (var j = 0; j < n; j++) {
                    for (var d = 0; d < model.headDim; d++) {
                        out[d] += weights[i][j] * V[j][d];
                    }
                }
                output.push(out);
            }

            heads.push({ Q: Q, K: K, V: V, scores: scores, weights: weights, output: output });
        }

        // 4. Concatenate head outputs
        var concat = [];
        for (var i = 0; i < n; i++) {
            var c = [];
            for (var h = 0; h < model.numHeads; h++) {
                c = c.concat(heads[h].output[i]);
            }
            concat.push(c);
        }

        return {
            embeddings: embeddings,
            pe: pe,
            embedded: embedded,
            heads: heads,
            concat: concat
        };
    }

    // ─── Heatmap Grid Builder ────────────────────────────────────────
    function buildHeatmapGrid(weights, tokens, activeToken) {
        var n = tokens.length;
        var grid = el('div', { className: 'tf-heatmap' });
        grid.style.gridTemplateColumns = 'auto repeat(' + n + ', 1fr)';

        // Corner cell
        grid.appendChild(el('span', { className: 'tf-heatmap-corner' }));

        // Top row: key token labels
        for (var j = 0; j < n; j++) {
            var topLabel = el('span', {
                className: 'tf-heatmap-label-top' + (j === activeToken ? ' col-active' : ''),
                textContent: displayWord(tokens[j].word).substring(0, 4),
                title: tokens[j].word
            });
            topLabel.setAttribute('data-col', j);
            grid.appendChild(topLabel);
        }

        // Rows: query tokens + cells
        for (var i = 0; i < n; i++) {
            var rowLabel = el('span', {
                className: 'tf-heatmap-label-left' + (i === activeToken ? ' row-active' : ''),
                textContent: displayWord(tokens[i].word).substring(0, 4),
                title: tokens[i].word
            });
            rowLabel.setAttribute('data-row', i);
            grid.appendChild(rowLabel);

            for (var j = 0; j < n; j++) {
                var w = weights[i][j];
                var cell = el('span', {
                    className: 'tf-heatmap-cell' +
                        (i === activeToken ? ' row-active' : '') +
                        (j === activeToken ? ' col-active' : ''),
                    title: displayWord(tokens[i].word) + ' \u2192 ' + displayWord(tokens[j].word) + ': ' + w.toFixed(4)
                });
                cell.style.backgroundColor = heatmapColor(w);
                cell.style.color = heatmapTextColor(w);
                cell.setAttribute('data-row', i);
                cell.setAttribute('data-col', j);
                grid.appendChild(cell);
            }
        }

        return grid;
    }

    // ─── Heatmap Row Builder (for embedding/vector display) ─────────
    function buildHeatmapRow(values, maxCells, ariaLabel) {
        var n = values.length;
        var show = Math.min(n, maxCells || 16);
        var am = absMaxOf(values);

        var wrap = el('div', {
            className: 'tf-heatmap-row',
            'aria-label': ariaLabel || 'Vector'
        });

        for (var i = 0; i < show; i++) {
            var cell = el('span', {
                className: 'tf-heatmap-cell',
                title: values[i].toFixed(4)
            });
            cell.style.backgroundColor = vecHeatmapColor(values[i], am);
            wrap.appendChild(cell);
        }

        if (n > show) {
            wrap.appendChild(el('span', { className: 'tf-heatmap-ellipsis', textContent: '\u2026' }));
        }

        return wrap;
    }

    // ─── Attention Bar Chart Builder ─────────────────────────────────
    function buildAttnBarChart(weights, tokens, activeToken, colorIdx) {
        var n = tokens.length;
        var row = weights[activeToken];
        var maxW = 0;
        for (var j = 0; j < n; j++) maxW = Math.max(maxW, row[j]);
        if (maxW < 0.001) maxW = 1;

        var chart = el('div', { className: 'tf-hist-vchart' });

        for (var j = 0; j < n; j++) {
            var col = el('div', { className: 'tf-bar-col' });

            var pct = el('span', { className: 'tf-bar-pct' });
            pct.textContent = (row[j] * 100).toFixed(1) + '%';
            col.appendChild(pct);

            var track = el('div', { className: 'tf-bar-track' });
            var fill = el('div', { className: 'tf-bar-fill' });
            fill.style.height = (row[j] / maxW * 100) + '%';
            if (j === activeToken) {
                fill.style.background = 'var(--viz-class-' + colorIdx + ')';
            }
            track.appendChild(fill);
            col.appendChild(track);

            var label = el('span', { className: 'tf-bar-label' });
            label.textContent = displayWord(tokens[j].word).substring(0, 5);
            col.appendChild(label);

            chart.appendChild(col);
        }

        return chart;
    }

    // ─── Main Visualization ──────────────────────────────────────────

    function TransformerViz() {
        this.container = document.getElementById('tf-diagram');
        this.embedDim = 64;
        this.numHeads = 4;
        this.temperature = 1.0;
        this.usePE = true;
        this.exIdx = 0;
        this.activeToken = 0;
        this.activeHead = 0;
        this.expandedBlocks = { tokens: true, embed: false, qkv: false, output: true };
        this.model = null;
        this.result = null;
        this.dom = {};

        this.rebuildModel();
        this.buildDiagram();
        this.applyBlockStates();
        this.updateDiagram();
        this.bindEvents();
    }

    TransformerViz.prototype.rebuildModel = function () {
        this.model = buildModel(this.embedDim, this.numHeads);
        this.result = runForward(this.model, EXAMPLES[this.exIdx].tokens, this.temperature, this.usePE);
        if (this.activeToken >= EXAMPLES[this.exIdx].tokens.length) {
            this.activeToken = EXAMPLES[this.exIdx].tokens.length - 1;
        }
        if (this.activeHead >= this.numHeads) {
            this.activeHead = this.numHeads - 1;
        }
        this.updateStepperDisplay();
    };

    // ─── Build the DOM skeleton ──────────────────────────────────────

    TransformerViz.prototype.buildDiagram = function () {
        var self = this;
        var c = this.container;
        c.innerHTML = '';

        var tokens = EXAMPLES[this.exIdx].tokens;

        // ── Token block (always expanded) ──
        var tokensBlock = el('div', {
            id: 'tf-tokens-block',
            className: 'tf-block tf-block--embed tf-block--token-expand expanded',
            'aria-label': 'Tokenizer'
        });

        var tokensExpanded = el('div', { className: 'tf-expanded-content', id: 'tf-tokens-expanded' });
        var tokRow = el('div', { className: 'tf-token-row' });

        // Labels column
        var labels = el('div', { className: 'tf-token-labels' });
        labels.appendChild(el('span', { className: 'tf-token-label', textContent: 'Pos' }));
        labels.appendChild(el('span', { className: 'tf-token-label', textContent: 'Token' }));
        labels.appendChild(el('span', { className: 'tf-token-label', textContent: 'ID' }));
        tokRow.appendChild(labels);

        // One card per token
        this.dom.wordBoxes = [];
        this.dom.tokenCards = [];
        for (var i = 0; i < tokens.length; i++) {
            var card = el('div', { className: 'tf-token-card tf-token-color-' + (i % 10) });
            card.style.cursor = 'pointer';
            (function (idx, viz) {
                card.addEventListener('click', function (e) {
                    e.stopPropagation();
                    viz.activeToken = idx;
                    viz.updateStepperDisplay();
                    viz.updateDiagram();
                });
            })(i, self);

            card.appendChild(el('span', {
                className: 'tf-token-card-time',
                textContent: 't=' + i
            }));

            var wordSpan = el('span', {
                className: 'tf-token-card-word',
                textContent: tokens[i].word
            });
            this.dom.wordBoxes.push(wordSpan);
            card.appendChild(wordSpan);

            var idText = String(tokens[i].id);
            if (i === 0) idText = '[' + idText;
            if (i === tokens.length - 1) idText = idText + ']';
            card.appendChild(el('span', {
                className: 'tf-token-card-id',
                textContent: idText
            }));

            this.dom.tokenCards.push(card);
            tokRow.appendChild(card);
        }

        tokensExpanded.appendChild(tokRow);
        tokensBlock.appendChild(tokensExpanded);
        tokensBlock.appendChild(el('span', {
            className: 'tf-block-dim',
            id: 'tf-tokens-dim',
            textContent: '<1, ' + tokens.length + '>'
        }));

        // Shared wrapper: tokens + embed
        var tokEmbWrap = el('div', { className: 'tf-tok-embed-group' });
        tokEmbWrap.appendChild(tokensBlock);

        // ── Embeddings block (expandable) ──
        var embedBlock = el('div', {
            id: 'tf-embed-block',
            className: 'tf-block tf-block--embed tf-block--clickable',
            'aria-label': 'Embedding Layer'
        });

        var embedTitleText = this.usePE ? 'Embedding + PE' : 'Embedding';
        embedBlock.appendChild(el('span', {
            className: 'tf-block-title',
            id: 'tf-embed-title',
            textContent: embedTitleText
        }));
        embedBlock.appendChild(el('span', {
            className: 'tf-block-dim',
            id: 'tf-embed-dim',
            textContent: '<' + tokens.length + ', ' + this.embedDim + '>'
        }));

        // Collapsed: mini summary
        var embedCollapsed = el('div', { className: 'tf-collapsed-content', id: 'tf-embed-collapsed' });
        embedBlock.appendChild(embedCollapsed);

        // Expanded: embedding table
        var embedExpanded = el('div', { className: 'tf-expanded-content', id: 'tf-embed-expanded' });
        var embedTable = el('div', {
            className: 'tf-embeddings',
            id: 'tf-embeddings',
            'aria-label': 'Embedding Table'
        });

        this.dom.embedRows = [];
        for (var i = 0; i < tokens.length; i++) {
            var eRow = el('div', { className: 'tf-embed-row' });

            var tokenIdBox = el('span', {
                className: 'tf-token-id-box tf-token-color-' + (i % 10),
                textContent: '[' + tokens[i].id + ']'
            });
            eRow.appendChild(tokenIdBox);

            var embedVec = el('span', {
                className: 'tf-embed-vector tf-token-color-' + (i % 10),
                textContent: '<1, ' + this.embedDim + '>'
            });
            eRow.appendChild(embedVec);

            this.dom.embedRows.push({
                row: eRow,
                idBox: tokenIdBox,
                vector: embedVec
            });
            embedTable.appendChild(eRow);
        }
        embedExpanded.appendChild(embedTable);
        embedBlock.appendChild(embedExpanded);
        tokEmbWrap.appendChild(embedBlock);
        c.appendChild(tokEmbWrap);

        // ── Connector: embed → QKV ──
        var conn1 = el('div', { className: 'tf-connector' }, [
            el('div', { className: 'tf-connector-line' }),
            el('div', { className: 'tf-connector-arrow' })
        ]);
        c.appendChild(conn1);

        // ── Q/K/V Projection block (expandable) ──
        var qkvBlock = el('div', {
            id: 'tf-qkv-block',
            className: 'tf-block tf-block--qkv tf-block--clickable',
            'aria-label': 'Q/K/V Projection'
        });
        qkvBlock.appendChild(el('span', {
            className: 'tf-block-title',
            textContent: 'Q / K / V Projection'
        }));
        qkvBlock.appendChild(el('span', {
            className: 'tf-block-dim',
            id: 'tf-qkv-dim',
            innerHTML: '&lt;' + tokens.length + ', ' + this.embedDim + '&gt; &rarr; &lt;' + tokens.length + ', ' + this.model.headDim + '&gt; &times; 3'
        }));

        // Collapsed: Q_i, K_i, V_i summary for active token
        var qkvCollapsed = el('div', { className: 'tf-collapsed-content', id: 'tf-qkv-collapsed' });
        qkvBlock.appendChild(qkvCollapsed);

        // Expanded: show Q, K, V projections for active token with vector display
        var qkvExpanded = el('div', { className: 'tf-expanded-content', id: 'tf-qkv-expanded' });
        qkvBlock.appendChild(qkvExpanded);

        c.appendChild(qkvBlock);

        // ── Connector: QKV → head carousel ──
        var conn2 = el('div', { className: 'tf-connector' }, [
            el('div', { className: 'tf-connector-line' }),
            el('div', { className: 'tf-connector-arrow' })
        ]);
        c.appendChild(conn2);

        // ── Head Carousel (THE CORE) ──
        var headChain = el('div', {
            id: 'tf-head-chain',
            className: 'tf-head-chain',
            'aria-label': 'Multi-Head Attention Carousel'
        });

        var headTrack = el('div', { className: 'tf-head-track', id: 'tf-head-track' });
        var headTrackInner = el('div', { className: 'tf-head-track-inner' });

        // Start spacer
        headTrackInner.appendChild(el('div', { className: 'tf-head-track-spacer' }));

        this.dom.headCards = [];
        for (var h = 0; h < this.numHeads; h++) {
            var headCard = el('div', {
                className: 'tf-head-card' + (h === this.activeHead ? ' active' : ''),
                'aria-label': 'Attention Head ' + h
            });
            headCard.style.cursor = 'pointer';

            headCard.appendChild(el('span', {
                className: 'tf-head-title',
                textContent: 'Head ' + h
            }));

            // Heatmap grid
            var heatmap = buildHeatmapGrid(
                this.result.heads[h].weights,
                tokens,
                this.activeToken
            );
            headCard.appendChild(heatmap);

            // Summary: max attention weight
            var maxW = 0;
            for (var i = 0; i < tokens.length; i++) {
                for (var j = 0; j < tokens.length; j++) {
                    maxW = Math.max(maxW, this.result.heads[h].weights[i][j]);
                }
            }
            headCard.appendChild(el('span', {
                className: 'tf-head-summary',
                textContent: 'max: ' + maxW.toFixed(2)
            }));

            (function (idx, viz) {
                headCard.addEventListener('click', function (e) {
                    e.stopPropagation();
                    viz.activeHead = idx;
                    viz.updateDiagram();
                });
            })(h, self);

            this.dom.headCards.push(headCard);
            headTrackInner.appendChild(headCard);
        }

        // End spacer
        headTrackInner.appendChild(el('div', { className: 'tf-head-track-spacer' }));

        headTrack.appendChild(headTrackInner);
        headChain.appendChild(headTrack);

        // ── Head Navigation: prev arrow, dots, next arrow ──
        var headNav = el('div', { className: 'tf-head-nav' });

        var prevBtn = el('button', {
            className: 'tf-head-nav-arrow',
            'aria-label': 'Previous head'
        });
        prevBtn.innerHTML = '<i class="fa fa-chevron-left"></i>';
        prevBtn.addEventListener('click', function () {
            if (self.activeHead > 0) {
                self.activeHead--;
                self.updateDiagram();
            }
        });
        headNav.appendChild(prevBtn);

        var dotsWrap = el('div', { className: 'tf-head-nav-dots' });
        this.dom.navDots = [];
        for (var h = 0; h < this.numHeads; h++) {
            var navDot = el('button', {
                className: 'tf-head-nav-dot' + (h === this.activeHead ? ' active' : ''),
                'aria-label': 'Head ' + h
            });
            (function (idx) {
                navDot.addEventListener('click', function () {
                    self.activeHead = idx;
                    self.updateDiagram();
                });
            })(h);
            this.dom.navDots.push(navDot);
            dotsWrap.appendChild(navDot);
        }
        headNav.appendChild(dotsWrap);

        var nextBtn = el('button', {
            className: 'tf-head-nav-arrow',
            'aria-label': 'Next head'
        });
        nextBtn.innerHTML = '<i class="fa fa-chevron-right"></i>';
        nextBtn.addEventListener('click', function () {
            if (self.activeHead < self.numHeads - 1) {
                self.activeHead++;
                self.updateDiagram();
            }
        });
        headNav.appendChild(nextBtn);

        this.dom.prevBtn = prevBtn;
        this.dom.nextBtn = nextBtn;

        headChain.appendChild(headNav);
        c.appendChild(headChain);

        // ── Connector: head carousel → output ──
        var conn3 = el('div', { className: 'tf-connector' }, [
            el('div', { className: 'tf-connector-line' }),
            el('div', { className: 'tf-connector-arrow' })
        ]);
        c.appendChild(conn3);

        // ── Output block (always expanded) ──
        var outputBlock = el('div', {
            id: 'tf-output-block',
            className: 'tf-block tf-block--output tf-block--token-expand expanded',
            'aria-label': 'Attention Output'
        });

        outputBlock.appendChild(el('span', {
            className: 'tf-block-title',
            textContent: 'Attention Output'
        }));

        var outputExpanded = el('div', { className: 'tf-expanded-content', id: 'tf-output-expanded' });

        // Histogram panel (attention distribution for active token)
        var histPanel = el('div', { className: 'tf-hist-panel', id: 'tf-hist-panel' });
        outputExpanded.appendChild(histPanel);

        // Output token cards
        var outRow = el('div', { className: 'tf-token-row' });

        var outLabels = el('div', { className: 'tf-token-labels' });
        outLabels.appendChild(el('span', { className: 'tf-token-label', textContent: 'Pos' }));
        outLabels.appendChild(el('span', { className: 'tf-token-label', textContent: 'Token' }));
        outLabels.appendChild(el('span', { className: 'tf-token-label', textContent: 'Dim' }));
        outRow.appendChild(outLabels);

        this.dom.outputCards = [];
        for (var i = 0; i < tokens.length; i++) {
            var oCard = el('div', { className: 'tf-token-card tf-output-card tf-token-color-' + (i % 10) });
            oCard.style.cursor = 'pointer';
            (function (idx, viz) {
                oCard.addEventListener('click', function (e) {
                    e.stopPropagation();
                    viz.activeToken = idx;
                    viz.updateStepperDisplay();
                    viz.updateDiagram();
                });
            })(i, self);

            oCard.appendChild(el('span', {
                className: 'tf-token-card-time',
                textContent: 't=' + i
            }));

            var outWord = el('span', {
                className: 'tf-token-card-word tf-output-word',
                textContent: tokens[i].word
            });
            oCard.appendChild(outWord);

            var outVecLabel = el('span', {
                className: 'tf-token-card-id tf-output-vector',
                textContent: '<1, ' + this.embedDim + '>'
            });
            oCard.appendChild(outVecLabel);

            this.dom.outputCards.push(oCard);
            outRow.appendChild(oCard);
        }

        outputExpanded.appendChild(outRow);
        outputBlock.appendChild(outputExpanded);
        outputBlock.appendChild(el('span', {
            className: 'tf-block-dim',
            id: 'tf-output-dim',
            textContent: '<' + tokens.length + ', ' + this.embedDim + '>'
        }));
        c.appendChild(outputBlock);

        // Store references
        this.dom.tokensBlock = tokensBlock;
        this.dom.tokensDim = document.getElementById('tf-tokens-dim');
        this.dom.tokensExpanded = tokensExpanded;
        this.dom.embedBlock = embedBlock;
        this.dom.embedTitle = document.getElementById('tf-embed-title');
        this.dom.embedDim = document.getElementById('tf-embed-dim');
        this.dom.embedCollapsed = embedCollapsed;
        this.dom.embedExpanded = embedExpanded;
        this.dom.qkvBlock = qkvBlock;
        this.dom.qkvCollapsed = qkvCollapsed;
        this.dom.qkvExpanded = qkvExpanded;
        this.dom.qkvDim = document.getElementById('tf-qkv-dim');
        this.dom.headChain = headChain;
        this.dom.headTrack = headTrack;
        this.dom.headTrackInner = headTrackInner;
        this.dom.histPanel = histPanel;
        this.dom.outputDim = document.getElementById('tf-output-dim');

        // Click handlers for expandable blocks
        function blockClick(blockName) {
            return function (e) {
                e.stopPropagation();
                self.toggleBlock(blockName);
            };
        }
        embedBlock.addEventListener('click', blockClick('embed'));
        qkvBlock.addEventListener('click', blockClick('qkv'));
    };

    // ─── Expand / Collapse ──────────────────────────────────────────

    TransformerViz.prototype.toggleBlock = function (blockName) {
        this.expandedBlocks[blockName] = !this.expandedBlocks[blockName];
        this.applyBlockStates();
        this.updateDiagram();
    };

    TransformerViz.prototype.applyBlockStates = function () {
        var blocks = [
            { name: 'embed', el: this.dom.embedBlock },
            { name: 'qkv', el: this.dom.qkvBlock }
        ];
        for (var i = 0; i < blocks.length; i++) {
            var b = blocks[i];
            if (b.el) {
                b.el.classList.toggle('expanded', !!this.expandedBlocks[b.name]);
            }
        }
    };

    // ─── Update all dynamic content ─────────────────────────────────

    TransformerViz.prototype.updateDiagram = function () {
        var tokens = EXAMPLES[this.exIdx].tokens;
        var n = tokens.length;

        // Highlight active token across all blocks
        for (var i = 0; i < n; i++) {
            var isActive = (i === this.activeToken);

            if (this.dom.tokenCards && this.dom.tokenCards[i]) {
                this.dom.tokenCards[i].classList.toggle('active', isActive);
            }
            if (this.dom.embedRows && this.dom.embedRows[i]) {
                this.dom.embedRows[i].row.classList.toggle('active', isActive);
            }
            if (this.dom.outputCards && this.dom.outputCards[i]) {
                this.dom.outputCards[i].classList.toggle('active', isActive);
            }
        }

        // Update embedding vector dimension labels
        for (var i = 0; i < this.dom.embedRows.length; i++) {
            this.dom.embedRows[i].vector.textContent = '<1, ' + this.embedDim + '>';
        }

        // Token block dim label
        if (this.dom.tokensDim) {
            this.dom.tokensDim.textContent = '<1, ' + n + '>';
        }

        // Embed block title (PE status)
        if (this.dom.embedTitle) {
            this.dom.embedTitle.textContent = this.usePE ? 'Embedding + PE' : 'Embedding';
        }

        // Embed block dim label
        if (this.dom.embedDim) {
            this.dom.embedDim.textContent = '<' + n + ', ' + this.embedDim + '>';
        }

        // QKV dim label
        if (this.dom.qkvDim) {
            this.dom.qkvDim.innerHTML = '&lt;' + n + ', ' + this.embedDim + '&gt; &rarr; &lt;' + n + ', ' + this.model.headDim + '&gt; &times; 3';
        }

        // Output dim label
        if (this.dom.outputDim) {
            this.dom.outputDim.textContent = '<' + n + ', ' + this.embedDim + '>';
        }

        // Output vector labels
        if (this.dom.outputCards) {
            for (var i = 0; i < this.dom.outputCards.length; i++) {
                var vecEl = this.dom.outputCards[i].querySelector('.tf-output-vector');
                if (vecEl) vecEl.textContent = '<1, ' + this.embedDim + '>';
            }
        }

        // Collapsed content
        this.updateEmbedCollapsed();
        this.updateQKVCollapsed();
        this.updateQKVExpanded();

        // Update head cards: highlight active, update heatmap row/col highlighting
        this.updateHeadCarousel();

        // Update output histogram
        this.updateOutputBar();

        // Update header badges
        this.updateBadges();

        // Update math panel
        this.updateMathPanel();

        // Snap active head to center
        this.snapHeadToCenter();
    };

    // ─── Collapsed content: Embedding ───────────────────────────────

    TransformerViz.prototype.updateEmbedCollapsed = function () {
        var c = this.dom.embedCollapsed;
        if (!c) return;
        c.innerHTML = '';

        var t = this.activeToken;
        var tokens = EXAMPLES[this.exIdx].tokens;
        var colorIdx = t % 10;

        var row = el('div', { className: 'tf-embed-row active' });

        var label = el('span', { className: 'tf-embed-row-label' });
        label.innerHTML = 'x<sub>' + t + '</sub>';
        row.appendChild(label);

        var tokenIdBox = el('span', {
            className: 'tf-token-id-box tf-token-color-' + colorIdx,
            textContent: '[' + tokens[t].id + ']'
        });
        row.appendChild(tokenIdBox);

        var embedVec = el('span', {
            className: 'tf-embed-vector tf-token-color-' + colorIdx,
            textContent: '<1, ' + this.embedDim + '>'
        });
        row.appendChild(embedVec);

        c.appendChild(row);
    };

    // ─── Collapsed content: QKV ─────────────────────────────────────

    TransformerViz.prototype.updateQKVCollapsed = function () {
        var c = this.dom.qkvCollapsed;
        if (!c) return;
        c.innerHTML = '';

        var t = this.activeToken;
        var headDim = this.model.headDim;

        var wrap = el('div', { className: 'tf-qkv-summary' });

        var qLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--q' });
        qLabel.innerHTML = 'Q<sub>' + t + '</sub>';
        wrap.appendChild(qLabel);
        wrap.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));

        var kLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--k' });
        kLabel.innerHTML = 'K<sub>' + t + '</sub>';
        wrap.appendChild(kLabel);
        wrap.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));

        var vLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--v' });
        vLabel.innerHTML = 'V<sub>' + t + '</sub>';
        wrap.appendChild(vLabel);
        wrap.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));

        c.appendChild(wrap);
    };

    // ─── Expanded content: QKV ──────────────────────────────────────

    TransformerViz.prototype.updateQKVExpanded = function () {
        var ex = this.dom.qkvExpanded;
        if (!ex) return;
        ex.innerHTML = '';

        var t = this.activeToken;
        var h = this.activeHead;
        var headData = this.result.heads[h];
        var headDim = this.model.headDim;
        var showN = Math.min(8, headDim);

        // Q row
        var qRow = el('div', { className: 'tf-qkv-row' });
        var qLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--q' });
        qLabel.innerHTML = 'Q<sub>' + t + '</sub> =';
        qRow.appendChild(qLabel);
        qRow.appendChild(buildHeatmapRow(headData.Q[t], showN, 'Query vector'));
        qRow.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));
        ex.appendChild(qRow);

        // K row
        var kRow = el('div', { className: 'tf-qkv-row' });
        var kLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--k' });
        kLabel.innerHTML = 'K<sub>' + t + '</sub> =';
        kRow.appendChild(kLabel);
        kRow.appendChild(buildHeatmapRow(headData.K[t], showN, 'Key vector'));
        kRow.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));
        ex.appendChild(kRow);

        // V row
        var vRow = el('div', { className: 'tf-qkv-row' });
        var vLabel = el('span', { className: 'tf-qkv-label tf-qkv-label--v' });
        vLabel.innerHTML = 'V<sub>' + t + '</sub> =';
        vRow.appendChild(vLabel);
        vRow.appendChild(buildHeatmapRow(headData.V[t], showN, 'Value vector'));
        vRow.appendChild(el('span', {
            className: 'tf-qkv-dim-label',
            textContent: '<1, ' + headDim + '>'
        }));
        ex.appendChild(vRow);
    };

    // ─── Head Carousel Update ───────────────────────────────────────

    TransformerViz.prototype.updateHeadCarousel = function () {
        var cards = this.dom.headCards;
        if (!cards) return;
        var h = this.activeHead;

        for (var hi = 0; hi < cards.length; hi++) {
            cards[hi].classList.toggle('active', hi === h);

            // Update heatmap cell highlighting within each card
            var cells = cards[hi].querySelectorAll('.tf-heatmap-cell');
            for (var ci = 0; ci < cells.length; ci++) {
                var cellRow = parseInt(cells[ci].getAttribute('data-row'));
                var cellCol = parseInt(cells[ci].getAttribute('data-col'));
                cells[ci].classList.toggle('row-active', cellRow === this.activeToken);
                cells[ci].classList.toggle('col-active', cellCol === this.activeToken);
            }

            // Update label highlighting
            var topLabels = cards[hi].querySelectorAll('.tf-heatmap-label-top');
            for (var li = 0; li < topLabels.length; li++) {
                var colIdx = parseInt(topLabels[li].getAttribute('data-col'));
                topLabels[li].classList.toggle('col-active', colIdx === this.activeToken);
            }

            var leftLabels = cards[hi].querySelectorAll('.tf-heatmap-label-left');
            for (var li = 0; li < leftLabels.length; li++) {
                var rowIdx = parseInt(leftLabels[li].getAttribute('data-row'));
                leftLabels[li].classList.toggle('row-active', rowIdx === this.activeToken);
            }
        }

        // Update nav dots
        if (this.dom.navDots) {
            for (var i = 0; i < this.dom.navDots.length; i++) {
                this.dom.navDots[i].classList.toggle('active', i === h);
            }
        }

        // Update arrow disabled states
        if (this.dom.prevBtn) this.dom.prevBtn.disabled = (h === 0);
        if (this.dom.nextBtn) this.dom.nextBtn.disabled = (h === cards.length - 1);
    };

    // ─── Snap active head card to center ────────────────────────────

    TransformerViz.prototype.snapHeadToCenter = function () {
        var track = this.dom.headTrack;
        var cards = this.dom.headCards;
        if (!track || !cards || !cards[this.activeHead]) return;
        var card = cards[this.activeHead];
        var trackRect = track.getBoundingClientRect();
        var cardRect = card.getBoundingClientRect();
        var cardCenterX = cardRect.left + cardRect.width / 2;
        var trackCenterX = trackRect.left + trackRect.width / 2;
        track.scrollLeft += (cardCenterX - trackCenterX);
    };

    // ─── Output Bar Chart ───────────────────────────────────────────

    TransformerViz.prototype.updateOutputBar = function () {
        var panel = this.dom.histPanel;
        if (!panel) panel = document.getElementById('tf-hist-panel');
        if (!panel) return;
        panel.innerHTML = '';

        var t = this.activeToken;
        var h = this.activeHead;
        var tokens = EXAMPLES[this.exIdx].tokens;
        var headData = this.result.heads[h];
        if (!headData) return;

        var colorIdx = t % 10;

        // Header
        var header = el('div', { className: 'tf-hist-header' });
        header.innerHTML = 'Attention weights: Head ' + h + ', query t=' + t + ' (' + displayWord(tokens[t].word) + ')';
        panel.appendChild(header);

        // Vertical bar chart
        var chart = buildAttnBarChart(headData.weights, tokens, t, colorIdx);
        panel.appendChild(chart);
    };

    // ─── Update Header Badges ───────────────────────────────────────

    TransformerViz.prototype.updateBadges = function () {
        var headBadge = document.getElementById('tf-head-badge');
        if (headBadge) headBadge.textContent = this.numHeads + ' Head' + (this.numHeads > 1 ? 's' : '');

        var dimBadge = document.getElementById('tf-dim-badge');
        if (dimBadge) dimBadge.textContent = 'd=' + this.embedDim;
    };

    // ─── Stepper Display ────────────────────────────────────────────

    TransformerViz.prototype.updateStepperDisplay = function () {
        var valEl = document.getElementById('tf-token-step-value');
        if (valEl) valEl.value = this.activeToken;
    };

    // ─── Math Panel ─────────────────────────────────────────────────

    TransformerViz.prototype.updateMathPanel = function () {
        var panel = document.getElementById('tf-math-panel');
        if (!panel || !this.result) return;

        var t = this.activeToken;
        var h = this.activeHead;
        var tokens = EXAMPLES[this.exIdx].tokens;
        var n = tokens.length;
        var headData = this.result.heads[h];
        var headDim = this.model.headDim;
        var showN = Math.min(6, this.embedDim);
        var showH = Math.min(6, headDim);
        var suffix = function (dim, show) { return dim > show ? ', \u2026' : ''; };

        var html = '';

        // Active token info
        html += '<div class="tf-math-section">';
        html += '<div class="tf-math-title">Active Token</div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">Token:</span><span class="tf-math-value">\u201c' + displayWord(tokens[t].word) + '\u201d (id ' + tokens[t].id + ', pos ' + t + ')</span></div>';
        html += '</div>';

        // Embedding
        html += '<div class="tf-math-section">';
        html += '<div class="tf-math-title">Embedding</div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">E[' + tokens[t].id + ']:</span></div>';
        html += '<div class="tf-math-vector">';
        var emb = this.result.embeddings[t];
        for (var i = 0; i < Math.min(showN, emb.length); i++) {
            html += '<span class="tf-vec-cell">' + emb[i].toFixed(3) + '</span>';
        }
        if (this.embedDim > showN) html += '<span class="tf-vec-cell">\u2026</span>';
        html += '</div>';

        // PE
        if (this.usePE) {
            html += '<div class="tf-math-row"><span class="tf-math-label">PE[' + t + ']:</span></div>';
            html += '<div class="tf-math-vector">';
            var peVec = this.result.pe[t];
            for (var i = 0; i < Math.min(showN, peVec.length); i++) {
                html += '<span class="tf-vec-cell">' + peVec[i].toFixed(3) + '</span>';
            }
            if (this.embedDim > showN) html += '<span class="tf-vec-cell">\u2026</span>';
            html += '</div>';
        }
        html += '</div>';

        // Q, K, V for active head
        html += '<div class="tf-math-section">';
        html += '<div class="tf-math-title">Head ' + h + ' Projections (d<sub>k</sub>=' + headDim + ')</div>';

        html += '<div class="tf-math-row"><span class="tf-math-label">Q<sub>' + t + '</sub>:</span><span class="tf-math-value">[' + headData.Q[t].slice(0, showH).map(function (v) { return v.toFixed(3); }).join(', ') + suffix(headDim, showH) + ']</span></div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">K<sub>' + t + '</sub>:</span><span class="tf-math-value">[' + headData.K[t].slice(0, showH).map(function (v) { return v.toFixed(3); }).join(', ') + suffix(headDim, showH) + ']</span></div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">V<sub>' + t + '</sub>:</span><span class="tf-math-value">[' + headData.V[t].slice(0, showH).map(function (v) { return v.toFixed(3); }).join(', ') + suffix(headDim, showH) + ']</span></div>';
        html += '</div>';

        // Scores and weights
        html += '<div class="tf-math-section">';
        html += '<div class="tf-math-title">Attention Scores (query ' + t + ')</div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">\u221Ad<sub>k</sub> = ' + Math.sqrt(headDim).toFixed(2) + ', T = ' + this.temperature.toFixed(1) + '</span></div>';

        for (var j = 0; j < n; j++) {
            var rawScore = headData.scores[t][j];
            var weight = headData.weights[t][j];
            html += '<div class="tf-math-row">';
            html += '<span class="tf-math-label">q<sub>' + t + '</sub>\u00b7k<sub>' + j + '</sub>/\u221Ad<sub>k</sub>/T:</span>';
            html += '<span class="tf-math-value">' + rawScore.toFixed(3) + ' \u2192 softmax: <strong>' + (weight * 100).toFixed(1) + '%</strong></span>';
            html += '</div>';
        }
        html += '</div>';

        // Output
        html += '<div class="tf-math-section">';
        html += '<div class="tf-math-title">Attended Output</div>';
        html += '<div class="tf-math-row"><span class="tf-math-label">out<sub>' + t + '</sub> = \u03A3 w<sub>j</sub>\u00b7V<sub>j</sub>:</span></div>';
        html += '<div class="tf-math-vector">';
        var outVec = headData.output[t];
        for (var i = 0; i < Math.min(showH, outVec.length); i++) {
            html += '<span class="tf-vec-cell">' + outVec[i].toFixed(3) + '</span>';
        }
        if (headDim > showH) html += '<span class="tf-vec-cell">\u2026</span>';
        html += '</div>';
        html += '</div>';

        panel.innerHTML = html;
    };

    // ─── Refresh heatmaps in-place (for temperature/PE changes) ─────

    TransformerViz.prototype.refreshHeatmaps = function () {
        var tokens = EXAMPLES[this.exIdx].tokens;
        var cards = this.dom.headCards;
        if (!cards) return;

        for (var hi = 0; hi < cards.length; hi++) {
            // Remove old heatmap
            var oldHeatmap = cards[hi].querySelector('.tf-heatmap');
            if (oldHeatmap) oldHeatmap.remove();

            // Build new heatmap
            var newHeatmap = buildHeatmapGrid(
                this.result.heads[hi].weights,
                tokens,
                this.activeToken
            );

            // Insert after the title
            var title = cards[hi].querySelector('.tf-head-title');
            if (title && title.nextSibling) {
                cards[hi].insertBefore(newHeatmap, title.nextSibling);
            } else {
                cards[hi].appendChild(newHeatmap);
            }

            // Update summary
            var summary = cards[hi].querySelector('.tf-head-summary');
            if (summary) {
                var maxW = 0;
                for (var i = 0; i < tokens.length; i++) {
                    for (var j = 0; j < tokens.length; j++) {
                        maxW = Math.max(maxW, this.result.heads[hi].weights[i][j]);
                    }
                }
                summary.textContent = 'max: ' + maxW.toFixed(2);
            }
        }
    };

    // ─── Event binding ──────────────────────────────────────────────

    TransformerViz.prototype.bindEvents = function () {
        var self = this;

        // Example selector
        var exSelect = document.getElementById('tf-example-select');
        if (exSelect) {
            exSelect.addEventListener('change', function () {
                self.exIdx = parseInt(this.value);
                self.activeToken = 0;
                self.activeHead = 0;
                self.expandedBlocks = { tokens: true, embed: false, qkv: false, output: true };
                self.rebuildModel();
                self.buildDiagram();
                self.applyBlockStates();
                self.updateDiagram();
            });
        }

        // Token stepper
        DU.wireStepper('tf-token-step-minus', 'tf-token-step-plus', 'tf-token-step-value', {
            min: 0,
            max: 999,
            step: 1,
            onChange: function (val) {
                var maxT = EXAMPLES[self.exIdx].tokens.length - 1;
                var newT = Math.max(0, Math.min(val, maxT));
                self.activeToken = newT;
                var valEl = document.getElementById('tf-token-step-value');
                if (valEl) valEl.value = newT;
                self.updateDiagram();
            }
        });

        // Heads selector
        var headsSelect = document.getElementById('tf-num-heads');
        if (headsSelect) {
            headsSelect.addEventListener('change', function () {
                self.numHeads = parseInt(this.value);
                self.activeHead = 0;
                self.expandedBlocks = { tokens: true, embed: false, qkv: false, output: true };
                self.rebuildModel();
                self.buildDiagram();
                self.applyBlockStates();
                self.updateDiagram();
            });
        }

        // Model dim selector
        var dimSelect = document.getElementById('tf-model-dim');
        if (dimSelect) {
            dimSelect.addEventListener('change', function () {
                self.embedDim = parseInt(this.value);
                self.expandedBlocks = { tokens: true, embed: false, qkv: false, output: true };
                self.rebuildModel();
                self.buildDiagram();
                self.applyBlockStates();
                self.updateDiagram();
            });
        }

        // Temperature slider (frequent — don't rebuild entire diagram)
        var tempSlider = document.getElementById('tf-temperature');
        var tempValue = document.getElementById('tf-temp-value');
        if (tempSlider) {
            tempSlider.addEventListener('input', function () {
                self.temperature = parseFloat(this.value);
                if (tempValue) tempValue.textContent = self.temperature.toFixed(1);
                // Recompute forward pass only
                self.result = runForward(self.model, EXAMPLES[self.exIdx].tokens, self.temperature, self.usePE);
                // Refresh heatmaps in existing cards
                self.refreshHeatmaps();
                // Update output histogram and math panel
                self.updateOutputBar();
                self.updateMathPanel();
                // Update QKV expanded content if visible
                self.updateQKVExpanded();
            });
        }

        // PE toggle
        var peToggle = document.getElementById('tf-use-pe');
        if (peToggle) {
            peToggle.addEventListener('change', function () {
                self.usePE = this.checked;
                // Recompute forward pass
                self.result = runForward(self.model, EXAMPLES[self.exIdx].tokens, self.temperature, self.usePE);
                // Refresh heatmaps in existing cards
                self.refreshHeatmaps();
                // Update related displays
                if (self.dom.embedTitle) {
                    self.dom.embedTitle.textContent = self.usePE ? 'Embedding + PE' : 'Embedding';
                }
                self.updateEmbedCollapsed();
                self.updateQKVCollapsed();
                self.updateQKVExpanded();
                self.updateOutputBar();
                self.updateMathPanel();
            });
        }

        // Reset button
        var resetBtn = document.getElementById('tf-btn-reset');
        if (resetBtn) {
            resetBtn.addEventListener('click', function () {
                self.exIdx = 0;
                self.activeToken = 0;
                self.activeHead = 0;
                self.embedDim = 64;
                self.numHeads = 4;
                self.temperature = 1.0;
                self.usePE = true;
                self.expandedBlocks = { tokens: true, embed: false, qkv: false, output: true };

                // Reset UI controls
                if (exSelect) exSelect.value = '0';
                if (headsSelect) headsSelect.value = '4';
                if (dimSelect) dimSelect.value = '64';
                if (tempSlider) tempSlider.value = '1.0';
                if (tempValue) tempValue.textContent = '1.0';
                if (peToggle) peToggle.checked = true;

                self.rebuildModel();
                self.buildDiagram();
                self.applyBlockStates();
                self.updateDiagram();
            });
        }

        // Theme change: inline heatmap styles need refresh
        if (window.VizLib && window.VizLib.ThemeManager) {
            window.VizLib.ThemeManager.onThemeChange(function () {
                self.refreshHeatmaps();
                self.updateOutputBar();
                self.updateQKVExpanded();
                self.updateMathPanel();
            });
        }
        document.addEventListener('themechange', function () {
            self.refreshHeatmaps();
            self.updateOutputBar();
            self.updateQKVExpanded();
            self.updateMathPanel();
        });

        // Tab switching (btn-group variant)
        var tabBtns = document.querySelectorAll('.info-panel-tabs [data-tab]');
        tabBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                tabBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                var p = btn.closest('.panel');
                p.querySelectorAll('.info-tab-content').forEach(function (c) { c.classList.remove('active'); });
                var tgt = p.querySelector('#tab-' + btn.dataset.tab);
                if (tgt) tgt.classList.add('active');
            });
        });

        // Debounced resize
        var resizeTimer;
        window.addEventListener('resize', function () {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function () {
                self.snapHeadToCenter();
            }, 100);
        });
    };

    // ─── Init ───────────────────────────────────────────────────────

    function init() {
        resolveLibs();
        new TransformerViz();
    }

    if (window.VizLib && window.VizLib._ready) {
        init();
    } else {
        window.addEventListener('vizlib-ready', init);
    }
})();
