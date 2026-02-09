/**
 * Backpropagation Visualizer
 *
 * Interactive step-through of the backpropagation algorithm on a
 * simple 2-3-1 feed-forward network. Animates forward and backward
 * passes, shows gradient flow, loss curve, and a detailed computation log.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const NETWORK_W = 560;
    const NETWORK_H = 350;
    const LOSS_W = 560;
    const LOSS_H = 180;
    const MAX_LOSS_HISTORY = 100;

    // clamp assigned in init() after VizLib is available
    let clamp;

    // ============================================
    // Activation functions and derivatives
    // ============================================
    const Activations = {
        sigmoid: {
            fn: x => 1 / (1 + Math.exp(-clamp(x, -500, 500))),
            dfn: x => {
                const s = 1 / (1 + Math.exp(-clamp(x, -500, 500)));
                return s * (1 - s);
            },
            label: 'sigmoid'
        },
        relu: {
            fn: x => Math.max(0, x),
            dfn: x => x > 0 ? 1 : 0,
            label: 'ReLU'
        },
        tanh: {
            fn: x => Math.tanh(x),
            dfn: x => 1 - Math.tanh(x) ** 2,
            label: 'tanh'
        }
    };

    // ============================================
    // Simple 2-3-1 Network
    // ============================================
    class Network {
        constructor(activationName) {
            this.activation = Activations[activationName] || Activations.sigmoid;
            this.activationName = activationName;

            // Layer sizes: 2 inputs, 3 hidden, 1 output
            this.sizes = [2, 3, 1];

            // Weights: w_hidden[j][i] = weight from input i to hidden j
            // w_output[0][j] = weight from hidden j to output
            this.w_hidden = [];
            this.b_hidden = [];
            for (let j = 0; j < 3; j++) {
                this.w_hidden.push([]);
                for (let i = 0; i < 2; i++) {
                    this.w_hidden[j].push((Math.random() - 0.5) * 1.5);
                }
                this.b_hidden.push((Math.random() - 0.5) * 0.5);
            }

            this.w_output = [];
            this.b_output = 0;
            for (let j = 0; j < 3; j++) {
                this.w_output.push((Math.random() - 0.5) * 1.5);
            }
            this.b_output = (Math.random() - 0.5) * 0.5;

            // Forward pass intermediate values
            this.x = [0, 0];         // inputs
            this.z_hidden = [0, 0, 0]; // pre-activation hidden
            this.a_hidden = [0, 0, 0]; // post-activation hidden
            this.z_output = 0;         // pre-activation output
            this.a_output = 0;         // post-activation output (final prediction)

            // Backward pass gradient values
            this.dL_da_out = 0;  // dL/d(a_output)
            this.da_out_dz_out = 0; // da_output/dz_output
            this.dL_dz_out = 0;  // dL/dz_output (delta_output)

            this.dL_dw_output = [0, 0, 0]; // dL/dw for output weights
            this.dL_db_output = 0;

            this.dL_da_hidden = [0, 0, 0]; // dL/d(a_hidden[j])
            this.da_dz_hidden = [0, 0, 0]; // da/dz for hidden neurons
            this.dL_dz_hidden = [0, 0, 0]; // delta for hidden neurons

            this.dL_dw_hidden = [];  // dL/dw for hidden weights [j][i]
            this.dL_db_hidden = [0, 0, 0];
            for (let j = 0; j < 3; j++) {
                this.dL_dw_hidden.push([0, 0]);
            }

            this.forwardDone = false;
            this.backwardDone = false;
        }

        forward(x1, x2) {
            this.x = [x1, x2];

            // Hidden layer
            for (let j = 0; j < 3; j++) {
                this.z_hidden[j] = this.b_hidden[j];
                for (let i = 0; i < 2; i++) {
                    this.z_hidden[j] += this.w_hidden[j][i] * this.x[i];
                }
                this.a_hidden[j] = this.activation.fn(this.z_hidden[j]);
            }

            // Output layer (also uses the same activation)
            this.z_output = this.b_output;
            for (let j = 0; j < 3; j++) {
                this.z_output += this.w_output[j] * this.a_hidden[j];
            }
            this.a_output = this.activation.fn(this.z_output);

            this.forwardDone = true;
            this.backwardDone = false;
            return this.a_output;
        }

        backward(target) {
            if (!this.forwardDone) return;

            // Loss = 0.5 * (target - a_output)^2
            // dL/da_output = a_output - target
            this.dL_da_out = this.a_output - target;

            // da_output/dz_output = f'(z_output)
            this.da_out_dz_out = this.activation.dfn(this.z_output);

            // dL/dz_output = dL/da_output * da_output/dz_output
            this.dL_dz_out = this.dL_da_out * this.da_out_dz_out;

            // Gradients for output layer weights
            for (let j = 0; j < 3; j++) {
                // dz_output/dw_output[j] = a_hidden[j]
                this.dL_dw_output[j] = this.dL_dz_out * this.a_hidden[j];
            }
            this.dL_db_output = this.dL_dz_out;

            // Propagate to hidden layer
            for (let j = 0; j < 3; j++) {
                // dL/da_hidden[j] = dL/dz_output * dz_output/da_hidden[j]
                //                  = dL/dz_output * w_output[j]
                this.dL_da_hidden[j] = this.dL_dz_out * this.w_output[j];

                // da_hidden[j]/dz_hidden[j] = f'(z_hidden[j])
                this.da_dz_hidden[j] = this.activation.dfn(this.z_hidden[j]);

                // dL/dz_hidden[j] = dL/da_hidden[j] * da_hidden[j]/dz_hidden[j]
                this.dL_dz_hidden[j] = this.dL_da_hidden[j] * this.da_dz_hidden[j];

                // Gradients for hidden layer weights
                for (let i = 0; i < 2; i++) {
                    // dz_hidden[j]/dw_hidden[j][i] = x[i]
                    this.dL_dw_hidden[j][i] = this.dL_dz_hidden[j] * this.x[i];
                }
                this.dL_db_hidden[j] = this.dL_dz_hidden[j];
            }

            this.backwardDone = true;
        }

        applyGradients(lr) {
            if (!this.backwardDone) return;

            // Update output weights
            for (let j = 0; j < 3; j++) {
                this.w_output[j] -= lr * this.dL_dw_output[j];
            }
            this.b_output -= lr * this.dL_db_output;

            // Update hidden weights
            for (let j = 0; j < 3; j++) {
                for (let i = 0; i < 2; i++) {
                    this.w_hidden[j][i] -= lr * this.dL_dw_hidden[j][i];
                }
                this.b_hidden[j] -= lr * this.dL_db_hidden[j];
            }
        }

        getLoss(target) {
            return 0.5 * (target - this.a_output) ** 2;
        }

        getMaxGradMagnitude() {
            let maxG = 0;
            for (let j = 0; j < 3; j++) {
                maxG = Math.max(maxG, Math.abs(this.dL_dw_output[j]));
                for (let i = 0; i < 2; i++) {
                    maxG = Math.max(maxG, Math.abs(this.dL_dw_hidden[j][i]));
                }
            }
            return maxG;
        }
    }

    // ============================================
    // Network Diagram Renderer
    // ============================================
    class NetworkRenderer {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = null;
            this.dpr = 1;
            this.logicalW = NETWORK_W;
            this.logicalH = NETWORK_H;
            this._setup();
        }

        _setup() {
            if (!this.canvas) return;
            const CU = window.VizLib?.CanvasUtils;
            if (CU) {
                const info = CU.setupHiDPICanvas(this.canvas);
                this.ctx = info.ctx;
                this.dpr = info.dpr;
                this.logicalW = info.logicalWidth;
                this.logicalH = info.logicalHeight;
            } else {
                this.ctx = this.canvas.getContext('2d');
                this.dpr = window.devicePixelRatio || 1;
                const rect = this.canvas.getBoundingClientRect();
                this.logicalW = rect.width || NETWORK_W;
                this.logicalH = rect.height || NETWORK_H;
                this.canvas.width = this.logicalW * this.dpr;
                this.canvas.height = this.logicalH * this.dpr;
                this.canvas.style.width = this.logicalW + 'px';
                this.canvas.style.height = this.logicalH + 'px';
            }
        }

        resize() { this._setup(); }

        _resetTransform() {
            if (!this.ctx) return;
            const CU = window.VizLib?.CanvasUtils;
            if (CU) {
                CU.resetCanvasTransform(this.ctx, this.dpr);
            } else {
                this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            }
        }

        _readColors() {
            const s = getComputedStyle(document.documentElement);
            this.inputColor = s.getPropertyValue('--bp-input-node').trim() || '#377eb8';
            this.hiddenColor = s.getPropertyValue('--bp-hidden-node').trim() || '#4daf4a';
            this.outputColor = s.getPropertyValue('--bp-output-node').trim() || '#e41a1c';
            this.edgePositive = s.getPropertyValue('--bp-edge-positive').trim() || 'rgba(55,126,184,0.7)';
            this.edgeNegative = s.getPropertyValue('--bp-edge-negative').trim() || 'rgba(228,26,28,0.7)';
            this.edgeNeutral = s.getPropertyValue('--bp-edge-neutral').trim() || 'rgba(150,150,150,0.3)';
            this.gradientWarm = s.getPropertyValue('--bp-gradient-warm').trim() || '#ff6b35';
            this.gradientCool = s.getPropertyValue('--bp-gradient-cool').trim() || '#4a90d9';
            this.forwardHL = s.getPropertyValue('--bp-forward-highlight').trim() || 'rgba(76,175,80,0.5)';
            this.backwardHL = s.getPropertyValue('--bp-backward-highlight').trim() || 'rgba(255,152,0,0.5)';
            this.textColor = s.getPropertyValue('--viz-text').trim() || '#333333';
            this.nodeText = s.getPropertyValue('--bp-node-text').trim() || '#ffffff';
            this.canvasBg = s.getPropertyValue('--viz-canvas-bg').trim() || '#fafafa';
        }

        /**
         * Render the network diagram.
         * @param {Network} net - the network
         * @param {string} phase - 'idle', 'forward', 'backward'
         * @param {number} animProgress - 0..1 animation progress within the phase
         */
        render(net, phase, animProgress) {
            if (!this.ctx) return;
            this._readColors();
            this._resetTransform();

            const ctx = this.ctx;
            const w = this.logicalW;
            const h = this.logicalH;
            ctx.clearRect(0, 0, w, h);

            // Network layout: 3 columns for input(2), hidden(3), output(1)
            const layerX = [w * 0.15, w * 0.5, w * 0.85];
            const nodeRadius = 24;

            // Node positions
            const inputY = [h * 0.35, h * 0.65];
            const hiddenY = [h * 0.2, h * 0.5, h * 0.8];
            const outputY = [h * 0.5];

            const inputPos = inputY.map(y => ({ x: layerX[0], y }));
            const hiddenPos = hiddenY.map(y => ({ x: layerX[1], y }));
            const outputPos = outputY.map(y => ({ x: layerX[2], y }));

            // Determine which layer is "active" during animation
            // Forward: left-to-right; Backward: right-to-left
            let activeLayer = -1; // -1 = none
            if (phase === 'forward') {
                if (animProgress < 0.33) activeLayer = 0;
                else if (animProgress < 0.66) activeLayer = 1;
                else activeLayer = 2;
            } else if (phase === 'backward') {
                if (animProgress < 0.33) activeLayer = 2;
                else if (animProgress < 0.66) activeLayer = 1;
                else activeLayer = 0;
            }

            // ---- Draw edges ----
            // Input -> Hidden edges
            for (let j = 0; j < 3; j++) {
                for (let i = 0; i < 2; i++) {
                    const weight = net.w_hidden[j][i];
                    this._drawEdge(ctx, inputPos[i], hiddenPos[j], weight, nodeRadius,
                        net, phase, animProgress, 'hidden', j, i);
                }
            }
            // Hidden -> Output edges
            for (let j = 0; j < 3; j++) {
                const weight = net.w_output[j];
                this._drawEdge(ctx, hiddenPos[j], outputPos[0], weight, nodeRadius,
                    net, phase, animProgress, 'output', 0, j);
            }

            // ---- Draw nodes ----
            // Input nodes
            for (let i = 0; i < 2; i++) {
                const val = net.forwardDone ? net.x[i] : null;
                const label = 'x' + (i + 1);
                const isActive = (phase === 'forward' && activeLayer === 0) ||
                                 (phase === 'backward' && activeLayer === 0);
                this._drawNode(ctx, inputPos[i], nodeRadius, this.inputColor,
                    val, label, isActive, phase);
            }

            // Hidden nodes
            for (let j = 0; j < 3; j++) {
                const val = net.forwardDone ? net.a_hidden[j] : null;
                const label = 'h' + (j + 1);
                const isActive = (phase === 'forward' && activeLayer === 1) ||
                                 (phase === 'backward' && activeLayer === 1);
                this._drawNode(ctx, hiddenPos[j], nodeRadius, this.hiddenColor,
                    val, label, isActive, phase);

                // Show gradient below node during backward pass
                if (phase === 'backward' && net.backwardDone) {
                    const grad = net.dL_dz_hidden[j];
                    ctx.font = '10px ' + getComputedStyle(document.documentElement).getPropertyValue('--viz-mono-font').trim().split(',')[0].replace(/'/g, '');
                    ctx.fillStyle = this._gradientColor(Math.abs(grad), net.getMaxGradMagnitude());
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText('\u03B4=' + grad.toFixed(3), hiddenPos[j].x, hiddenPos[j].y + nodeRadius + 6);
                }
            }

            // Output node
            {
                const val = net.forwardDone ? net.a_output : null;
                const label = 'y';
                const isActive = (phase === 'forward' && activeLayer === 2) ||
                                 (phase === 'backward' && activeLayer === 2);
                this._drawNode(ctx, outputPos[0], nodeRadius, this.outputColor,
                    val, label, isActive, phase);

                // Show gradient below node during backward pass
                if (phase === 'backward' && net.backwardDone) {
                    const grad = net.dL_dz_out;
                    ctx.font = '10px sans-serif';
                    ctx.fillStyle = this._gradientColor(Math.abs(grad), net.getMaxGradMagnitude());
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText('\u03B4=' + grad.toFixed(3), outputPos[0].x, outputPos[0].y + nodeRadius + 6);
                }
            }

            // ---- Layer labels ----
            ctx.font = '12px sans-serif';
            ctx.fillStyle = this.textColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Input', layerX[0], h - 20);
            ctx.fillText('Hidden (3)', layerX[1], h - 20);
            ctx.fillText('Output', layerX[2], h - 20);
        }

        _drawEdge(ctx, from, to, weight, nodeRadius, net, phase, animProgress, layerType, toIdx, fromIdx) {
            const magnitude = Math.abs(weight);
            const maxW = 3;

            // Compute start and end, offset by node radius
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const nx = dx / dist;
            const ny = dy / dist;

            const sx = from.x + nx * nodeRadius;
            const sy = from.y + ny * nodeRadius;
            const ex = to.x - nx * nodeRadius;
            const ey = to.y - ny * nodeRadius;

            // Base edge
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(ex, ey);
            ctx.lineWidth = 0.5 + clamp(magnitude / maxW, 0, 1) * 3;

            if (weight > 0.01) {
                ctx.strokeStyle = this.edgePositive;
            } else if (weight < -0.01) {
                ctx.strokeStyle = this.edgeNegative;
            } else {
                ctx.strokeStyle = this.edgeNeutral;
            }
            ctx.globalAlpha = 0.3 + clamp(magnitude / maxW, 0, 1) * 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1;

            // Weight label at midpoint
            const mx = (sx + ex) / 2;
            const my = (sy + ey) / 2;
            ctx.save();
            ctx.font = '9px sans-serif';
            ctx.fillStyle = this.textColor;
            ctx.globalAlpha = 0.7;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            // Offset label slightly perpendicular to edge
            const perpX = -ny * 10;
            const perpY = nx * 10;
            ctx.fillText(weight.toFixed(2), mx + perpX, my + perpY);
            ctx.restore();

            // Gradient flow overlay during backward pass
            if (phase === 'backward' && net.backwardDone) {
                let grad = 0;
                if (layerType === 'output') {
                    grad = net.dL_dw_output[fromIdx];
                } else if (layerType === 'hidden') {
                    grad = net.dL_dw_hidden[toIdx][fromIdx];
                }
                const gradMag = Math.abs(grad);
                const maxG = net.getMaxGradMagnitude();

                if (maxG > 1e-8) {
                    const intensity = clamp(gradMag / maxG, 0, 1);
                    // Draw gradient flow as thick overlay in backward direction (to -> from)
                    ctx.beginPath();
                    ctx.moveTo(ex, ey);
                    ctx.lineTo(sx, sy);
                    ctx.lineWidth = 2 + intensity * 4;
                    ctx.strokeStyle = this._gradientColor(gradMag, maxG);
                    ctx.globalAlpha = 0.3 + intensity * 0.5;
                    ctx.stroke();
                    ctx.globalAlpha = 1;

                    // Small arrowhead on the from-end
                    if (intensity > 0.05) {
                        const arrowLen = 8 + intensity * 6;
                        const arrowAngle = Math.atan2(sy - ey, sx - ex);
                        ctx.beginPath();
                        ctx.moveTo(sx, sy);
                        ctx.lineTo(sx - arrowLen * Math.cos(arrowAngle - 0.4),
                                   sy - arrowLen * Math.sin(arrowAngle - 0.4));
                        ctx.moveTo(sx, sy);
                        ctx.lineTo(sx - arrowLen * Math.cos(arrowAngle + 0.4),
                                   sy - arrowLen * Math.sin(arrowAngle + 0.4));
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = this._gradientColor(gradMag, maxG);
                        ctx.globalAlpha = 0.5 + intensity * 0.5;
                        ctx.stroke();
                        ctx.globalAlpha = 1;
                    }
                }
            }

            // Forward pass animation pulse
            if (phase === 'forward') {
                const edgeLayerIdx = (layerType === 'hidden') ? 0 : 1;
                const normalizedProgress = animProgress;
                const edgeStart = edgeLayerIdx * 0.5;
                const edgeEnd = edgeStart + 0.5;
                if (normalizedProgress >= edgeStart && normalizedProgress <= edgeEnd) {
                    const localT = (normalizedProgress - edgeStart) / 0.5;
                    const pulseX = sx + (ex - sx) * localT;
                    const pulseY = sy + (ey - sy) * localT;
                    ctx.beginPath();
                    ctx.arc(pulseX, pulseY, 4, 0, Math.PI * 2);
                    ctx.fillStyle = this.forwardHL;
                    ctx.fill();
                }
            }
        }

        _drawNode(ctx, pos, radius, color, value, label, isActive, phase) {
            // Glow for active node
            if (isActive) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
                ctx.fillStyle = (phase === 'forward') ? this.forwardHL : this.backwardHL;
                ctx.fill();
            }

            // Node circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Value inside node
            if (value !== null && value !== undefined) {
                ctx.fillStyle = this.nodeText;
                ctx.font = 'bold 12px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(value.toFixed(3), pos.x, pos.y);
            }

            // Label above node
            ctx.fillStyle = this.textColor;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(label, pos.x, pos.y - radius - 4);
        }

        _gradientColor(magnitude, maxMag) {
            if (maxMag < 1e-8) return this.gradientCool;
            const t = clamp(magnitude / maxMag, 0, 1);
            // Interpolate between cool (blue) and warm (orange/red)
            // Parse colors
            const cool = this._hexToRgb(this.gradientCool);
            const warm = this._hexToRgb(this.gradientWarm);
            const r = Math.round(cool[0] * (1 - t) + warm[0] * t);
            const g = Math.round(cool[1] * (1 - t) + warm[1] * t);
            const b = Math.round(cool[2] * (1 - t) + warm[2] * t);
            return `rgb(${r},${g},${b})`;
        }

        _hexToRgb(str) {
            if (str.startsWith('#')) {
                const hex = str.slice(1);
                return [
                    parseInt(hex.substring(0, 2), 16),
                    parseInt(hex.substring(2, 4), 16),
                    parseInt(hex.substring(4, 6), 16)
                ];
            }
            const m = str.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
            if (m) return [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])];
            return [128, 128, 128];
        }
    }

    // ============================================
    // Loss Curve Renderer
    // ============================================
    class LossCurveRenderer {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = null;
            this.dpr = 1;
            this.logicalW = LOSS_W;
            this.logicalH = LOSS_H;
            this._setup();
        }

        _setup() {
            if (!this.canvas) return;
            const CU = window.VizLib?.CanvasUtils;
            if (CU) {
                const info = CU.setupHiDPICanvas(this.canvas);
                this.ctx = info.ctx;
                this.dpr = info.dpr;
                this.logicalW = info.logicalWidth;
                this.logicalH = info.logicalHeight;
            } else {
                this.ctx = this.canvas.getContext('2d');
                this.dpr = window.devicePixelRatio || 1;
                const rect = this.canvas.getBoundingClientRect();
                this.logicalW = rect.width || LOSS_W;
                this.logicalH = rect.height || LOSS_H;
                this.canvas.width = this.logicalW * this.dpr;
                this.canvas.height = this.logicalH * this.dpr;
                this.canvas.style.width = this.logicalW + 'px';
                this.canvas.style.height = this.logicalH + 'px';
            }
        }

        resize() { this._setup(); }

        _resetTransform() {
            if (!this.ctx) return;
            const CU = window.VizLib?.CanvasUtils;
            if (CU) {
                CU.resetCanvasTransform(this.ctx, this.dpr);
            } else {
                this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            }
        }

        render(lossHistory) {
            if (!this.ctx) return;
            this._resetTransform();

            const s = getComputedStyle(document.documentElement);
            const lossLine = s.getPropertyValue('--bp-loss-line').trim() || '#e41a1c';
            const lossFill = s.getPropertyValue('--bp-loss-fill').trim() || 'rgba(228,26,28,0.1)';
            const gridColor = s.getPropertyValue('--bp-loss-grid').trim() || 'rgba(0,0,0,0.08)';
            const textColor = s.getPropertyValue('--viz-text').trim() || '#333';
            const textMuted = s.getPropertyValue('--viz-text-muted').trim() || '#6c757d';

            const ctx = this.ctx;
            const w = this.logicalW;
            const h = this.logicalH;
            const pad = { top: 20, right: 20, bottom: 30, left: 50 };
            const plotW = w - pad.left - pad.right;
            const plotH = h - pad.top - pad.bottom;

            ctx.clearRect(0, 0, w, h);

            // Grid lines
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = pad.top + (plotH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(pad.left + plotW, y);
                ctx.stroke();
            }

            if (lossHistory.length === 0) {
                ctx.fillStyle = textMuted;
                ctx.font = '13px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No data yet. Run a training step.', w / 2, h / 2);
                return;
            }

            // Determine Y scale
            const maxLoss = Math.max(0.01, ...lossHistory);
            const yScale = plotH / maxLoss;
            const xStep = lossHistory.length > 1 ? plotW / (lossHistory.length - 1) : plotW;

            // Draw filled area
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top + plotH);
            for (let i = 0; i < lossHistory.length; i++) {
                const px = pad.left + i * xStep;
                const py = pad.top + plotH - lossHistory[i] * yScale;
                ctx.lineTo(px, py);
            }
            ctx.lineTo(pad.left + (lossHistory.length - 1) * xStep, pad.top + plotH);
            ctx.closePath();
            ctx.fillStyle = lossFill;
            ctx.fill();

            // Draw line
            ctx.beginPath();
            for (let i = 0; i < lossHistory.length; i++) {
                const px = pad.left + i * xStep;
                const py = pad.top + plotH - lossHistory[i] * yScale;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.strokeStyle = lossLine;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Latest point
            if (lossHistory.length > 0) {
                const lastI = lossHistory.length - 1;
                const lx = pad.left + lastI * xStep;
                const ly = pad.top + plotH - lossHistory[lastI] * yScale;
                ctx.beginPath();
                ctx.arc(lx, ly, 4, 0, Math.PI * 2);
                ctx.fillStyle = lossLine;
                ctx.fill();
            }

            // Y-axis labels
            ctx.fillStyle = textMuted;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i <= 4; i++) {
                const val = maxLoss * (1 - i / 4);
                const y = pad.top + (plotH / 4) * i;
                ctx.fillText(val.toFixed(3), pad.left - 6, y);
            }

            // X-axis label
            ctx.fillStyle = textMuted;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Training Step', w / 2, h - 12);

            // Y-axis title
            ctx.save();
            ctx.translate(14, h / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = textMuted;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Loss', 0, 0);
            ctx.restore();
        }
    }

    // ============================================
    // Computation Log Manager
    // ============================================
    class ComputationLog {
        constructor(container) {
            this.container = container;
        }

        clear() {
            if (this.container) this.container.innerHTML = '';
        }

        add(message, type) {
            if (!this.container) return;
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + (type || 'log-info');

            let icon = 'fa-info-circle';
            if (type === 'log-forward') icon = 'fa-arrow-right';
            else if (type === 'log-backward') icon = 'fa-arrow-left';
            else if (type === 'log-update') icon = 'fa-pencil';
            else if (type === 'log-success') icon = 'fa-check';
            else if (type === 'log-warning') icon = 'fa-exclamation-triangle';

            entry.innerHTML =
                '<span class="log-icon"><i class="fa ' + icon + '"></i></span>' +
                '<span class="log-message">' + message + '</span>';
            this.container.appendChild(entry);
            this.container.scrollTop = this.container.scrollHeight;
        }

        addSeparator() {
            if (!this.container) return;
            const hr = document.createElement('hr');
            hr.style.margin = '4px 0';
            hr.style.border = 'none';
            hr.style.borderTop = '1px dashed var(--viz-border)';
            this.container.appendChild(hr);
            this.container.scrollTop = this.container.scrollHeight;
        }
    }

    // ============================================
    // Main Visualization Controller
    // ============================================
    class BackpropViz {
        constructor() {
            this.networkCanvas = document.getElementById('network-canvas');
            this.lossCanvas = document.getElementById('loss-canvas');

            this.networkRenderer = new NetworkRenderer(this.networkCanvas);
            this.lossCurveRenderer = new LossCurveRenderer(this.lossCanvas);
            this.log = new ComputationLog(document.getElementById('computation-log'));

            this.network = null;
            this.step = 0;
            this.lossHistory = [];
            this.phase = 'idle'; // 'idle', 'forward', 'backward'
            this.animProgress = 0;
            this.animId = null;

            this._init();
        }

        _init() {
            this._buildNetwork();
            this._setupEventListeners();
            this._renderAll();

            // Theme change support
            if (window.VizLib?.ThemeManager) {
                window.VizLib.ThemeManager.onThemeChange(() => {
                    this._renderAll();
                });
            }

            // Resize handling
            window.addEventListener('resize', () => {
                this.networkRenderer.resize();
                this.lossCurveRenderer.resize();
                this._renderAll();
            });
        }

        _buildNetwork() {
            const activation = document.getElementById('activation-select')?.value || 'sigmoid';
            this.network = new Network(activation);
            this.step = 0;
            this.lossHistory = [];
            this.phase = 'idle';
            this._updateMetrics();
        }

        _setupEventListeners() {
            // Input sliders
            const bindSlider = (id, displayId, decimals) => {
                const slider = document.getElementById(id);
                const display = document.getElementById(displayId);
                slider?.addEventListener('input', () => {
                    display.textContent = parseFloat(slider.value).toFixed(decimals);
                });
            };
            bindSlider('x1-slider', 'x1-value', 2);
            bindSlider('x2-slider', 'x2-value', 2);
            bindSlider('target-slider', 'target-value', 2);
            bindSlider('lr-slider', 'lr-value', 3);

            // Activation change -> rebuild network
            document.getElementById('activation-select')?.addEventListener('change', () => {
                this._reset();
            });

            // Buttons
            document.getElementById('btn-forward')?.addEventListener('click', () => this._doForwardPass());
            document.getElementById('btn-backward')?.addEventListener('click', () => this._doBackwardPass());
            document.getElementById('btn-train-step')?.addEventListener('click', () => this._doTrainStep());
            document.getElementById('btn-reset')?.addEventListener('click', () => this._reset());
        }

        _getInputs() {
            const x1 = parseFloat(document.getElementById('x1-slider')?.value || '0.5');
            const x2 = parseFloat(document.getElementById('x2-slider')?.value || '0.8');
            return [x1, x2];
        }

        _getTarget() {
            return parseFloat(document.getElementById('target-slider')?.value || '1.0');
        }

        _getLR() {
            return parseFloat(document.getElementById('lr-slider')?.value || '0.5');
        }

        _doForwardPass() {
            if (this.animId) cancelAnimationFrame(this.animId);

            const [x1, x2] = this._getInputs();
            const target = this._getTarget();

            this.network.forward(x1, x2);
            const loss = this.network.getLoss(target);

            // Log forward pass computations
            this.log.addSeparator();
            this.log.add('<strong>--- Forward Pass ---</strong>', 'log-forward');
            this.log.add('Inputs: x1=' + x1.toFixed(2) + ', x2=' + x2.toFixed(2), 'log-forward');

            for (let j = 0; j < 3; j++) {
                const z = this.network.z_hidden[j];
                const a = this.network.a_hidden[j];
                this.log.add(
                    'h' + (j+1) + ': z=' + z.toFixed(4) +
                    ', a=' + this.network.activationName + '(z)=' + a.toFixed(4),
                    'log-forward'
                );
            }

            this.log.add(
                'Output: z=' + this.network.z_output.toFixed(4) +
                ', y=' + this.network.a_output.toFixed(4),
                'log-forward'
            );
            this.log.add('Loss = 0.5*('+target.toFixed(2)+' - '+this.network.a_output.toFixed(4)+')^2 = ' + loss.toFixed(6), 'log-forward');

            // Animate forward pass
            this.phase = 'forward';
            this._setBadge('Forward Pass', 'forward-active');
            this._animatePhase(() => {
                this.phase = 'idle';
                this._setBadge('Forward Done', 'complete');
                this._updateMetrics();
                this._renderAll();
            });
        }

        _doBackwardPass() {
            if (!this.network.forwardDone) {
                this.log.add('Run a forward pass first!', 'log-warning');
                return;
            }
            if (this.animId) cancelAnimationFrame(this.animId);

            const target = this._getTarget();
            this.network.backward(target);

            // Log backward pass computations
            this.log.addSeparator();
            this.log.add('<strong>--- Backward Pass ---</strong>', 'log-backward');
            this.log.add(
                '\u2202L/\u2202y = y - t = ' + this.network.a_output.toFixed(4) +
                ' - ' + target.toFixed(2) + ' = ' + this.network.dL_da_out.toFixed(6),
                'log-backward'
            );
            this.log.add(
                '\u2202y/\u2202z_out = f\'(z_out) = ' + this.network.da_out_dz_out.toFixed(6),
                'log-backward'
            );
            this.log.add(
                '\u03B4_out = \u2202L/\u2202z_out = ' + this.network.dL_dz_out.toFixed(6),
                'log-backward'
            );

            for (let j = 0; j < 3; j++) {
                this.log.add(
                    '\u2202L/\u2202w_out[' + j + '] = \u03B4_out * a_h' + (j+1) +
                    ' = ' + this.network.dL_dw_output[j].toFixed(6),
                    'log-backward'
                );
            }

            for (let j = 0; j < 3; j++) {
                this.log.add(
                    '\u03B4_h' + (j+1) + ' = w_out[' + j + '] * \u03B4_out * f\'(z_h' + (j+1) +
                    ') = ' + this.network.dL_dz_hidden[j].toFixed(6),
                    'log-backward'
                );
                for (let i = 0; i < 2; i++) {
                    this.log.add(
                        '  \u2202L/\u2202w_h' + (j+1) + '[' + i + '] = \u03B4_h' + (j+1) +
                        ' * x' + (i+1) + ' = ' + this.network.dL_dw_hidden[j][i].toFixed(6),
                        'log-backward'
                    );
                }
            }

            this.log.add('Max |gradient| = ' + this.network.getMaxGradMagnitude().toFixed(6), 'log-backward');

            // Animate backward pass
            this.phase = 'backward';
            this._setBadge('Backward Pass', 'backward-active');
            this._animatePhase(() => {
                this.phase = 'backward'; // keep showing gradients
                this._setBadge('Gradients Computed', 'complete');
                this._updateMetrics();
                this._renderAll();
            });
        }

        _doTrainStep() {
            if (this.animId) cancelAnimationFrame(this.animId);

            const [x1, x2] = this._getInputs();
            const target = this._getTarget();
            const lr = this._getLR();

            // Forward
            this.network.forward(x1, x2);
            const lossBefore = this.network.getLoss(target);

            // Backward
            this.network.backward(target);

            // Log
            this.log.addSeparator();
            this.log.add('<strong>=== Train Step ' + (this.step + 1) + ' ===</strong>', 'log-info');
            this.log.add('Forward: y=' + this.network.a_output.toFixed(4) + ', Loss=' + lossBefore.toFixed(6), 'log-forward');
            this.log.add('Max |grad| = ' + this.network.getMaxGradMagnitude().toFixed(6), 'log-backward');

            // Apply gradients
            this.network.applyGradients(lr);
            this.log.add('Weights updated (lr=' + lr.toFixed(3) + ')', 'log-update');

            // Re-run forward to get new loss
            this.network.forward(x1, x2);
            const lossAfter = this.network.getLoss(target);
            this.log.add('New output: y=' + this.network.a_output.toFixed(4) + ', Loss=' + lossAfter.toFixed(6), 'log-update');

            this.step++;
            this.lossHistory.push(lossBefore);
            if (this.lossHistory.length > MAX_LOSS_HISTORY) {
                this.lossHistory.shift();
            }

            // Animate: forward then backward
            this.phase = 'forward';
            this._setBadge('Training...', 'forward-active');
            this._animatePhase(() => {
                this.phase = 'backward';
                this._setBadge('Training...', 'backward-active');
                this._animatePhase(() => {
                    this.phase = 'idle';
                    this._setBadge('Step ' + this.step + ' Done', 'complete');
                    this._updateMetrics();
                    this._renderAll();
                });
            });
        }

        _reset() {
            if (this.animId) cancelAnimationFrame(this.animId);
            this.log.clear();
            this.log.add('Network reset. Press Forward Pass or Train Step to begin.', 'log-info');
            this._buildNetwork();
            this._setBadge('Ready', '');
            this._updateMetrics();
            this._renderAll();
        }

        _animatePhase(onComplete) {
            this.animProgress = 0;
            const duration = 600; // ms
            const startTime = performance.now();

            const animate = (now) => {
                const elapsed = now - startTime;
                this.animProgress = clamp(elapsed / duration, 0, 1);
                this._renderAll();

                if (this.animProgress < 1) {
                    this.animId = requestAnimationFrame(animate);
                } else {
                    this.animId = null;
                    if (onComplete) onComplete();
                }
            };
            this.animId = requestAnimationFrame(animate);
        }

        _renderAll() {
            this.networkRenderer.render(this.network, this.phase, this.animProgress);
            this.lossCurveRenderer.render(this.lossHistory);
        }

        _updateMetrics() {
            const target = this._getTarget();
            this._setMetric('metric-step', String(this.step));
            this._setMetric('metric-target', target.toFixed(2));

            if (this.network.forwardDone) {
                const loss = this.network.getLoss(target);
                this._setMetric('metric-loss', loss.toFixed(6));
                this._setMetric('metric-output', this.network.a_output.toFixed(4));
            } else {
                this._setMetric('metric-loss', '-');
                this._setMetric('metric-output', '-');
            }

            if (this.network.backwardDone) {
                this._setMetric('metric-max-grad', this.network.getMaxGradMagnitude().toFixed(6));
            } else {
                this._setMetric('metric-max-grad', '-');
            }

            this._setMetric('metric-status', this.phase === 'idle' ? 'Ready' : this.phase);

            const stepDisplay = document.getElementById('step-display');
            if (stepDisplay) stepDisplay.textContent = 'Step: ' + this.step;
        }

        _setMetric(id, value) {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        }

        _setBadge(text, className) {
            const badge = document.getElementById('pass-badge');
            if (!badge) return;
            badge.textContent = text;
            badge.className = 'viz-badge';
            if (className) badge.classList.add(className);
        }
    }

    // ============================================
    // Initialize
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new BackpropViz();
    }

    window.addEventListener('vizlib-ready', init);
})();
