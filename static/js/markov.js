/**
 * Markov Chain Visualizer
 * Interactive visualization for stochastic processes and random walks
 */
(function() {
    'use strict';

    // ============================================
    // Preset Markov Chains
    // ============================================
    const PRESETS = {
        weather: {
            name: 'Weather Model',
            description: 'Simple 2-state weather model (Sunny/Rainy)',
            states: [
                { id: 'sunny', name: 'Sunny', x: 250, y: 200 },
                { id: 'rainy', name: 'Rainy', x: 500, y: 200 }
            ],
            transitions: [
                { from: 'sunny', to: 'sunny', prob: 0.8 },
                { from: 'sunny', to: 'rainy', prob: 0.2 },
                { from: 'rainy', to: 'sunny', prob: 0.4 },
                { from: 'rainy', to: 'rainy', prob: 0.6 }
            ]
        },
        pagerank: {
            name: 'PageRank Example',
            description: '4-page web graph for PageRank demonstration',
            states: [
                { id: 'A', name: 'A', x: 200, y: 120 },
                { id: 'B', name: 'B', x: 500, y: 120 },
                { id: 'C', name: 'C', x: 200, y: 300 },
                { id: 'D', name: 'D', x: 500, y: 300 }
            ],
            transitions: [
                { from: 'A', to: 'B', prob: 0.5 },
                { from: 'A', to: 'C', prob: 0.5 },
                { from: 'B', to: 'D', prob: 1.0 },
                { from: 'C', to: 'A', prob: 0.5 },
                { from: 'C', to: 'D', prob: 0.5 },
                { from: 'D', to: 'A', prob: 0.5 },
                { from: 'D', to: 'B', prob: 0.5 }
            ]
        },
        gambler: {
            name: "Gambler's Ruin",
            description: 'Random walk with absorbing states (win/lose)',
            states: [
                { id: 's0', name: '$0', x: 120, y: 200 },
                { id: 's1', name: '$1', x: 280, y: 200 },
                { id: 's2', name: '$2', x: 440, y: 200 },
                { id: 's3', name: '$3', x: 600, y: 200 }
            ],
            transitions: [
                { from: 's0', to: 's0', prob: 1.0 },  // Absorbing (broke)
                { from: 's1', to: 's0', prob: 0.5 },
                { from: 's1', to: 's2', prob: 0.5 },
                { from: 's2', to: 's1', prob: 0.5 },
                { from: 's2', to: 's3', prob: 0.5 },
                { from: 's3', to: 's3', prob: 1.0 }   // Absorbing (won)
            ]
        },
        drunkard: {
            name: 'Drunkard Walk',
            description: 'Symmetric 1D random walk on 5 states',
            states: [
                { id: 's1', name: '1', x: 120, y: 200 },
                { id: 's2', name: '2', x: 240, y: 200 },
                { id: 's3', name: '3', x: 360, y: 200 },
                { id: 's4', name: '4', x: 480, y: 200 },
                { id: 's5', name: '5', x: 600, y: 200 }
            ],
            transitions: [
                { from: 's1', to: 's1', prob: 0.5 },
                { from: 's1', to: 's2', prob: 0.5 },
                { from: 's2', to: 's1', prob: 0.5 },
                { from: 's2', to: 's3', prob: 0.5 },
                { from: 's3', to: 's2', prob: 0.5 },
                { from: 's3', to: 's4', prob: 0.5 },
                { from: 's4', to: 's3', prob: 0.5 },
                { from: 's4', to: 's5', prob: 0.5 },
                { from: 's5', to: 's4', prob: 0.5 },
                { from: 's5', to: 's5', prob: 0.5 }
            ]
        }
    };

    // ============================================
    // Step Types for Playback
    // ============================================
    const STEP_TYPES = {
        INIT: 'INIT',
        TRANSITION: 'TRANSITION',
        UPDATE_DISTRIBUTION: 'UPDATE_DISTRIBUTION',
        CONVERGE_CHECK: 'CONVERGE_CHECK'
    };

    // ============================================
    // Theme Colors
    // ============================================
    function getColors() {
        const style = getComputedStyle(document.documentElement);
        const get = (prop) => style.getPropertyValue(prop).trim() || null;

        return {
            canvasBg: get('--markov-canvas-bg') || '#fafafa',
            nodeBg: get('--markov-node-bg') || '#ffffff',
            nodeBorder: get('--markov-node-border') || '#90a4ae',
            nodeText: get('--markov-node-text') || '#333333',
            current: get('--markov-current') || '#4caf50',
            currentText: get('--markov-current-text') || '#ffffff',
            target: get('--markov-target') || '#ff9800',
            edge: get('--markov-edge') || '#607d8b',
            edgeHighlight: get('--markov-edge-highlight') || '#ff9800',
            edgeText: get('--markov-edge-text') || '#455a64',
            histogram: get('--markov-histogram') || '#2196f3',
            stationary: get('--markov-stationary') || '#9c27b0',
            text: get('--markov-text') || '#333333',
            converged: get('--markov-converged') || '#4caf50',
            notConverged: get('--markov-not-converged') || '#ff9800'
        };
    }

    // ============================================
    // MarkovChainState Class - Data Model
    // ============================================
    class MarkovChainState {
        constructor() {
            this.states = new Map();           // id -> {id, name, x, y}
            this.transitions = new Map();       // fromId -> Map(toId -> probability)
            this.currentState = null;           // Current state id during simulation
            this.visitCounts = new Map();       // id -> number of visits
            this.totalSteps = 0;
        }

        addState(id, name, x, y) {
            this.states.set(id, { id, name, x, y });
            if (!this.transitions.has(id)) {
                this.transitions.set(id, new Map());
            }
            this.visitCounts.set(id, 0);
        }

        setTransition(fromId, toId, probability) {
            if (!this.transitions.has(fromId)) {
                this.transitions.set(fromId, new Map());
            }
            this.transitions.get(fromId).set(toId, probability);
        }

        getTransition(fromId, toId) {
            const fromMap = this.transitions.get(fromId);
            return fromMap ? (fromMap.get(toId) || 0) : 0;
        }

        getOutgoingTransitions(stateId) {
            return this.transitions.get(stateId) || new Map();
        }

        recordVisit(stateId) {
            this.visitCounts.set(stateId, (this.visitCounts.get(stateId) || 0) + 1);
            this.totalSteps++;
        }

        getEmpiricDistribution() {
            const dist = new Map();
            for (const [id] of this.states) {
                const count = this.visitCounts.get(id) || 0;
                dist.set(id, this.totalSteps > 0 ? count / this.totalSteps : 0);
            }
            return dist;
        }

        reset() {
            this.currentState = null;
            for (const [id] of this.states) {
                this.visitCounts.set(id, 0);
            }
            this.totalSteps = 0;
        }

        getTransitionMatrix() {
            const stateIds = Array.from(this.states.keys());
            const n = stateIds.length;
            const matrix = [];

            for (let i = 0; i < n; i++) {
                const row = [];
                for (let j = 0; j < n; j++) {
                    row.push(this.getTransition(stateIds[i], stateIds[j]));
                }
                matrix.push(row);
            }

            return matrix;
        }

        getStateIds() {
            return Array.from(this.states.keys());
        }

        isIrreducible() {
            // Check if all states are reachable from all other states (connected graph)
            const stateIds = this.getStateIds();
            if (stateIds.length === 0) return true;

            for (const startId of stateIds) {
                const visited = new Set();
                const queue = [startId];
                visited.add(startId);

                while (queue.length > 0) {
                    const current = queue.shift();
                    const transitions = this.getOutgoingTransitions(current);
                    for (const [toId, prob] of transitions) {
                        if (prob > 0 && !visited.has(toId)) {
                            visited.add(toId);
                            queue.push(toId);
                        }
                    }
                }

                if (visited.size !== stateIds.length) {
                    return false;
                }
            }

            return true;
        }

        isAperiodic() {
            // A chain is aperiodic if any state has a self-loop
            // (simplified check - proper check requires GCD of cycle lengths)
            for (const [id] of this.states) {
                if (this.getTransition(id, id) > 0) {
                    return true;
                }
            }
            return false;
        }

        isErgodic() {
            return this.isIrreducible() && this.isAperiodic();
        }

        loadPreset(name) {
            const preset = PRESETS[name];
            if (!preset) return;

            this.states.clear();
            this.transitions.clear();
            this.visitCounts.clear();
            this.totalSteps = 0;
            this.currentState = null;

            for (const state of preset.states) {
                this.addState(state.id, state.name, state.x, state.y);
            }

            for (const trans of preset.transitions) {
                this.setTransition(trans.from, trans.to, trans.prob);
            }
        }

        clear() {
            this.states.clear();
            this.transitions.clear();
            this.visitCounts.clear();
            this.totalSteps = 0;
            this.currentState = null;
        }
    }

    // ============================================
    // MarkovSolver Class - Algorithm Logic
    // ============================================
    class MarkovSolver {
        constructor() {
            this.rng = Math.random;
        }

        simulateRandomWalk(chain, startState, numSteps) {
            const steps = [];
            let current = startState;

            // Initialize visit counts
            const visitCounts = new Map();
            for (const [id] of chain.states) {
                visitCounts.set(id, 0);
            }

            // INIT step
            steps.push({
                type: STEP_TYPES.INIT,
                currentState: current,
                visitCounts: new Map(visitCounts),
                message: `Starting random walk from state "${chain.states.get(current).name}"`
            });

            for (let i = 0; i < numSteps; i++) {
                // Record visit to current state
                visitCounts.set(current, (visitCounts.get(current) || 0) + 1);

                // Choose next state
                const transitions = chain.getOutgoingTransitions(current);
                const next = this._sampleTransition(transitions);

                // Create transition step
                steps.push({
                    type: STEP_TYPES.TRANSITION,
                    fromState: current,
                    toState: next,
                    probability: chain.getTransition(current, next),
                    stepNumber: i + 1,
                    currentState: next,
                    visitCounts: new Map(visitCounts),
                    empiricalDist: this._calculateDistribution(visitCounts),
                    message: `Step ${i + 1}: ${chain.states.get(current).name} → ${chain.states.get(next).name} (p=${chain.getTransition(current, next).toFixed(2)})`
                });

                current = next;

                // Periodic convergence check
                if ((i + 1) % 10 === 0 || i === numSteps - 1) {
                    const empirical = this._calculateDistribution(visitCounts);
                    const stationary = this.computeStationaryDistribution(chain);
                    const distance = this._tvDistance(empirical, stationary);

                    steps.push({
                        type: STEP_TYPES.CONVERGE_CHECK,
                        stepNumber: i + 1,
                        currentState: current,
                        empiricalDist: empirical,
                        stationaryDist: stationary,
                        tvDistance: distance,
                        visitCounts: new Map(visitCounts),
                        hasConverged: distance < 0.05,
                        message: `Convergence check: TV distance = ${distance.toFixed(4)}`
                    });
                }
            }

            return { steps, finalVisitCounts: visitCounts };
        }

        _sampleTransition(transitions) {
            const r = this.rng();
            let cumulative = 0;
            for (const [toId, prob] of transitions) {
                cumulative += prob;
                if (r <= cumulative) return toId;
            }
            // Fallback to last state (numerical precision)
            return Array.from(transitions.keys()).pop();
        }

        _calculateDistribution(visitCounts) {
            const total = Array.from(visitCounts.values()).reduce((a, b) => a + b, 0);
            const dist = new Map();
            for (const [id, count] of visitCounts) {
                dist.set(id, total > 0 ? count / total : 0);
            }
            return dist;
        }

        computeStationaryDistribution(chain, tolerance = 1e-8, maxIter = 1000) {
            const matrix = chain.getTransitionMatrix();
            const n = matrix.length;
            const stateIds = chain.getStateIds();

            if (n === 0) return new Map();

            // Initialize uniform distribution
            let dist = new Array(n).fill(1 / n);

            for (let iter = 0; iter < maxIter; iter++) {
                const newDist = new Array(n).fill(0);

                // Matrix-vector multiplication: newDist = dist * P
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        newDist[j] += dist[i] * matrix[i][j];
                    }
                }

                // Check convergence
                let maxDiff = 0;
                for (let i = 0; i < n; i++) {
                    maxDiff = Math.max(maxDiff, Math.abs(newDist[i] - dist[i]));
                }

                dist = newDist;
                if (maxDiff < tolerance) break;
            }

            // Convert to Map
            const result = new Map();
            for (let i = 0; i < n; i++) {
                result.set(stateIds[i], dist[i]);
            }
            return result;
        }

        _tvDistance(dist1, dist2) {
            let sum = 0;
            for (const [id, p1] of dist1) {
                const p2 = dist2.get(id) || 0;
                sum += Math.abs(p1 - p2);
            }
            return sum / 2;
        }

        hasConverged(empirical, stationary, threshold = 0.05) {
            return this._tvDistance(empirical, stationary) < threshold;
        }
    }

    // ============================================
    // UIRenderer Class - Canvas Drawing
    // ============================================
    class UIRenderer {
        constructor(canvasId, histogramCanvasId) {
            this.canvas = document.getElementById(canvasId);
            this.ctx = this.canvas.getContext('2d');

            this.histCanvas = document.getElementById(histogramCanvasId);
            this.histCtx = this.histCanvas ? this.histCanvas.getContext('2d') : null;

            this.dpr = window.devicePixelRatio || 1;
            this.nodeRadius = 35;

            this.colors = getColors();
            this.setupCanvas();
            if (this.histCanvas) {
                this.setupHistogramCanvas();
            }
        }

        setupCanvas() {
            const displayWidth = this.canvas.width;
            const displayHeight = this.canvas.height;

            this.canvas.width = displayWidth * this.dpr;
            this.canvas.height = displayHeight * this.dpr;
            this.canvas.style.width = displayWidth + 'px';
            this.canvas.style.height = displayHeight + 'px';

            this.displayWidth = displayWidth;
            this.displayHeight = displayHeight;

            this.ctx.scale(this.dpr, this.dpr);
        }

        setupHistogramCanvas() {
            const rect = this.histCanvas.parentElement.getBoundingClientRect();
            const displayWidth = rect.width || 700;
            const displayHeight = 150;

            this.histCanvas.width = displayWidth * this.dpr;
            this.histCanvas.height = displayHeight * this.dpr;
            this.histCanvas.style.width = displayWidth + 'px';
            this.histCanvas.style.height = displayHeight + 'px';

            this.histDisplayWidth = displayWidth;
            this.histDisplayHeight = displayHeight;

            this.histCtx.scale(this.dpr, this.dpr);
        }

        updateColors() {
            this.colors = getColors();
        }

        render(chain, step = null) {
            const ctx = this.ctx;

            // Clear canvas
            ctx.save();
            ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            ctx.fillStyle = this.colors.canvasBg;
            ctx.fillRect(0, 0, this.displayWidth, this.displayHeight);

            if (chain.states.size === 0) {
                ctx.restore();
                return;
            }

            // Draw edges first (behind nodes)
            this._drawEdges(chain, step);

            // Draw nodes
            this._drawNodes(chain, step);

            ctx.restore();
        }

        _drawNodes(chain, step) {
            const ctx = this.ctx;

            for (const [id, state] of chain.states) {
                const { x, y, name } = state;
                const isCurrent = step && step.currentState === id;
                const isTarget = step && step.type === STEP_TYPES.TRANSITION && step.toState === id && step.fromState !== id;

                // Node circle
                ctx.beginPath();
                ctx.arc(x, y, this.nodeRadius, 0, Math.PI * 2);

                if (isCurrent) {
                    ctx.fillStyle = this.colors.current;
                    ctx.strokeStyle = this.colors.current;
                } else if (isTarget) {
                    ctx.fillStyle = this.colors.target;
                    ctx.strokeStyle = this.colors.target;
                } else {
                    ctx.fillStyle = this.colors.nodeBg;
                    ctx.strokeStyle = this.colors.nodeBorder;
                }

                ctx.lineWidth = 3;
                ctx.fill();
                ctx.stroke();

                // State name
                ctx.fillStyle = isCurrent ? this.colors.currentText : this.colors.nodeText;
                ctx.font = 'bold 16px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(name, x, y);

                // Visit count (if available)
                if (step && step.visitCounts) {
                    const visits = step.visitCounts.get(id) || 0;
                    ctx.font = '11px sans-serif';
                    ctx.fillStyle = this.colors.nodeText;
                    ctx.fillText(`(${visits})`, x, y + this.nodeRadius + 15);
                }
            }
        }

        _drawEdges(chain, step) {
            const ctx = this.ctx;
            const drawnBidirectional = new Set();

            for (const [fromId, transitions] of chain.transitions) {
                for (const [toId, prob] of transitions) {
                    if (prob <= 0) continue;

                    const from = chain.states.get(fromId);
                    const to = chain.states.get(toId);
                    const isHighlighted = step && step.type === STEP_TYPES.TRANSITION &&
                                          step.fromState === fromId && step.toState === toId;

                    if (fromId === toId) {
                        this._drawSelfLoop(from, prob, isHighlighted);
                    } else {
                        // Check for bidirectional edge
                        const reverseProb = chain.getTransition(toId, fromId);
                        const pairKey = [fromId, toId].sort().join('-');

                        if (reverseProb > 0 && !drawnBidirectional.has(pairKey)) {
                            drawnBidirectional.add(pairKey);
                            const isReverseHighlighted = step && step.type === STEP_TYPES.TRANSITION &&
                                                         step.fromState === toId && step.toState === fromId;
                            this._drawBidirectionalEdge(from, to, prob, reverseProb, isHighlighted, isReverseHighlighted);
                        } else if (!drawnBidirectional.has(pairKey)) {
                            this._drawDirectedEdge(from, to, prob, isHighlighted);
                        }
                    }
                }
            }
        }

        _drawDirectedEdge(from, to, prob, isHighlighted) {
            const ctx = this.ctx;
            const angle = Math.atan2(to.y - from.y, to.x - from.x);

            const startX = from.x + this.nodeRadius * Math.cos(angle);
            const startY = from.y + this.nodeRadius * Math.sin(angle);
            const endX = to.x - (this.nodeRadius + 8) * Math.cos(angle);
            const endY = to.y - (this.nodeRadius + 8) * Math.sin(angle);

            // Draw line
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = isHighlighted ? this.colors.edgeHighlight : this.colors.edge;
            ctx.lineWidth = isHighlighted ? 4 : 2;
            ctx.stroke();

            // Arrow head
            this._drawArrowHead(endX, endY, angle, isHighlighted);

            // Probability label
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2;
            this._drawProbabilityLabel(midX, midY - 10, prob, isHighlighted);
        }

        _drawBidirectionalEdge(from, to, prob1, prob2, isHighlighted1, isHighlighted2) {
            const ctx = this.ctx;
            const angle = Math.atan2(to.y - from.y, to.x - from.x);
            const perpAngle = angle + Math.PI / 2;
            const offset = 12;

            // First direction (from -> to) - offset up
            const startX1 = from.x + this.nodeRadius * Math.cos(angle) + offset * Math.cos(perpAngle);
            const startY1 = from.y + this.nodeRadius * Math.sin(angle) + offset * Math.sin(perpAngle);
            const endX1 = to.x - (this.nodeRadius + 8) * Math.cos(angle) + offset * Math.cos(perpAngle);
            const endY1 = to.y - (this.nodeRadius + 8) * Math.sin(angle) + offset * Math.sin(perpAngle);

            ctx.beginPath();
            ctx.moveTo(startX1, startY1);
            ctx.lineTo(endX1, endY1);
            ctx.strokeStyle = isHighlighted1 ? this.colors.edgeHighlight : this.colors.edge;
            ctx.lineWidth = isHighlighted1 ? 4 : 2;
            ctx.stroke();
            this._drawArrowHead(endX1, endY1, angle, isHighlighted1);

            // Label for first direction
            const midX1 = (startX1 + endX1) / 2;
            const midY1 = (startY1 + endY1) / 2;
            this._drawProbabilityLabel(midX1, midY1 - 8, prob1, isHighlighted1);

            // Second direction (to -> from) - offset down
            const startX2 = to.x + this.nodeRadius * Math.cos(angle + Math.PI) - offset * Math.cos(perpAngle);
            const startY2 = to.y + this.nodeRadius * Math.sin(angle + Math.PI) - offset * Math.sin(perpAngle);
            const endX2 = from.x - (this.nodeRadius + 8) * Math.cos(angle + Math.PI) - offset * Math.cos(perpAngle);
            const endY2 = from.y - (this.nodeRadius + 8) * Math.sin(angle + Math.PI) - offset * Math.sin(perpAngle);

            ctx.beginPath();
            ctx.moveTo(startX2, startY2);
            ctx.lineTo(endX2, endY2);
            ctx.strokeStyle = isHighlighted2 ? this.colors.edgeHighlight : this.colors.edge;
            ctx.lineWidth = isHighlighted2 ? 4 : 2;
            ctx.stroke();
            this._drawArrowHead(endX2, endY2, angle + Math.PI, isHighlighted2);

            // Label for second direction
            const midX2 = (startX2 + endX2) / 2;
            const midY2 = (startY2 + endY2) / 2;
            this._drawProbabilityLabel(midX2, midY2 + 12, prob2, isHighlighted2);
        }

        _drawSelfLoop(state, prob, isHighlighted) {
            const ctx = this.ctx;
            const loopRadius = 22;
            const centerX = state.x;
            const centerY = state.y - this.nodeRadius - loopRadius + 5;

            ctx.beginPath();
            ctx.arc(centerX, centerY, loopRadius, 0.4, Math.PI * 2 - 0.4);
            ctx.strokeStyle = isHighlighted ? this.colors.edgeHighlight : this.colors.edge;
            ctx.lineWidth = isHighlighted ? 4 : 2;
            ctx.stroke();

            // Arrow head at end of loop
            const arrowAngle = Math.PI * 2 - 0.4;
            const arrowX = centerX + loopRadius * Math.cos(arrowAngle);
            const arrowY = centerY + loopRadius * Math.sin(arrowAngle);
            this._drawArrowHead(arrowX, arrowY, arrowAngle + Math.PI / 2, isHighlighted);

            // Probability label
            this._drawProbabilityLabel(centerX, centerY - loopRadius - 8, prob, isHighlighted);
        }

        _drawArrowHead(x, y, angle, isHighlighted) {
            const ctx = this.ctx;
            const size = 10;

            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - size * Math.cos(angle - Math.PI / 6), y - size * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(x - size * Math.cos(angle + Math.PI / 6), y - size * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fillStyle = isHighlighted ? this.colors.edgeHighlight : this.colors.edge;
            ctx.fill();
        }

        _drawProbabilityLabel(x, y, prob, isHighlighted) {
            const ctx = this.ctx;
            const text = prob.toFixed(2);

            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Background
            const metrics = ctx.measureText(text);
            const padding = 3;
            ctx.fillStyle = this.colors.canvasBg;
            ctx.fillRect(x - metrics.width / 2 - padding, y - 7, metrics.width + padding * 2, 14);

            // Text
            ctx.fillStyle = isHighlighted ? this.colors.edgeHighlight : this.colors.edgeText;
            ctx.fillText(text, x, y);
        }

        drawHistogram(chain, empiricalDist, stationaryDist) {
            if (!this.histCtx) return;

            const ctx = this.histCtx;
            const width = this.histDisplayWidth;
            const height = this.histDisplayHeight;

            // Clear
            ctx.save();
            ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            ctx.fillStyle = this.colors.canvasBg;
            ctx.fillRect(0, 0, width, height);

            const stateIds = chain.getStateIds();
            const n = stateIds.length;

            if (n === 0) {
                ctx.restore();
                return;
            }

            const padding = { left: 40, right: 20, top: 20, bottom: 35 };
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            const groupWidth = chartWidth / n;
            const barWidth = Math.min(25, groupWidth * 0.35);
            const barGap = 4;

            // Draw bars
            stateIds.forEach((id, i) => {
                const groupX = padding.left + i * groupWidth + groupWidth / 2;
                const empProb = empiricalDist ? (empiricalDist.get(id) || 0) : 0;
                const statProb = stationaryDist ? (stationaryDist.get(id) || 0) : 0;

                // Empirical bar
                const empHeight = empProb * chartHeight;
                ctx.fillStyle = this.colors.histogram;
                ctx.fillRect(groupX - barWidth - barGap / 2, padding.top + chartHeight - empHeight, barWidth, empHeight);

                // Stationary bar
                const statHeight = statProb * chartHeight;
                ctx.fillStyle = this.colors.stationary;
                ctx.fillRect(groupX + barGap / 2, padding.top + chartHeight - statHeight, barWidth, statHeight);

                // State label
                const state = chain.states.get(id);
                ctx.fillStyle = this.colors.text;
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(state ? state.name : id, groupX, height - padding.bottom + 15);
            });

            // Y-axis labels
            ctx.fillStyle = this.colors.text;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('1.0', padding.left - 5, padding.top + 4);
            ctx.fillText('0.5', padding.left - 5, padding.top + chartHeight / 2 + 4);
            ctx.fillText('0.0', padding.left - 5, padding.top + chartHeight + 4);

            // Y-axis line
            ctx.strokeStyle = this.colors.text;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.3;
            ctx.beginPath();
            ctx.moveTo(padding.left, padding.top);
            ctx.lineTo(padding.left, padding.top + chartHeight);
            ctx.stroke();

            // Horizontal grid lines
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(padding.left, padding.top + chartHeight / 2);
            ctx.lineTo(width - padding.right, padding.top + chartHeight / 2);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;

            ctx.restore();
        }

        updateTransitionMatrix(container, chain) {
            if (!container) return;

            const matrix = chain.getTransitionMatrix();
            const stateIds = chain.getStateIds();

            if (stateIds.length === 0) {
                container.innerHTML = '<p class="text-muted">No states in chain.</p>';
                return;
            }

            const names = stateIds.map(id => {
                const state = chain.states.get(id);
                return state ? state.name : id;
            });

            let html = '<table class="transition-matrix">';
            html += '<tr><th></th>' + names.map(n => `<th>${n}</th>`).join('') + '</tr>';

            for (let i = 0; i < matrix.length; i++) {
                html += `<tr><th>${names[i]}</th>`;
                for (let j = 0; j < matrix[i].length; j++) {
                    const prob = matrix[i][j];
                    let className = prob > 0 ? 'nonzero' : 'zero';
                    if (i === j && prob > 0) className += ' self-loop';
                    html += `<td class="${className}">${prob.toFixed(2)}</td>`;
                }
                html += '</tr>';
            }
            html += '</table>';

            container.innerHTML = html;
        }

        updateErgodicStatus(container, chain) {
            if (!container) return;

            const isIrreducible = chain.isIrreducible();
            const isAperiodic = chain.isAperiodic();
            const isErgodic = isIrreducible && isAperiodic;

            container.innerHTML = `
                <span class="${isIrreducible ? 'check' : 'x'}">
                    <i class="fa fa-${isIrreducible ? 'check' : 'times'}"></i> Irreducible
                </span>
                <span class="${isAperiodic ? 'check' : 'x'}">
                    <i class="fa fa-${isAperiodic ? 'check' : 'times'}"></i> Aperiodic
                </span>
                <span class="${isErgodic ? 'check' : 'x'}">
                    <i class="fa fa-${isErgodic ? 'check' : 'times'}"></i> Ergodic
                </span>
            `;
        }

        updateStepDisplay(el, current, total) {
            if (el) {
                el.textContent = `Step: ${current} / ${total}`;
            }
        }

        updateConvergenceIndicator(el, tvDistance, hasConverged) {
            if (el) {
                if (tvDistance !== null) {
                    el.textContent = hasConverged ?
                        `Converged (TV: ${tvDistance.toFixed(4)})` :
                        `TV Distance: ${tvDistance.toFixed(4)}`;
                    el.className = 'convergence-indicator ' + (hasConverged ? 'converged' : 'not-converged');
                } else {
                    el.textContent = '';
                    el.className = 'convergence-indicator';
                }
            }
        }
    }

    // ============================================
    // MarkovApp Class - Main Orchestrator
    // ============================================
    class MarkovApp {
        constructor() {
            this.chain = new MarkovChainState();
            this.solver = new MarkovSolver();
            this.ui = new UIRenderer('markov-canvas', 'histogram-canvas');

            this.playback = null;
            this.simulationResult = null;
            this.numSteps = 50;

            // DOM elements
            this.stepDisplayEl = document.getElementById('step-display');
            this.convergenceEl = document.getElementById('convergence-indicator');
            this.statusEl = document.getElementById('playback-status');
            this.matrixContainer = document.getElementById('matrix-container');
            this.ergodicContainer = document.getElementById('ergodic-status');
            this.stateCountEl = document.getElementById('markov-state-count');

            this._initPlaybackController();
            this._initEventListeners();
            this.loadPreset('weather');
        }

        _initPlaybackController() {
            if (typeof window.VizLib !== 'undefined' && window.VizLib.PlaybackController) {
                this.playback = new window.VizLib.PlaybackController({
                    initialSpeed: 5,
                    onRenderStep: (step, index) => this._renderStep(step, index),
                    onPlayStateChange: (isPlaying) => this._updatePlayButton(isPlaying),
                    onStepChange: (index, total) => this._onStepChange(index, total),
                    onFinished: () => this._onSimulationFinished(),
                    onReset: () => this._onReset()
                });
            }
        }

        _initEventListeners() {
            // Preset selector
            const presetSelect = document.getElementById('preset-select');
            if (presetSelect) {
                presetSelect.addEventListener('change', (e) => {
                    this.loadPreset(e.target.value);
                });
            }

            // Playback controls
            document.getElementById('btn-play')?.addEventListener('click', () => this.startSimulation());
            document.getElementById('btn-pause')?.addEventListener('click', () => {
                if (this.playback) this.playback.toggle();
            });
            document.getElementById('btn-step-forward')?.addEventListener('click', () => {
                if (this.playback) {
                    this.playback.pause();
                    this.playback.stepForward();
                }
            });
            document.getElementById('btn-step-back')?.addEventListener('click', () => {
                if (this.playback) {
                    this.playback.pause();
                    this.playback.stepBackward();
                }
            });
            document.getElementById('btn-reset')?.addEventListener('click', () => this.reset());

            // Speed slider
            document.getElementById('speed-slider')?.addEventListener('input', (e) => {
                if (this.playback) this.playback.setSpeed(parseInt(e.target.value));
            });

            // Step count input
            const numStepsInput = document.getElementById('num-steps');
            if (numStepsInput) {
                numStepsInput.addEventListener('change', (e) => {
                    this.numSteps = Math.max(10, Math.min(500, parseInt(e.target.value) || 50));
                    e.target.value = this.numSteps;
                });
            }

            // Theme change listener
            document.addEventListener('themechange', () => {
                this.ui.updateColors();
                this.ui.render(this.chain);
                const stationary = this.solver.computeStationaryDistribution(this.chain);
                this.ui.drawHistogram(this.chain, this.chain.getEmpiricDistribution(), stationary);
            });

            // Handle window resize for histogram
            window.addEventListener('resize', () => {
                if (this.ui.histCanvas) {
                    this.ui.setupHistogramCanvas();
                    const stationary = this.solver.computeStationaryDistribution(this.chain);
                    this.ui.drawHistogram(this.chain, this.chain.getEmpiricDistribution(), stationary);
                }
            });
        }

        loadPreset(name) {
            this.reset();
            this.chain.loadPreset(name);

            // Update UI
            this.ui.render(this.chain);
            this.ui.updateTransitionMatrix(this.matrixContainer, this.chain);
            this.ui.updateErgodicStatus(this.ergodicContainer, this.chain);

            // Update state count
            if (this.stateCountEl) {
                this.stateCountEl.textContent = `${this.chain.states.size} states`;
            }

            // Draw initial histogram
            const stationary = this.solver.computeStationaryDistribution(this.chain);
            this.ui.drawHistogram(this.chain, new Map(), stationary);

            // Update status
            if (this.statusEl) {
                this.statusEl.innerHTML = `Loaded <strong>${PRESETS[name]?.name || name}</strong>. Click <strong>Play</strong> to start the random walk.`;
            }

            // Reset convergence indicator
            this.ui.updateConvergenceIndicator(this.convergenceEl, null, false);
        }

        startSimulation() {
            if (this.chain.states.size === 0) {
                if (this.statusEl) this.statusEl.textContent = 'No states in chain. Select a preset first.';
                return;
            }

            // Get starting state (first state)
            const startState = this.chain.getStateIds()[0];

            // Run simulation
            this.simulationResult = this.solver.simulateRandomWalk(this.chain, startState, this.numSteps);

            if (this.playback) {
                this.playback.load(this.simulationResult.steps);

                // Enable playback controls
                this._setPlaybackButtonsEnabled(true);

                // Start playback
                this.playback.play();
            }
        }

        _renderStep(step, index) {
            // Update canvas
            this.ui.render(this.chain, step);

            // Update histogram
            if (step.empiricalDist || step.visitCounts) {
                const empirical = step.empiricalDist || this.solver._calculateDistribution(step.visitCounts);
                const stationary = this.solver.computeStationaryDistribution(this.chain);
                this.ui.drawHistogram(this.chain, empirical, stationary);
            }

            // Update convergence indicator
            if (step.type === STEP_TYPES.CONVERGE_CHECK) {
                this.ui.updateConvergenceIndicator(this.convergenceEl, step.tvDistance, step.hasConverged);
            }

            // Update status message
            if (this.statusEl) {
                this.statusEl.textContent = step.message;
            }
        }

        _onStepChange(index, total) {
            this.ui.updateStepDisplay(this.stepDisplayEl, index + 1, total);
        }

        _updatePlayButton(isPlaying) {
            const pauseBtn = document.getElementById('btn-pause');
            const playBtn = document.getElementById('btn-play');

            if (pauseBtn) {
                pauseBtn.innerHTML = isPlaying ? '<i class="fa fa-pause"></i>' : '<i class="fa fa-play"></i>';
            }
            if (playBtn) {
                playBtn.disabled = isPlaying;
            }
        }

        _onSimulationFinished() {
            if (this.statusEl) {
                this.statusEl.innerHTML = '<strong>Simulation complete!</strong> Use step controls to review or click Reset.';
            }
            this._updatePlayButton(false);
        }

        _onReset() {
            this.chain.reset();
            this.ui.render(this.chain);

            const stationary = this.solver.computeStationaryDistribution(this.chain);
            this.ui.drawHistogram(this.chain, new Map(), stationary);
        }

        reset() {
            if (this.playback) {
                this.playback.reset();
            }

            this.chain.reset();
            this.ui.render(this.chain);

            const stationary = this.solver.computeStationaryDistribution(this.chain);
            this.ui.drawHistogram(this.chain, new Map(), stationary);

            // Reset UI elements
            this._setPlaybackButtonsEnabled(false);
            this.ui.updateStepDisplay(this.stepDisplayEl, 0, 0);
            this.ui.updateConvergenceIndicator(this.convergenceEl, null, false);

            if (this.statusEl) {
                this.statusEl.innerHTML = 'Click <strong>Play</strong> to start the random walk simulation.';
            }
        }

        _setPlaybackButtonsEnabled(enabled) {
            const btnPause = document.getElementById('btn-pause');
            const btnStepForward = document.getElementById('btn-step-forward');
            const btnStepBack = document.getElementById('btn-step-back');

            if (btnPause) btnPause.disabled = !enabled;
            if (btnStepForward) btnStepForward.disabled = !enabled;
            if (btnStepBack) btnStepBack.disabled = !enabled;
        }
    }

    // ============================================
    // Initialize on DOM ready
    // ============================================
    document.addEventListener('DOMContentLoaded', () => {
        window.markovApp = new MarkovApp();
    });
})();
