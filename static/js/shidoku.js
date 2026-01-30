/**
 * Shidoku CSP Solver with Visualization
 *
 * Implements AC-3, Backtracking with MRV, and MAC algorithms
 * for solving 4x4 Shidoku puzzles with step-by-step visualization.
 */

(function() {
    'use strict';

    // Constants
    const N = 4;
    const DIGITS = [1, 2, 3, 4];
    const BOX = 2;

    // Cell naming: A1-D4 (row letter + column number)
    const ROWS = ['A', 'B', 'C', 'D'];
    const COLS = ['1', '2', '3', '4'];

    // All cell names
    const CELLS = [];
    for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
            CELLS.push(ROWS[r] + COLS[c]);
        }
    }

    // Precompute peers and arcs
    const PEERS = {};
    const UNITS = {};

    // Initialize peers for each cell
    CELLS.forEach(cell => {
        PEERS[cell] = new Set();
        UNITS[cell] = [];
    });

    // Build units (rows, columns, boxes)
    // Row units
    for (let r = 0; r < N; r++) {
        const unit = [];
        for (let c = 0; c < N; c++) {
            unit.push(ROWS[r] + COLS[c]);
        }
        unit.forEach(cell => UNITS[cell].push(unit));
    }

    // Column units
    for (let c = 0; c < N; c++) {
        const unit = [];
        for (let r = 0; r < N; r++) {
            unit.push(ROWS[r] + COLS[c]);
        }
        unit.forEach(cell => UNITS[cell].push(unit));
    }

    // Box units (2x2)
    for (let br = 0; br < N; br += BOX) {
        for (let bc = 0; bc < N; bc += BOX) {
            const unit = [];
            for (let r = br; r < br + BOX; r++) {
                for (let c = bc; c < bc + BOX; c++) {
                    unit.push(ROWS[r] + COLS[c]);
                }
            }
            unit.forEach(cell => UNITS[cell].push(unit));
        }
    }

    // Build peers from units
    CELLS.forEach(cell => {
        UNITS[cell].forEach(unit => {
            unit.forEach(peer => {
                if (peer !== cell) {
                    PEERS[cell].add(peer);
                }
            });
        });
    });

    // Build all arcs for AC-3
    const ALL_ARCS = [];
    CELLS.forEach(cell => {
        PEERS[cell].forEach(peer => {
            ALL_ARCS.push([cell, peer]);
        });
    });

    // Step types for visualization
    const StepType = {
        INIT: 'INIT',
        AC3_START: 'AC3_START',
        AC3_CHECK: 'AC3_CHECK',
        AC3_PRUNE: 'AC3_PRUNE',
        AC3_DONE: 'AC3_DONE',
        FIXED_PRUNE: 'FIXED_PRUNE',  // Basic propagation from fixed clues (not AC-3)
        ASSIGN: 'ASSIGN',
        TRY_VALUE: 'TRY_VALUE',
        BACKTRACK: 'BACKTRACK',
        MAC_CHECK: 'MAC_CHECK',
        SOLVED: 'SOLVED',
        UNSOLVABLE: 'UNSOLVABLE'
    };

    // Sample puzzles
    const SAMPLE_PUZZLES = {
        easy: '12..34........12',    // 8 clues
        medium: '1...3.......2..4',  // 6 clues
        hard: '1.........3.....',    // 4 clues
        empty: '................'    // 0 clues
    };

    // Algorithm configurations for comparison
    const ALGORITHM_CONFIGS = [
        { name: 'AC-3 Only', ac3Only: true },
        { name: 'Plain Backtracking', ac3: false, mrv: false, fc: false, mac: false },
        { name: 'Backtracking + FC', ac3: false, mrv: false, fc: true, mac: false },
        { name: 'Backtracking + MRV', ac3: false, mrv: true, fc: false, mac: false },
        { name: 'Backtracking + FC + MRV', ac3: false, mrv: true, fc: true, mac: false },
        { name: 'AC-3 + Backtracking', ac3: true, mrv: false, fc: false, mac: false },
        { name: 'AC-3 + BT + FC', ac3: true, mrv: false, fc: true, mac: false },
        { name: 'AC-3 + BT + MRV', ac3: true, mrv: true, fc: false, mac: false },
        { name: 'AC-3 + BT + FC + MRV', ac3: true, mrv: true, fc: true, mac: false },
        { name: 'Backtracking + MAC', ac3: false, mrv: false, fc: false, mac: true },
        { name: 'Backtracking + MAC + MRV', ac3: false, mrv: true, fc: false, mac: true },
    ];

    /**
     * CSP State class - manages domains for all cells
     */
    class CSPState {
        constructor() {
            this.domains = {};
            this.fixed = new Set();
            CELLS.forEach(cell => {
                this.domains[cell] = new Set(DIGITS);
            });
        }

        clone() {
            const copy = new CSPState();
            CELLS.forEach(cell => {
                copy.domains[cell] = new Set(this.domains[cell]);
            });
            copy.fixed = new Set(this.fixed);
            return copy;
        }

        setFixed(cell, value) {
            this.domains[cell] = new Set([value]);
            this.fixed.add(cell);
        }

        isAssigned(cell) {
            return this.domains[cell].size === 1;
        }

        getValue(cell) {
            if (this.isAssigned(cell)) {
                return [...this.domains[cell]][0];
            }
            return null;
        }

        getDomain(cell) {
            return [...this.domains[cell]];
        }

        removeFromDomain(cell, value) {
            this.domains[cell].delete(value);
        }

        isSolved() {
            return CELLS.every(cell => this.isAssigned(cell));
        }

        isConsistent() {
            return CELLS.every(cell => this.domains[cell].size > 0);
        }

        getSnapshot() {
            const snapshot = {};
            CELLS.forEach(cell => {
                snapshot[cell] = [...this.domains[cell]];
            });
            return snapshot;
        }
    }

    /**
     * Solver class - implements AC-3, Backtracking, and MAC
     */
    class ShidokuSolver {
        constructor() {
            this.steps = [];
            this.metrics = {
                assignments: 0,
                backtracks: 0,
                arcChecks: 0,
                prunings: 0
            };
            // Tree tracking
            this.treeNodes = [];
            this.currentNodeId = null;
            this.nextNodeId = 0;
        }

        reset() {
            this.steps = [];
            this.metrics = {
                assignments: 0,
                backtracks: 0,
                arcChecks: 0,
                prunings: 0
            };
            this.treeNodes = [];
            this.currentNodeId = null;
            this.nextNodeId = 0;
        }

        /**
         * Create a tree node
         */
        createTreeNode(cell, value, parentId, depth, snapshot) {
            const node = {
                id: this.nextNodeId++,
                cell,
                value,
                parentId,
                children: [],
                depth,
                status: 'exploring', // 'exploring', 'success', 'backtracked'
                snapshot: snapshot // Full puzzle state at this point
            };
            this.treeNodes.push(node);

            // Link to parent
            if (parentId !== null) {
                const parent = this.treeNodes.find(n => n.id === parentId);
                if (parent) {
                    parent.children.push(node.id);
                }
            }

            return node;
        }

        /**
         * Create root node showing initial puzzle state
         */
        createRootNode(snapshot) {
            const node = {
                id: this.nextNodeId++,
                cell: null,
                value: null,
                parentId: null,
                children: [],
                depth: -1,
                status: 'exploring',
                snapshot: snapshot,
                isRoot: true
            };
            this.treeNodes.push(node);
            this.currentNodeId = node.id;
            return node;
        }

        /**
         * Get current tree state for visualization
         */
        getTreeState() {
            return {
                nodes: this.treeNodes.map(n => ({ ...n })),
                currentNodeId: this.currentNodeId
            };
        }

        addStep(type, state, metadata = {}) {
            this.steps.push({
                type,
                snapshot: state.getSnapshot(),
                metadata,
                metrics: { ...this.metrics },
                treeState: this.getTreeState()
            });
        }

        /**
         * Revise function for AC-3
         * Returns true if we removed any values from Xi's domain
         */
        revise(state, xi, xj) {
            let revised = false;
            const toRemove = [];

            for (const x of state.getDomain(xi)) {
                // Check if there exists any value in Xj's domain that is different from x
                let hasSupport = false;
                for (const y of state.getDomain(xj)) {
                    if (x !== y) {
                        hasSupport = true;
                        break;
                    }
                }
                if (!hasSupport) {
                    toRemove.push(x);
                    revised = true;
                }
            }

            toRemove.forEach(val => {
                // Record the step BEFORE removing the value so visualization shows it being pruned
                this.metrics.prunings++;
                this.addStep(StepType.AC3_PRUNE, state, {
                    cell: xi,
                    value: val,
                    reason: `No support in ${xj}`,
                    peer: xj,
                    valueToPrune: val  // Mark which value is about to be pruned
                });
                // Now actually remove the value
                state.removeFromDomain(xi, val);
            });

            return revised;
        }

        /**
         * Propagate constraints from fixed clues (used when AC-3 is disabled)
         * This ensures basic constraint satisfaction for fixed values
         */
        propagateFixedClues(state) {
            for (const cell of CELLS) {
                if (state.fixed.has(cell)) {
                    const value = state.getValue(cell);
                    // Remove this value from all peers
                    for (const peer of PEERS[cell]) {
                        if (state.domains[peer].has(value)) {
                            this.metrics.prunings++;
                            this.addStep(StepType.FIXED_PRUNE, state, {
                                cell: peer,
                                value: value,
                                reason: `Conflicts with fixed ${cell}=${value}`,
                                fixedCell: cell,
                                valueToPrune: value
                            });
                            state.removeFromDomain(peer, value);
                        }
                    }
                }
            }
        }

        /**
         * AC-3 Algorithm
         * Returns true if the CSP is still consistent after enforcing arc consistency
         */
        ac3(state, initialArcs = null) {
            const queue = initialArcs ? [...initialArcs] : [...ALL_ARCS];

            this.addStep(StepType.AC3_START, state, {
                queueSize: queue.length
            });

            while (queue.length > 0) {
                const [xi, xj] = queue.shift();
                this.metrics.arcChecks++;

                this.addStep(StepType.AC3_CHECK, state, {
                    cell: xi,
                    peer: xj,
                    queueSize: queue.length
                });

                if (this.revise(state, xi, xj)) {
                    if (state.domains[xi].size === 0) {
                        this.addStep(StepType.AC3_DONE, state, {
                            success: false,
                            reason: `Domain of ${xi} is empty`
                        });
                        return false;
                    }

                    // Add arcs (Xk, Xi) where Xk is a neighbor of Xi (except Xj)
                    PEERS[xi].forEach(xk => {
                        if (xk !== xj) {
                            queue.push([xk, xi]);
                        }
                    });
                }
            }

            this.addStep(StepType.AC3_DONE, state, {
                success: true
            });

            return true;
        }

        /**
         * Select first unassigned variable (simple, no heuristic)
         */
        selectUnassignedSimple(state) {
            for (const cell of CELLS) {
                if (!state.isAssigned(cell)) {
                    return cell;
                }
            }
            return null;
        }

        /**
         * Select unassigned variable using MRV heuristic
         */
        selectUnassignedMRV(state) {
            let best = null;
            let bestSize = Infinity;

            for (const cell of CELLS) {
                if (!state.isAssigned(cell)) {
                    const size = state.domains[cell].size;
                    if (size < bestSize) {
                        bestSize = size;
                        best = cell;
                        if (size === 2) break; // Can't do better than 2 for unassigned
                    }
                }
            }

            return best;
        }

        /**
         * Select unassigned variable based on useMRV flag
         */
        selectUnassigned(state, useMRV) {
            return useMRV ? this.selectUnassignedMRV(state) : this.selectUnassignedSimple(state);
        }

        /**
         * Backtracking search (optionally with MRV heuristic and forward checking)
         */
        backtrack(state, useMRV = false, useFC = false, depth = 0) {
            if (state.isSolved()) {
                // Mark current node and path as success
                if (this.currentNodeId !== null) {
                    this.markPathAsSuccess(this.currentNodeId);
                }
                this.addStep(StepType.SOLVED, state, {});
                return true;
            }

            const cell = this.selectUnassigned(state, useMRV);
            if (!cell) {
                return false;
            }

            const values = state.getDomain(cell);
            const parentNodeId = this.currentNodeId;

            for (const value of values) {
                // Make assignment (create state with this value)
                const newState = state.clone();
                newState.domains[cell] = new Set([value]);
                this.metrics.assignments++;

                // Create tree node for this attempt (shows the assignment)
                const node = this.createTreeNode(cell, value, parentNodeId, depth, newState.getSnapshot());
                this.currentNodeId = node.id;

                this.addStep(StepType.ASSIGN, newState, {
                    cell,
                    value,
                    treeNodeId: node.id,
                    depth
                });

                // Check if assignment is consistent with already-assigned peers
                let consistent = true;
                let conflictPeers = [];
                for (const peer of PEERS[cell]) {
                    if (state.isAssigned(peer) && state.getValue(peer) === value) {
                        consistent = false;
                        conflictPeers.push(peer);
                    }
                }

                if (!consistent) {
                    // Immediate conflict with existing assignment
                    node.status = 'backtracked';
                    node.conflictPeers = conflictPeers; // Store which cell(s) caused the conflict
                    this.currentNodeId = parentNodeId;
                    this.metrics.backtracks++;
                    this.addStep(StepType.BACKTRACK, newState, {
                        cell,
                        value,
                        treeNodeId: node.id,
                        depth,
                        reason: `Conflicts with ${conflictPeers.join(', ')}`
                    });
                    continue;
                }

                // Forward checking: remove value from peers' domains (only if enabled)
                let valid = true;
                if (useFC) {
                    for (const peer of PEERS[cell]) {
                        if (newState.domains[peer].has(value)) {
                            newState.domains[peer].delete(value);
                            if (newState.domains[peer].size === 0) {
                                valid = false;
                                break;
                            }
                        }
                    }
                }

                if (valid && this.backtrack(newState, useMRV, useFC, depth + 1)) {
                    // Copy solution back
                    CELLS.forEach(c => {
                        state.domains[c] = newState.domains[c];
                    });
                    return true;
                }

                // Mark this node as backtracked (recursive search failed)
                node.status = 'backtracked';
                this.currentNodeId = parentNodeId;

                this.metrics.backtracks++;
                this.addStep(StepType.BACKTRACK, state, {
                    cell,
                    value,
                    treeNodeId: node.id,
                    depth
                });
            }

            return false;
        }

        /**
         * Mark a node and its path to root as success
         */
        markPathAsSuccess(nodeId) {
            let current = this.treeNodes.find(n => n.id === nodeId);
            while (current) {
                current.status = 'success';
                if (current.parentId === null) break;
                current = this.treeNodes.find(n => n.id === current.parentId);
            }
        }

        /**
         * MAC (Maintaining Arc Consistency) - runs AC-3 after each assignment
         */
        mac(state, useMRV = false, depth = 0) {
            if (state.isSolved()) {
                // Mark current node and path as success
                if (this.currentNodeId !== null) {
                    this.markPathAsSuccess(this.currentNodeId);
                }
                this.addStep(StepType.SOLVED, state, {});
                return true;
            }

            const cell = this.selectUnassigned(state, useMRV);
            if (!cell) {
                return false;
            }

            const values = [...state.getDomain(cell)];
            const parentNodeId = this.currentNodeId;

            for (const value of values) {
                // Make assignment (create state with this value)
                const newState = state.clone();
                newState.domains[cell] = new Set([value]);
                this.metrics.assignments++;

                // Create tree node for this attempt (shows the assignment)
                const node = this.createTreeNode(cell, value, parentNodeId, depth, newState.getSnapshot());
                this.currentNodeId = node.id;

                this.addStep(StepType.ASSIGN, newState, {
                    cell,
                    value,
                    treeNodeId: node.id,
                    depth
                });

                // Check if assignment is consistent with already-assigned peers
                let consistent = true;
                let conflictPeers = [];
                for (const peer of PEERS[cell]) {
                    if (state.isAssigned(peer) && state.getValue(peer) === value) {
                        consistent = false;
                        conflictPeers.push(peer);
                    }
                }

                if (!consistent) {
                    // Immediate conflict with existing assignment
                    node.status = 'backtracked';
                    node.conflictPeers = conflictPeers; // Store which cell(s) caused the conflict
                    this.currentNodeId = parentNodeId;
                    this.metrics.backtracks++;
                    this.addStep(StepType.BACKTRACK, newState, {
                        cell,
                        value,
                        treeNodeId: node.id,
                        depth,
                        reason: `Conflicts with ${conflictPeers.join(', ')}`
                    });
                    continue;
                }

                // Run AC-3 on arcs involving neighbors
                const arcs = [];
                PEERS[cell].forEach(peer => {
                    PEERS[peer].forEach(neighbor => {
                        if (neighbor !== cell) {
                            arcs.push([neighbor, peer]);
                        }
                    });
                    arcs.push([peer, cell]);
                });

                this.addStep(StepType.MAC_CHECK, newState, {
                    cell,
                    arcsToCheck: arcs.length,
                    treeNodeId: node.id,
                    depth
                });

                if (this.ac3(newState, arcs) && this.mac(newState, useMRV, depth + 1)) {
                    // Copy solution back
                    CELLS.forEach(c => {
                        state.domains[c] = newState.domains[c];
                    });
                    return true;
                }

                // Mark this node as backtracked
                node.status = 'backtracked';
                this.currentNodeId = parentNodeId;

                this.metrics.backtracks++;
                this.addStep(StepType.BACKTRACK, state, {
                    cell,
                    value,
                    treeNodeId: node.id,
                    depth
                });
            }

            return false;
        }

        /**
         * Main solve function
         */
        solve(puzzle, useAC3 = false, useMRV = false, useFC = false, useMAC = false) {
            this.reset();

            const state = new CSPState();

            // Initialize from puzzle string
            // Only set the fixed cell's domain - do NOT propagate to peers yet
            // This lets AC-3 visualize all the constraint propagation
            for (let i = 0; i < puzzle.length && i < 16; i++) {
                const ch = puzzle[i];
                if (ch >= '1' && ch <= '4') {
                    const cell = CELLS[i];
                    const value = parseInt(ch);
                    state.setFixed(cell, value);
                    // Note: We intentionally don't remove from peers here
                    // AC-3 will handle propagation so it can be visualized
                }
            }

            // Create root node showing initial puzzle state
            const rootNode = this.createRootNode(state.getSnapshot());

            this.addStep(StepType.INIT, state, {
                puzzle,
                treeNodeId: rootNode.id
            });

            // Run AC-3 preprocessing if requested
            // AC-3 will propagate constraints from fixed clues and visualize them
            // If AC-3 is disabled, backtracking will discover conflicts naturally
            if (useAC3) {
                if (!this.ac3(state)) {
                    this.addStep(StepType.UNSOLVABLE, state, {
                        reason: 'AC-3 preprocessing found inconsistency'
                    });
                    return { success: false, steps: this.steps, metrics: this.metrics };
                }
            }

            // Run search
            let success;
            if (useMAC) {
                success = this.mac(state, useMRV);
            } else {
                success = this.backtrack(state, useMRV, useFC);
            }

            if (!success) {
                this.addStep(StepType.UNSOLVABLE, state, {
                    reason: 'No solution exists'
                });
            }

            return {
                success,
                steps: this.steps,
                metrics: this.metrics,
                solution: success ? state : null,
                treeNodes: this.treeNodes
            };
        }

        /**
         * Solve using AC-3 only (no backtracking search)
         * This will only succeed if AC-3 alone can deduce all values
         */
        solveAC3Only(puzzle) {
            this.reset();

            const state = new CSPState();

            // Initialize from puzzle string
            for (let i = 0; i < puzzle.length && i < 16; i++) {
                const ch = puzzle[i];
                if (ch >= '1' && ch <= '4') {
                    const cell = CELLS[i];
                    const value = parseInt(ch);
                    state.setFixed(cell, value);
                }
            }

            // Run AC-3
            const consistent = this.ac3(state);

            // Check if puzzle is solved (all cells have exactly one value)
            const success = consistent && state.isSolved();

            return {
                success,
                steps: this.steps,
                metrics: this.metrics,
                solution: success ? state : null,
                treeNodes: this.treeNodes
            };
        }
    }

    /**
     * Search Tree Visualizer - renders the backtracking search tree with mini-puzzle grids
     */
    class SearchTreeVisualizer {
        constructor(containerId, svgId) {
            this.container = document.getElementById(containerId);
            this.svg = document.getElementById(svgId);
            this.tooltip = document.getElementById('tree-tooltip');
            this.nodes = [];
            this.currentNodeId = null;
            this.zoom = 1;
            this.panX = 0;
            this.panY = 0;
            this.isDragging = false;
            this.dragStartX = 0;
            this.dragStartY = 0;

            // Layout settings for mini-grid nodes
            this.gridSize = 52; // Size of mini-grid (4x4 cells at 13px each)
            this.cellSize = 13; // Size of each cell in mini-grid
            this.levelHeight = 85; // Vertical spacing between levels
            this.nodeSpacing = 62; // Horizontal spacing between nodes

            this.initEventListeners();
            this.showEmptyMessage();
        }

        initEventListeners() {
            // Pan controls
            this.svg.addEventListener('mousedown', (e) => this.onMouseDown(e));
            this.svg.addEventListener('mousemove', (e) => this.onMouseMove(e));
            this.svg.addEventListener('mouseup', () => this.onMouseUp());
            this.svg.addEventListener('mouseleave', () => this.onMouseUp());

            // Zoom slider
            const zoomSlider = document.getElementById('tree-zoom-slider');
            if (zoomSlider) {
                zoomSlider.addEventListener('input', (e) => {
                    this.zoom = parseInt(e.target.value) / 100;
                    this.constrainPan();
                    this.updateTransform();
                });
            }

            // Mouse wheel zoom
            this.svg.addEventListener('wheel', (e) => {
                e.preventDefault();
                if (e.deltaY < 0) {
                    this.zoomIn();
                } else {
                    this.zoomOut();
                }
                // Update slider to match
                if (zoomSlider) {
                    zoomSlider.value = Math.round(this.zoom * 100);
                }
            });
        }

        showEmptyMessage() {
            this.svg.innerHTML = `
                <text x="50%" y="50%" text-anchor="middle" class="tree-empty-message" fill="currentColor">
                    Search tree will appear here during backtracking
                </text>
            `;
        }

        onMouseDown(e) {
            if (e.target === this.svg || e.target.tagName === 'svg') {
                this.isDragging = true;
                this.dragStartX = e.clientX - this.panX;
                this.dragStartY = e.clientY - this.panY;
                this.svg.style.cursor = 'grabbing';
            }
        }

        onMouseMove(e) {
            if (this.isDragging) {
                this.panX = e.clientX - this.dragStartX;
                this.panY = e.clientY - this.dragStartY;
                this.constrainPan();
                this.updateTransform();
            }
        }

        onMouseUp() {
            this.isDragging = false;
            this.svg.style.cursor = 'grab';
        }

        /**
         * Constrain panning so the tree stays at least partially visible
         */
        constrainPan() {
            if (this.nodes.length === 0) return;

            // Calculate tree bounds
            const bounds = this.getTreeBounds();
            if (!bounds) return;

            const containerWidth = this.container.clientWidth;
            const containerHeight = this.container.clientHeight;

            // Calculate scaled tree dimensions
            const scaledWidth = (bounds.maxX - bounds.minX) * this.zoom;
            const scaledHeight = (bounds.maxY - bounds.minY) * this.zoom;

            // Minimum visible margin (keep at least this much of tree visible)
            const margin = 50;

            // Calculate pan limits
            const minPanX = containerWidth - (bounds.maxX * this.zoom) - margin;
            const maxPanX = -(bounds.minX * this.zoom) + margin;
            const minPanY = containerHeight - (bounds.maxY * this.zoom) - margin;
            const maxPanY = -(bounds.minY * this.zoom) + margin;

            // Constrain pan values
            if (scaledWidth < containerWidth - margin * 2) {
                // Tree fits horizontally, center it
                this.panX = (containerWidth - scaledWidth) / 2 - bounds.minX * this.zoom;
            } else {
                this.panX = Math.max(minPanX, Math.min(maxPanX, this.panX));
            }

            if (scaledHeight < containerHeight - margin * 2) {
                // Tree fits vertically, keep it at top
                this.panY = margin - bounds.minY * this.zoom;
            } else {
                this.panY = Math.max(minPanY, Math.min(maxPanY, this.panY));
            }
        }

        /**
         * Get the bounding box of the tree
         */
        getTreeBounds() {
            if (this.nodes.length === 0) return null;

            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            const halfGrid = this.gridSize / 2;

            this.nodes.forEach(node => {
                if (node.x !== undefined) {
                    minX = Math.min(minX, node.x - halfGrid);
                    maxX = Math.max(maxX, node.x + halfGrid);
                    minY = Math.min(minY, node.y - halfGrid);
                    maxY = Math.max(maxY, node.y + halfGrid);
                }
            });

            if (minX === Infinity) return null;
            return { minX, maxX, minY, maxY };
        }

        zoomIn() {
            this.zoom = Math.min(1.5, this.zoom * 1.2);
            this.constrainPan();
            this.updateTransform();
        }

        zoomOut() {
            this.zoom = Math.max(0.2, this.zoom / 1.2);
            this.constrainPan();
            this.updateTransform();
        }

        zoomFit() {
            const bounds = this.getTreeBounds();
            if (!bounds) return;

            const { minX, maxX, minY, maxY } = bounds;
            const padding = 40;
            const treeWidth = maxX - minX + padding * 2;
            const treeHeight = maxY - minY + padding * 2;
            const containerWidth = this.container.clientWidth;
            const containerHeight = this.container.clientHeight;

            this.zoom = Math.min(
                containerWidth / treeWidth,
                containerHeight / treeHeight,
                1.2
            );
            this.zoom = Math.max(0.2, this.zoom);

            // Center the tree
            const centerX = (minX + maxX) / 2;
            this.panX = containerWidth / 2 - centerX * this.zoom;
            this.panY = padding * this.zoom - minY * this.zoom;

            // Update slider
            const zoomSlider = document.getElementById('tree-zoom-slider');
            if (zoomSlider) {
                zoomSlider.value = Math.round(this.zoom * 100);
            }

            this.updateTransform();
        }

        updateTransform() {
            const group = this.svg.querySelector('.tree-group');
            if (group) {
                group.setAttribute('transform', `translate(${this.panX}, ${this.panY}) scale(${this.zoom})`);
            }
        }

        /**
         * Update the tree visualization with new state
         */
        update(treeState) {
            if (!treeState || !treeState.nodes || treeState.nodes.length === 0) {
                this.showEmptyMessage();
                this.updateInfo(0, 0);
                return;
            }

            this.nodes = treeState.nodes;
            this.currentNodeId = treeState.currentNodeId;

            // Calculate layout
            this.calculateLayout();

            // Render
            this.render();

            // Update info
            const maxDepth = Math.max(...this.nodes.map(n => n.depth), 0);
            this.updateInfo(maxDepth + 1, this.nodes.length);

            // Auto-fit on first node
            if (this.nodes.length === 1) {
                setTimeout(() => this.zoomFit(), 50);
            }
        }

        /**
         * Calculate positions for all nodes using a proper tree layout
         */
        calculateLayout() {
            if (this.nodes.length === 0) return;

            // Find root node
            const root = this.nodes.find(n => n.parentId === null);
            if (!root) return;

            // First pass: calculate the width needed for each subtree
            const subtreeWidths = {};
            this.calculateSubtreeWidths(root, subtreeWidths);

            // Second pass: assign x positions based on subtree widths
            const containerWidth = this.container.clientWidth;
            const startX = containerWidth / 2;
            this.assignXPositions(root, startX, subtreeWidths);

            // Assign y positions based on depth (normalize depth so root is at 0)
            const minDepth = Math.min(...this.nodes.map(n => n.depth));
            this.nodes.forEach(node => {
                const normalizedDepth = node.depth - minDepth;
                node.y = 40 + normalizedDepth * this.levelHeight;
            });
        }

        /**
         * Calculate width needed for each node's subtree
         */
        calculateSubtreeWidths(node, widths) {
            const children = this.nodes.filter(n => n.parentId === node.id);

            if (children.length === 0) {
                widths[node.id] = this.nodeSpacing;
            } else {
                let totalWidth = 0;
                children.forEach(child => {
                    this.calculateSubtreeWidths(child, widths);
                    totalWidth += widths[child.id];
                });
                widths[node.id] = Math.max(this.nodeSpacing, totalWidth);
            }

            return widths[node.id];
        }

        /**
         * Assign x positions to nodes based on subtree widths
         */
        assignXPositions(node, centerX, widths) {
            node.x = centerX;

            const children = this.nodes.filter(n => n.parentId === node.id);
            if (children.length === 0) return;

            // Calculate starting position for children
            const totalWidth = children.reduce((sum, c) => sum + widths[c.id], 0);
            let currentX = centerX - totalWidth / 2;

            children.forEach(child => {
                const childWidth = widths[child.id];
                const childCenterX = currentX + childWidth / 2;
                this.assignXPositions(child, childCenterX, widths);
                currentX += childWidth;
            });
        }

        /**
         * Render the tree to SVG
         */
        render() {
            this.svg.innerHTML = '';

            // Create main group for transformations
            const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.classList.add('tree-group');
            this.svg.appendChild(group);

            // Draw edges first (so they're behind nodes)
            this.nodes.forEach(node => {
                if (node.parentId !== null) {
                    const parent = this.nodes.find(n => n.id === node.parentId);
                    if (parent && parent.x !== undefined && node.x !== undefined) {
                        this.drawEdge(group, parent, node);
                    }
                }
            });

            // Draw nodes
            this.nodes.forEach(node => {
                if (node.x !== undefined) {
                    this.drawNode(group, node);
                }
            });

            this.updateTransform();
        }

        /**
         * Draw an edge between two nodes
         */
        drawEdge(group, parent, child) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.classList.add('tree-edge');

            // Connect from bottom center of parent to top center of child
            const halfGrid = this.gridSize / 2;
            line.setAttribute('x1', parent.x);
            line.setAttribute('y1', parent.y + halfGrid);
            line.setAttribute('x2', child.x);
            line.setAttribute('y2', child.y - halfGrid);

            // Set edge style based on child status
            if (child.status === 'success') {
                line.classList.add('success');
            } else if (child.status === 'backtracked') {
                line.classList.add('backtracked');
            } else if (child.id === this.currentNodeId) {
                line.classList.add('active');
            }

            group.appendChild(line);
        }

        /**
         * Draw a tree node as a mini 4x4 puzzle grid
         */
        drawNode(group, node) {
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.classList.add('tree-node');

            // Position so node.x, node.y is the center
            const halfGrid = this.gridSize / 2;
            g.setAttribute('transform', `translate(${node.x - halfGrid}, ${node.y - halfGrid})`);

            // Determine status class for border styling
            // Note: success takes precedence over current (so final solution shows green)
            let statusClass = 'exploring';
            if (node.isRoot) {
                statusClass = node.status === 'success' ? 'on-solution-path' : 'root';
            } else if (node.status === 'success') {
                statusClass = 'on-solution-path';
            } else if (node.id === this.currentNodeId) {
                statusClass = 'current';
            } else if (node.status === 'backtracked') {
                statusClass = 'backtracked';
            }

            // Determine cell highlight color based on status
            // Yellow = attempted assignment, Red = conflict cells, Green = success
            const isFailed = node.status === 'backtracked';
            const isSuccess = node.status === 'success';
            const hasConflict = isFailed && node.conflictPeers && node.conflictPeers.length > 0;

            // Background rectangle for the grid
            const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            bg.classList.add('tree-grid-bg', statusClass);
            bg.setAttribute('x', 0);
            bg.setAttribute('y', 0);
            bg.setAttribute('width', this.gridSize);
            bg.setAttribute('height', this.gridSize);
            bg.setAttribute('rx', 2);
            g.appendChild(bg);

            // Draw the 4x4 grid cells
            if (node.snapshot) {
                for (let r = 0; r < N; r++) {
                    for (let c = 0; c < N; c++) {
                        const cellName = ROWS[r] + COLS[c];
                        const domain = node.snapshot[cellName];
                        const x = c * this.cellSize;
                        const y = r * this.cellSize;

                        // Cell background
                        const cellRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                        cellRect.classList.add('tree-grid-cell');
                        cellRect.setAttribute('x', x);
                        cellRect.setAttribute('y', y);
                        cellRect.setAttribute('width', this.cellSize);
                        cellRect.setAttribute('height', this.cellSize);

                        // Highlight the cell that was just assigned (always yellow for the attempt)
                        // Yellow for attempted assignment, Green for success, Red for conflict cells
                        if (cellName === node.cell) {
                            if (isSuccess) {
                                cellRect.classList.add('tree-grid-cell-success');
                            } else {
                                // Yellow for both exploring and failed (shows what was attempted)
                                cellRect.classList.add('tree-grid-cell-exploring');
                            }
                        }
                        // Highlight conflict cells in red (the cells that caused the failure)
                        else if (hasConflict && node.conflictPeers.includes(cellName)) {
                            cellRect.classList.add('tree-grid-cell-failed');
                        }

                        g.appendChild(cellRect);

                        // Draw value if cell is assigned (domain has 1 value)
                        if (domain && domain.length === 1) {
                            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                            text.classList.add('tree-grid-value');
                            text.setAttribute('x', x + this.cellSize / 2);
                            text.setAttribute('y', y + this.cellSize / 2 + 3);
                            text.textContent = domain[0];

                            // Style the assigned cell's value
                            if (cellName === node.cell) {
                                if (isSuccess) {
                                    text.classList.add('tree-grid-value-success');
                                } else {
                                    // Yellow for both exploring and failed attempts
                                    text.classList.add('tree-grid-value-exploring');
                                }
                            }
                            // Style conflict cells' values in red
                            else if (hasConflict && node.conflictPeers.includes(cellName)) {
                                text.classList.add('tree-grid-value-failed');
                            }

                            g.appendChild(text);
                        }
                    }
                }

                // Draw 2x2 box borders
                const boxBorder1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                boxBorder1.classList.add('tree-grid-box-border');
                boxBorder1.setAttribute('x1', this.cellSize * 2);
                boxBorder1.setAttribute('y1', 0);
                boxBorder1.setAttribute('x2', this.cellSize * 2);
                boxBorder1.setAttribute('y2', this.gridSize);
                g.appendChild(boxBorder1);

                const boxBorder2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                boxBorder2.classList.add('tree-grid-box-border');
                boxBorder2.setAttribute('x1', 0);
                boxBorder2.setAttribute('y1', this.cellSize * 2);
                boxBorder2.setAttribute('x2', this.gridSize);
                boxBorder2.setAttribute('y2', this.cellSize * 2);
                g.appendChild(boxBorder2);
            }

            // Tooltip events
            g.addEventListener('mouseenter', (e) => this.showTooltip(e, node));
            g.addEventListener('mouseleave', () => this.hideTooltip());

            group.appendChild(g);
        }

        showTooltip(e, node) {
            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left + 10;
            const y = e.clientY - rect.top + 10;

            let statusText = node.status;
            if (node.id === this.currentNodeId) statusText = 'current';

            this.tooltip.innerHTML = `
                <div class="tree-tooltip-title">${node.cell} = ${node.value}</div>
                <div class="tree-tooltip-info">
                    Depth: ${node.depth}<br>
                    Status: ${statusText}<br>
                    Node ID: ${node.id}
                </div>
            `;

            this.tooltip.style.left = x + 'px';
            this.tooltip.style.top = y + 'px';
            this.tooltip.classList.add('visible');
        }

        hideTooltip() {
            this.tooltip.classList.remove('visible');
        }

        updateInfo(depth, nodes) {
            const depthEl = document.getElementById('tree-depth');
            const nodesEl = document.getElementById('tree-nodes');
            if (depthEl) depthEl.textContent = `Depth: ${depth}`;
            if (nodesEl) nodesEl.textContent = `Nodes: ${nodes}`;
        }

        clear() {
            this.nodes = [];
            this.currentNodeId = null;
            this.zoom = 1;
            this.panX = 0;
            this.panY = 0;
            this.showEmptyMessage();
            this.updateInfo(0, 0);
        }
    }

    /**
     * Playback Controller - manages visualization playback
     */
    class PlaybackController {
        constructor(ui, treeViz) {
            this.ui = ui;
            this.treeViz = treeViz;
            this.steps = [];
            this.currentStep = 0;
            this.playing = false;
            this.paused = false;
            this.speed = 5;
            this.timeoutId = null;
            this.startTime = null;
        }

        load(steps) {
            this.steps = steps;
            this.currentStep = 0;
            this.playing = false;
            this.paused = false;
        }

        getDelay() {
            // Speed 1 = 1000ms, Speed 10 = 50ms
            return 1050 - (this.speed * 100);
        }

        setSpeed(speed) {
            this.speed = Math.max(1, Math.min(10, speed));
        }

        play() {
            if (this.steps.length === 0) return;

            this.playing = true;
            this.paused = false;
            this.startTime = Date.now();
            this.ui.setPlayingState(true);
            this.advance();
        }

        pause() {
            this.paused = true;
            this.playing = false;
            if (this.timeoutId) {
                clearTimeout(this.timeoutId);
                this.timeoutId = null;
            }
            this.ui.setPlayingState(false);
        }

        resume() {
            if (!this.paused) return;
            this.paused = false;
            this.playing = true;
            this.ui.setPlayingState(true);
            this.advance();
        }

        step() {
            if (this.currentStep < this.steps.length) {
                this.renderStep(this.steps[this.currentStep]);
                this.currentStep++;

                if (this.currentStep >= this.steps.length) {
                    this.playing = false;
                    this.ui.setPlayingState(false);
                    this.ui.setFinishedState();
                }
            }
        }

        advance() {
            if (!this.playing || this.paused) return;

            if (this.currentStep < this.steps.length) {
                this.renderStep(this.steps[this.currentStep]);
                this.currentStep++;

                if (this.currentStep < this.steps.length) {
                    this.timeoutId = setTimeout(() => this.advance(), this.getDelay());
                } else {
                    this.playing = false;
                    this.ui.setPlayingState(false);
                    this.ui.setFinishedState();
                }
            }
        }

        renderStep(step) {
            this.ui.renderState(step.snapshot, step.type, step.metadata);
            this.ui.updateMetrics(step.metrics);
            this.ui.addLogEntry(step.type, step.metadata, step.snapshot);

            // Update tree visualization
            if (this.treeViz && step.treeState) {
                this.treeViz.update(step.treeState);
            }

            if (this.startTime) {
                const elapsed = (Date.now() - this.startTime) / 1000;
                this.ui.updateElapsedTime(elapsed);
            }
        }

        stop() {
            this.playing = false;
            this.paused = false;
            if (this.timeoutId) {
                clearTimeout(this.timeoutId);
                this.timeoutId = null;
            }
            this.currentStep = 0;
        }

        isPlaying() {
            return this.playing;
        }

        isPaused() {
            return this.paused;
        }
    }

    /**
     * UI Controller - manages DOM interactions
     */
    class UIController {
        constructor() {
            this.gridEl = document.getElementById('shidoku-grid');
            this.cells = {};
            this.puzzle = '................';
            this.selectedCell = null;
            this.solving = false;
            this.fixedCells = new Set();
            this.svgOverlay = null;
            this.failedAttempts = {};  // Track failed values per cell: cell -> Set of failed values
            this.userSelections = {};  // Track user-selected values for custom puzzle input: cell -> value (1-4)
            this.onPuzzleChange = null;  // Callback when puzzle changes

            this.initGrid();
            this.initEventListeners();
        }

        initGrid() {
            // Wrap grid in a container for positioning the SVG overlay
            const wrapper = document.createElement('div');
            wrapper.className = 'shidoku-grid-wrapper';

            // Move grid into wrapper
            const parent = this.gridEl.parentNode;
            parent.insertBefore(wrapper, this.gridEl);
            wrapper.appendChild(this.gridEl);

            this.gridEl.innerHTML = '';

            CELLS.forEach((cell, index) => {
                const div = document.createElement('div');
                div.className = 'shidoku-cell';
                div.dataset.cell = cell;
                div.dataset.index = index;

                // Create domain grid
                const domainGrid = document.createElement('div');
                domainGrid.className = 'domain-grid';
                for (let d = 1; d <= 4; d++) {
                    const dv = document.createElement('div');
                    dv.className = 'domain-value';
                    dv.dataset.value = d;
                    dv.textContent = d;
                    dv.addEventListener('click', (e) => {
                        e.stopPropagation();  // Don't trigger cell click
                        this.onDomainValueClick(cell, d);
                    });
                    domainGrid.appendChild(dv);
                }
                div.appendChild(domainGrid);

                div.addEventListener('click', () => this.onCellClick(cell));
                this.gridEl.appendChild(div);
                this.cells[cell] = div;
            });

            // Create SVG overlay for arc lines
            this.svgOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            this.svgOverlay.classList.add('arc-overlay');
            wrapper.appendChild(this.svgOverlay);
        }

        /**
         * Get the center position of a cell relative to the grid wrapper
         */
        getCellCenter(cell) {
            const cellEl = this.cells[cell];
            const wrapperRect = this.gridEl.parentNode.getBoundingClientRect();
            const cellRect = cellEl.getBoundingClientRect();

            return {
                x: cellRect.left - wrapperRect.left + cellRect.width / 2,
                y: cellRect.top - wrapperRect.top + cellRect.height / 2
            };
        }

        /**
         * Draw an arc line between two cells
         */
        drawArcLine(fromCell, toCell) {
            this.clearArcLine();

            const from = this.getCellCenter(fromCell);
            const to = this.getCellCenter(toCell);

            // Calculate the direction vector
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const unitX = dx / length;
            const unitY = dy / length;

            // Offset start and end points to be at cell edges (not centers)
            const offset = 30; // Half cell size minus some padding
            const startX = from.x + unitX * offset;
            const startY = from.y + unitY * offset;
            const endX = to.x - unitX * offset;
            const endY = to.y - unitY * offset;

            // Create glow effect line (wider, blurred)
            const glowLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            glowLine.classList.add('arc-line-glow');
            glowLine.setAttribute('x1', startX);
            glowLine.setAttribute('y1', startY);
            glowLine.setAttribute('x2', endX);
            glowLine.setAttribute('y2', endY);
            this.svgOverlay.appendChild(glowLine);

            // Create main arc line
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.classList.add('arc-line');
            line.setAttribute('x1', startX);
            line.setAttribute('y1', startY);
            line.setAttribute('x2', endX);
            line.setAttribute('y2', endY);
            this.svgOverlay.appendChild(line);

            // Create arrowhead pointing to the target cell
            const arrowSize = 10;
            const arrowAngle = Math.atan2(dy, dx);
            const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            arrow.classList.add('arc-arrow');

            const arrowTipX = endX;
            const arrowTipY = endY;
            const arrowBase1X = arrowTipX - arrowSize * Math.cos(arrowAngle - Math.PI / 6);
            const arrowBase1Y = arrowTipY - arrowSize * Math.sin(arrowAngle - Math.PI / 6);
            const arrowBase2X = arrowTipX - arrowSize * Math.cos(arrowAngle + Math.PI / 6);
            const arrowBase2Y = arrowTipY - arrowSize * Math.sin(arrowAngle + Math.PI / 6);

            arrow.setAttribute('points', `${arrowTipX},${arrowTipY} ${arrowBase1X},${arrowBase1Y} ${arrowBase2X},${arrowBase2Y}`);
            this.svgOverlay.appendChild(arrow);
        }

        /**
         * Clear the arc line
         */
        clearArcLine() {
            while (this.svgOverlay.firstChild) {
                this.svgOverlay.removeChild(this.svgOverlay.firstChild);
            }
        }

        /**
         * Highlight specific domain values in a cell
         * @param {string} cell - Cell name
         * @param {number[]} values - Values to highlight
         * @param {string} className - CSS class to add
         * @param {boolean} includePruned - Whether to also highlight pruned values (for no-support case)
         */
        highlightDomainValues(cell, values, className, includePruned = false) {
            const cellEl = this.cells[cell];
            const domainGrid = cellEl.querySelector('.domain-grid');
            if (!domainGrid) return;

            values.forEach(val => {
                const dv = domainGrid.querySelector(`[data-value="${val}"]`);
                if (dv) {
                    // Skip pruned values unless includePruned is true
                    if (!dv.classList.contains('pruned') || includePruned) {
                        dv.classList.add(className);
                    }
                }
            });
        }

        /**
         * Clear all domain value highlights
         */
        clearDomainHighlights() {
            document.querySelectorAll('.domain-value.checking, .domain-value.support, .domain-value.no-support').forEach(el => {
                el.classList.remove('checking', 'support', 'no-support');
            });
        }

        initEventListeners() {
            // Keyboard input for cell editing
            document.addEventListener('keydown', (e) => this.onKeyDown(e));

            // Sample puzzle buttons
            document.querySelectorAll('[data-puzzle]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const puzzleKey = btn.dataset.puzzle;
                    if (SAMPLE_PUZZLES[puzzleKey]) {
                        // Update active state on puzzle buttons
                        document.querySelectorAll('.puzzle-header-tabs [data-puzzle]').forEach(b => {
                            b.classList.remove('active');
                        });
                        btn.classList.add('active');

                        this.loadPuzzle(SAMPLE_PUZZLES[puzzleKey]);
                    }
                });
            });

            // Clear log button
            document.getElementById('btn-clear-log').addEventListener('click', () => {
                this.clearLog();
            });

            // Tab switching for info panel
            document.querySelectorAll('.info-panel-tabs [data-tab]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tabName = btn.dataset.tab;

                    // Update button active states
                    document.querySelectorAll('.info-panel-tabs [data-tab]').forEach(b => {
                        b.classList.remove('active');
                    });
                    btn.classList.add('active');

                    // Update tab content visibility
                    document.querySelectorAll('.info-tab-content').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    const targetTab = document.getElementById('tab-' + tabName);
                    if (targetTab) {
                        targetTab.classList.add('active');
                    }
                });
            });

            // Click outside to deselect
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.shidoku-cell') && this.selectedCell) {
                    this.deselectCell();
                }
            });
        }

        onCellClick(cell) {
            if (this.solving) return;

            if (this.selectedCell === cell) {
                this.deselectCell();
            } else {
                this.selectCell(cell);
            }
        }

        selectCell(cell) {
            if (this.selectedCell) {
                this.cells[this.selectedCell].classList.remove('input-mode');
            }
            this.selectedCell = cell;
            this.cells[cell].classList.add('input-mode');
        }

        deselectCell() {
            if (this.selectedCell) {
                this.cells[this.selectedCell].classList.remove('input-mode');
                this.selectedCell = null;
            }
        }

        /**
         * Handle click on a domain value in the mini-grid
         * Used for interactive custom puzzle input
         */
        onDomainValueClick(cell, value) {
            if (this.solving) return;
            if (this.fixedCells.has(cell)) return;  // Can't edit preset clues

            const currentSelection = this.userSelections[cell];

            if (currentSelection === value) {
                // Toggle off
                delete this.userSelections[cell];
                this.updateCellDisplay(cell);
                // Notify listeners that puzzle changed
                if (this.onPuzzleChange) {
                    this.onPuzzleChange();
                }
            } else {
                // Check for conflicts
                const conflictCells = this.checkConflicts(cell, value);
                if (conflictCells.length > 0) {
                    // Pulse conflict cells red
                    this.flashConflicts(conflictCells);
                } else {
                    // Select new value
                    this.userSelections[cell] = value;
                    this.updateCellDisplay(cell);
                    // Notify listeners that puzzle changed
                    if (this.onPuzzleChange) {
                        this.onPuzzleChange();
                    }
                }
            }
        }

        /**
         * Check if selecting a value would conflict with other cells
         * Uses PEERS structure to check row, column, and box constraints
         */
        checkConflicts(cell, value) {
            const conflicts = [];
            for (const peer of PEERS[cell]) {
                // Check against other user selections
                if (this.userSelections[peer] === value) {
                    conflicts.push(peer);
                }
                // Check against fixed cells (preset puzzles)
                if (this.fixedCells.has(peer)) {
                    const peerValue = parseInt(this.puzzle[CELLS.indexOf(peer)]);
                    if (peerValue === value) {
                        conflicts.push(peer);
                    }
                }
            }
            return conflicts;
        }

        /**
         * Flash conflict cells red to indicate invalid selection
         */
        flashConflicts(cells) {
            cells.forEach(cell => {
                const cellEl = this.cells[cell];
                cellEl.classList.add('conflict-cell');
                // Also flash the value
                const valueEl = cellEl.querySelector('.domain-value.user-selected, .assigned-value');
                if (valueEl) valueEl.classList.add('conflict-value');
            });

            setTimeout(() => {
                cells.forEach(cell => {
                    const cellEl = this.cells[cell];
                    cellEl.classList.remove('conflict-cell');
                    const valueEl = cellEl.querySelector('.conflict-value');
                    if (valueEl) valueEl.classList.remove('conflict-value');
                });
            }, 600);
        }

        /**
         * Update a cell's display to show current user selection state
         */
        updateCellDisplay(cell) {
            const cellEl = this.cells[cell];
            const selectedValue = this.userSelections[cell];

            // Rebuild domain grid with selection state
            cellEl.innerHTML = '';
            const domainGrid = document.createElement('div');
            domainGrid.className = 'domain-grid';

            for (let d = 1; d <= 4; d++) {
                const dv = document.createElement('div');
                dv.className = 'domain-value';
                dv.dataset.value = d;
                dv.textContent = d;

                if (selectedValue === d) {
                    dv.classList.add('user-selected');
                }

                dv.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.onDomainValueClick(cell, d);
                });

                domainGrid.appendChild(dv);
            }
            cellEl.appendChild(domainGrid);
        }

        /**
         * Convert user selections to fixed cells for solving
         * Called before running the solver
         */
        applyUserSelections() {
            for (const [cell, value] of Object.entries(this.userSelections)) {
                const index = CELLS.indexOf(cell);
                const puzzleArr = this.puzzle.split('');
                puzzleArr[index] = value.toString();
                this.puzzle = puzzleArr.join('');
                this.fixedCells.add(cell);
            }

            // Update UI to show fixed style
            for (const [cell, value] of Object.entries(this.userSelections)) {
                const cellEl = this.cells[cell];
                cellEl.classList.add('fixed');
                cellEl.innerHTML = `<span class="assigned-value">${value}</span>`;
            }

            // Clear user selections
            this.userSelections = {};
        }

        onKeyDown(e) {
            if (!this.selectedCell || this.solving) return;

            const key = e.key;
            const index = CELLS.indexOf(this.selectedCell);

            if (key >= '1' && key <= '4') {
                this.setCellValue(this.selectedCell, key);
                e.preventDefault();
            } else if (key === '0' || key === 'Backspace' || key === 'Delete') {
                this.clearCellValue(this.selectedCell);
                e.preventDefault();
            } else if (key === 'ArrowRight' && index < 15) {
                this.selectCell(CELLS[index + 1]);
                e.preventDefault();
            } else if (key === 'ArrowLeft' && index > 0) {
                this.selectCell(CELLS[index - 1]);
                e.preventDefault();
            } else if (key === 'ArrowDown' && index < 12) {
                this.selectCell(CELLS[index + 4]);
                e.preventDefault();
            } else if (key === 'ArrowUp' && index >= 4) {
                this.selectCell(CELLS[index - 4]);
                e.preventDefault();
            } else if (key === 'Escape') {
                this.deselectCell();
                e.preventDefault();
            }
        }

        setCellValue(cell, value) {
            const index = CELLS.indexOf(cell);
            const puzzleArr = this.puzzle.split('');
            puzzleArr[index] = value;
            this.puzzle = puzzleArr.join('');
            this.fixedCells.add(cell);

            const cellEl = this.cells[cell];
            cellEl.classList.add('fixed');
            cellEl.innerHTML = `<span class="assigned-value">${value}</span>`;
        }

        clearCellValue(cell) {
            const index = CELLS.indexOf(cell);
            const puzzleArr = this.puzzle.split('');
            puzzleArr[index] = '.';
            this.puzzle = puzzleArr.join('');
            this.fixedCells.delete(cell);

            const cellEl = this.cells[cell];
            cellEl.classList.remove('fixed');

            // Restore domain grid
            cellEl.innerHTML = '';
            const domainGrid = document.createElement('div');
            domainGrid.className = 'domain-grid';
            for (let d = 1; d <= 4; d++) {
                const dv = document.createElement('div');
                dv.className = 'domain-value';
                dv.dataset.value = d;
                dv.textContent = d;
                domainGrid.appendChild(dv);
            }
            cellEl.appendChild(domainGrid);
        }

        loadPuzzle(puzzleStr) {
            this.puzzle = puzzleStr.padEnd(16, '.').substring(0, 16);
            this.fixedCells.clear();
            this.solving = false;
            this.failedAttempts = {};  // Reset failed attempts tracking
            this.userSelections = {};  // Clear any user selections

            // Clear any existing arc visualization
            this.clearArcLine();
            this.clearDomainHighlights();

            CELLS.forEach((cell, index) => {
                const ch = this.puzzle[index];
                const cellEl = this.cells[cell];

                // Reset cell state (remove all state classes)
                cellEl.className = 'shidoku-cell';
                cellEl.classList.remove('arc-source', 'arc-target');

                if (ch >= '1' && ch <= '4') {
                    this.fixedCells.add(cell);
                    cellEl.classList.add('fixed');
                    cellEl.innerHTML = `<span class="assigned-value">${ch}</span>`;
                } else {
                    // Show domain grid with click handlers
                    cellEl.innerHTML = '';
                    const domainGrid = document.createElement('div');
                    domainGrid.className = 'domain-grid';
                    for (let d = 1; d <= 4; d++) {
                        const dv = document.createElement('div');
                        dv.className = 'domain-value';
                        dv.dataset.value = d;
                        dv.textContent = d;
                        dv.addEventListener('click', (e) => {
                            e.stopPropagation();
                            this.onDomainValueClick(cell, d);
                        });
                        domainGrid.appendChild(dv);
                    }
                    cellEl.appendChild(domainGrid);
                }
            });

            this.resetMetrics();
            this.clearLog();
            this.addLogEntry(StepType.INIT, { message: 'Puzzle loaded. Click Solve to start.' });

            // Notify listeners that puzzle changed
            if (this.onPuzzleChange) {
                this.onPuzzleChange();
            }
        }

        renderState(snapshot, stepType, metadata) {
            // Clear previous arc visualization
            this.clearArcLine();
            this.clearDomainHighlights();

            // Track failed attempts when we see a BACKTRACK step
            if (stepType === StepType.BACKTRACK && metadata.cell && metadata.value) {
                if (!this.failedAttempts[metadata.cell]) {
                    this.failedAttempts[metadata.cell] = new Set();
                }
                this.failedAttempts[metadata.cell].add(metadata.value);
            }

            // Clear failed attempts for cells that are no longer assigned
            // (we've backtracked past them, so their search history resets)
            // But DON'T clear the cell we just backtracked from - it still has failed attempts to show
            const backtrackCell = (stepType === StepType.BACKTRACK) ? metadata.cell : null;
            CELLS.forEach(cell => {
                const domain = snapshot[cell];
                // If cell is not assigned (domain > 1), not fixed, and not the backtrack cell, clear its failed attempts
                if (domain.length > 1 && !this.fixedCells.has(cell) && cell !== backtrackCell) {
                    delete this.failedAttempts[cell];
                }
            });

            // Determine conflict cells for BACKTRACK steps
            let conflictCellSet = new Set();
            if (stepType === StepType.BACKTRACK && metadata.reason) {
                const conflictMatch = metadata.reason.match(/Conflicts with (.+)/);
                if (conflictMatch) {
                    conflictMatch[1].split(', ').forEach(c => conflictCellSet.add(c));
                }
            }

            CELLS.forEach(cell => {
                const cellEl = this.cells[cell];
                const domain = snapshot[cell];

                // Reset state classes (keep fixed)
                cellEl.classList.remove('assigned', 'examining', 'conflict', 'arc-source', 'arc-target', 'trying-cell', 'provisional-cell', 'conflict-cell');

                if (this.fixedCells.has(cell)) {
                    // Fixed cells don't change
                    return;
                }

                // Check if this cell is being assigned/tried in current step
                const isTryingCell = (stepType === StepType.ASSIGN || stepType === StepType.TRY_VALUE)
                                     && metadata.cell === cell;

                // During backtracking steps (ASSIGN, TRY_VALUE, BACKTRACK), always show domain mini-grid
                const showMiniGridDuringBacktrack = stepType === StepType.ASSIGN ||
                                                    stepType === StepType.TRY_VALUE ||
                                                    stepType === StepType.BACKTRACK ||
                                                    stepType === StepType.SOLVED;

                // Check if we've reached the solution
                const isSolved = stepType === StepType.SOLVED;

                // Check if this cell has been assigned during backtracking (domain reduced to 1)
                const isAssignedDuringBacktrack = showMiniGridDuringBacktrack && domain.length === 1;

                if (domain.length === 1 && !showMiniGridDuringBacktrack) {
                    // Assigned cell (not during backtracking visualization)
                    cellEl.innerHTML = `<span class="assigned-value">${domain[0]}</span>`;
                    cellEl.classList.add('assigned');
                } else {
                    // Show domain values as mini-grid
                    cellEl.innerHTML = '';
                    const domainGrid = document.createElement('div');
                    domainGrid.className = 'domain-grid';

                    // Get failed attempts for this cell
                    const cellFailedAttempts = this.failedAttempts[cell] || new Set();

                    for (let d = 1; d <= 4; d++) {
                        const dv = document.createElement('div');
                        dv.className = 'domain-value';
                        dv.dataset.value = d;
                        dv.textContent = d;

                        // Check if this value was tried and failed (backtracked)
                        const isFailedAttempt = cellFailedAttempts.has(d);

                        // Check if this is the value currently being tried
                        const isCurrentlyTrying = isTryingCell && metadata.value === d;

                        // Check if this is the assigned value for a previously assigned cell
                        const isAssignedValue = isAssignedDuringBacktrack && domain[0] === d;

                        // Check if this cell is a conflict cell
                        const isConflictCell = conflictCellSet.has(cell);

                        if (isCurrentlyTrying) {
                            // Currently being tried - blue highlight with pulse
                            dv.classList.add('trying');
                        } else if (isFailedAttempt) {
                            // This value was already tried and failed - cross it off
                            dv.classList.add('failed-attempt');
                        } else if (isAssignedValue) {
                            // Assigned value - style depends on state
                            if (isConflictCell) {
                                dv.classList.add('conflict-value');  // Red - conflict
                            } else if (isSolved) {
                                dv.classList.add('assigned-mini');  // Green - confirmed
                            } else {
                                dv.classList.add('provisional');    // Yellow - still provisional
                            }
                        } else if (isTryingCell || isAssignedDuringBacktrack) {
                            // For trying cell or assigned cells during backtracking,
                            // DON'T cross off based on domain - just shrink untried values
                            dv.classList.add('shrunk');
                        } else if (!domain.includes(d)) {
                            // Pruned by AC-3 or constraint propagation (for unassigned cells)
                            dv.classList.add('pruned');
                        }

                        domainGrid.appendChild(dv);
                    }
                    cellEl.appendChild(domainGrid);
                }

                // Highlight based on step type
                if (metadata.cell === cell) {
                    if (stepType === StepType.ASSIGN || stepType === StepType.TRY_VALUE) {
                        cellEl.classList.add('trying-cell');
                    } else if (stepType === StepType.AC3_CHECK) {
                        cellEl.classList.add('arc-source');
                    } else if (stepType === StepType.BACKTRACK) {
                        // Keep the backtracked cell blue (it's still the active cell)
                        cellEl.classList.add('trying-cell');
                    } else if (stepType === StepType.AC3_PRUNE) {
                        cellEl.classList.add('arc-source');
                    }
                }

                // Highlight conflict cells in red during BACKTRACK
                if (conflictCellSet.has(cell)) {
                    cellEl.classList.add('conflict-cell');
                }

                // Highlight provisional cells (assigned but not the active trying cell)
                if (isAssignedDuringBacktrack && !isTryingCell && metadata.cell !== cell && !isSolved) {
                    cellEl.classList.add('provisional-cell');
                }

                if (metadata.peer === cell) {
                    if (stepType === StepType.AC3_CHECK || stepType === StepType.AC3_PRUNE) {
                        cellEl.classList.add('arc-target');
                    } else {
                        cellEl.classList.add('examining');
                    }
                }
            });

            // Draw arc line and highlight values for AC3_CHECK and AC3_PRUNE steps
            if ((stepType === StepType.AC3_CHECK || stepType === StepType.AC3_PRUNE) && metadata.cell && metadata.peer) {
                // Draw the arc line between cells
                this.drawArcLine(metadata.cell, metadata.peer);

                // Get domains for both cells
                const sourceDomain = snapshot[metadata.cell];
                const targetDomain = snapshot[metadata.peer];

                // For AC3_CHECK: highlight all values being compared
                if (stepType === StepType.AC3_CHECK) {
                    // Highlight source cell's domain values as "checking"
                    this.highlightDomainValues(metadata.cell, sourceDomain, 'checking');

                    // Highlight target cell's domain values as "support"
                    this.highlightDomainValues(metadata.peer, targetDomain, 'support');
                }

                // For AC3_PRUNE: highlight the specific value being pruned
                if (stepType === StepType.AC3_PRUNE && metadata.valueToPrune) {
                    // The value to be pruned is still in the domain at this step
                    // Highlight it with "no-support" to show it's about to be removed
                    this.highlightDomainValues(metadata.cell, [metadata.valueToPrune], 'no-support', false);

                    // Show what values in target cell exist (they all equal the pruned value, hence no support)
                    this.highlightDomainValues(metadata.peer, targetDomain, 'support');
                }
            }
        }

        /**
         * Render state for comparison view - shows full domains with failed attempts crossed off
         */
        renderComparisonState(snapshot, assignedValues, stepType) {
            // Clear previous arc visualization
            this.clearArcLine();
            this.clearDomainHighlights();

            CELLS.forEach(cell => {
                const cellEl = this.cells[cell];
                const domain = snapshot[cell];
                const assignedValue = assignedValues[cell];

                // Reset state classes
                cellEl.classList.remove('assigned', 'examining', 'conflict', 'arc-source', 'arc-target', 'trying-cell', 'provisional-cell', 'conflict-cell');

                if (this.fixedCells.has(cell)) {
                    // Fixed cells don't change
                    return;
                }

                // Always show domain grid for comparison view
                cellEl.innerHTML = '';
                const domainGrid = document.createElement('div');
                domainGrid.className = 'domain-grid';

                // Get failed attempts for this cell
                const cellFailedAttempts = this.failedAttempts[cell] || new Set();

                for (let d = 1; d <= 4; d++) {
                    const dv = document.createElement('div');
                    dv.className = 'domain-value';
                    dv.dataset.value = d;
                    dv.textContent = d;

                    const isFailedAttempt = cellFailedAttempts.has(d);
                    const isAssignedValue = assignedValue === d;
                    const isPruned = !domain.includes(d);

                    if (isFailedAttempt) {
                        // Value was tried and failed during backtracking
                        dv.classList.add('failed-attempt');
                    } else if (isPruned) {
                        // Value was pruned (by AC-3 or constraint propagation)
                        dv.classList.add('pruned');
                    } else if (isAssignedValue) {
                        // This is the assigned value - highlight it
                        dv.classList.add('assigned-mini');
                    }

                    domainGrid.appendChild(dv);
                }

                cellEl.appendChild(domainGrid);

                // Add cell-level styling for assigned cells
                if (assignedValue) {
                    cellEl.classList.add('assigned');
                }
            });
        }

        updateMetrics(metrics) {
            document.getElementById('metric-assignments').textContent = metrics.assignments;
            document.getElementById('metric-backtracks').textContent = metrics.backtracks;
            document.getElementById('metric-arcs').textContent = metrics.arcChecks;
            document.getElementById('metric-prunings').textContent = metrics.prunings;
        }

        resetMetrics() {
            document.getElementById('metric-assignments').textContent = '0';
            document.getElementById('metric-backtracks').textContent = '0';
            document.getElementById('metric-arcs').textContent = '0';
            document.getElementById('metric-prunings').textContent = '0';
            document.getElementById('metric-time').textContent = '0.00s';
        }

        updateElapsedTime(seconds) {
            document.getElementById('metric-time').textContent = seconds.toFixed(2) + 's';
        }

        addLogEntry(stepType, metadata, snapshot = null) {
            const logEl = document.getElementById('execution-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';

            let icon = 'fa-circle';
            let message = '';
            let logClass = 'log-info';
            let showGrid = false;
            let highlightCell = null;
            let highlightType = 'exploring'; // 'exploring', 'success', 'failed'
            let conflictCells = [];

            switch (stepType) {
                case StepType.INIT:
                    icon = 'fa-play';
                    message = metadata.message || 'Initialized puzzle state';
                    logClass = 'log-info';
                    showGrid = true;
                    break;
                case StepType.AC3_START:
                    icon = 'fa-filter';
                    message = `AC-3 starting with ${metadata.queueSize} arcs in queue`;
                    logClass = 'log-ac3';
                    break;
                case StepType.AC3_CHECK:
                    icon = 'fa-exchange';
                    message = `AC-3: Checking arc (${metadata.cell}, ${metadata.peer})`;
                    logClass = 'log-ac3';
                    break;
                case StepType.AC3_PRUNE:
                    icon = 'fa-cut';
                    message = `AC-3: Pruned ${metadata.value} from ${metadata.cell} (${metadata.reason})`;
                    logClass = 'log-prune';
                    showGrid = true;
                    highlightCell = metadata.cell;
                    highlightType = 'failed';
                    break;
                case StepType.AC3_DONE:
                    icon = metadata.success ? 'fa-check' : 'fa-times';
                    message = metadata.success ? 'AC-3 completed successfully' : `AC-3 failed: ${metadata.reason}`;
                    logClass = metadata.success ? 'log-info' : 'log-failure';
                    showGrid = true;
                    break;
                case StepType.FIXED_PRUNE:
                    icon = 'fa-minus-circle';
                    message = `Removed ${metadata.value} from ${metadata.cell} (clue: ${metadata.fixedCell}=${metadata.value})`;
                    logClass = 'log-prune';
                    break;
                case StepType.TRY_VALUE:
                    icon = 'fa-question';
                    message = `Trying ${metadata.cell} = ${metadata.value} (domain: {${metadata.domain.join(',')}})`;
                    logClass = 'log-info';
                    break;
                case StepType.ASSIGN:
                    icon = 'fa-pencil';
                    message = `Assigned ${metadata.cell} = ${metadata.value}`;
                    logClass = 'log-assign';
                    showGrid = true;
                    highlightCell = metadata.cell;
                    highlightType = 'exploring';
                    break;
                case StepType.BACKTRACK:
                    icon = 'fa-undo';
                    message = `Backtracking from ${metadata.cell} = ${metadata.value}`;
                    if (metadata.reason) {
                        message += ` (${metadata.reason})`;
                    }
                    logClass = 'log-backtrack';
                    showGrid = true;
                    highlightCell = metadata.cell;
                    highlightType = 'exploring'; // Show attempted cell in yellow
                    // Extract conflict cells from reason if present
                    if (metadata.reason && metadata.reason.startsWith('Conflicts with')) {
                        const conflictMatch = metadata.reason.match(/Conflicts with (.+)/);
                        if (conflictMatch) {
                            conflictCells = conflictMatch[1].split(', ').map(c => c.split('=')[0]);
                        }
                    }
                    break;
                case StepType.MAC_CHECK:
                    icon = 'fa-refresh';
                    message = `MAC: Running AC-3 on ${metadata.arcsToCheck} arcs after assigning ${metadata.cell}`;
                    logClass = 'log-ac3';
                    break;
                case StepType.SOLVED:
                    icon = 'fa-trophy';
                    message = 'Puzzle solved!';
                    logClass = 'log-success';
                    showGrid = true;
                    highlightType = 'success';
                    break;
                case StepType.UNSOLVABLE:
                    icon = 'fa-ban';
                    message = `No solution: ${metadata.reason}`;
                    logClass = 'log-failure';
                    break;
            }

            entry.classList.add(logClass);

            // Build entry HTML with optional mini-grid
            let gridHtml = '';
            if (showGrid && snapshot) {
                gridHtml = this.renderLogMiniGrid(snapshot, highlightCell, highlightType, conflictCells);
            }

            entry.innerHTML = `
                <div class="log-entry-content">
                    <span class="log-icon"><i class="fa ${icon}"></i></span>
                    <span class="log-message">${message}</span>
                </div>
                ${gridHtml}
            `;

            logEl.appendChild(entry);
            logEl.scrollTop = logEl.scrollHeight;
        }

        /**
         * Render a mini-grid for the execution log
         */
        renderLogMiniGrid(snapshot, highlightCell, highlightType, conflictCells = []) {
            const cellSize = 18;
            const gridSize = cellSize * N;

            let html = `<div class="log-mini-grid">`;

            for (let r = 0; r < N; r++) {
                for (let c = 0; c < N; c++) {
                    const cellName = ROWS[r] + COLS[c];
                    const domain = snapshot[cellName];
                    const isAssigned = domain && domain.length === 1;
                    const value = isAssigned ? domain[0] : '';

                    let cellClass = 'log-grid-cell';

                    // Determine highlighting
                    if (cellName === highlightCell) {
                        if (highlightType === 'success') {
                            cellClass += ' log-grid-cell-success';
                        } else {
                            cellClass += ' log-grid-cell-exploring';
                        }
                    } else if (conflictCells.includes(cellName)) {
                        cellClass += ' log-grid-cell-failed';
                    }

                    // Add box border classes
                    if (c === 1) cellClass += ' log-grid-cell-box-right';
                    if (r === 1) cellClass += ' log-grid-cell-box-bottom';

                    html += `<div class="${cellClass}">${value}</div>`;
                }
            }

            html += `</div>`;
            return html;
        }

        clearLog() {
            const logEl = document.getElementById('execution-log');
            logEl.innerHTML = '';
        }

        setPlayingState(playing) {
            this.solving = playing;

            const solveBtn = document.getElementById('btn-solve');
            const pauseBtn = document.getElementById('btn-pause');
            const stepBtn = document.getElementById('btn-step');

            if (playing) {
                solveBtn.disabled = true;
                pauseBtn.disabled = false;
                stepBtn.disabled = true;
                pauseBtn.innerHTML = '<i class="fa fa-pause"></i> Pause';

                // Mark all cells as solving
                CELLS.forEach(cell => {
                    this.cells[cell].classList.add('solving');
                });
            } else {
                solveBtn.disabled = false;
                pauseBtn.disabled = false;
                stepBtn.disabled = false;
                pauseBtn.innerHTML = '<i class="fa fa-play"></i> Resume';

                // Remove solving class
                CELLS.forEach(cell => {
                    this.cells[cell].classList.remove('solving');
                });
            }
        }

        setFinishedState() {
            document.getElementById('btn-solve').disabled = false;
            document.getElementById('btn-pause').disabled = true;
            document.getElementById('btn-step').disabled = true;

            // Clear arc visualization
            this.clearArcLine();
            this.clearDomainHighlights();

            CELLS.forEach(cell => {
                this.cells[cell].classList.remove('solving', 'arc-source', 'arc-target', 'trying-cell', 'provisional-cell', 'conflict-cell');
            });
        }

        getPuzzle() {
            return this.puzzle;
        }
    }

    /**
     * Main Application Controller
     */
    class ShidokuApp {
        constructor() {
            this.ui = new UIController();
            this.solver = new ShidokuSolver();
            this.treeViz = new SearchTreeVisualizer('search-tree-container', 'search-tree-svg');
            this.playback = new PlaybackController(this.ui, this.treeViz);

            this.initControls();

            // Listen for theme changes
            document.addEventListener('themechange', () => {
                // CSS variables handle the theming automatically
                // Re-render tree with new colors
                if (this.treeViz.nodes.length > 0) {
                    this.treeViz.render();
                }
            });

            // Run comparison whenever puzzle changes
            this.ui.onPuzzleChange = () => this.runComparison();

            // Load a sample puzzle by default (this triggers the comparison)
            this.ui.loadPuzzle(SAMPLE_PUZZLES.easy);
        }

        initControls() {
            const solveBtn = document.getElementById('btn-solve');
            const pauseBtn = document.getElementById('btn-pause');
            const stepBtn = document.getElementById('btn-step');
            const resetBtn = document.getElementById('btn-reset');
            const speedSlider = document.getElementById('speed-slider');

            solveBtn.addEventListener('click', () => this.solve());
            pauseBtn.addEventListener('click', () => this.togglePause());
            stepBtn.addEventListener('click', () => this.step());
            resetBtn.addEventListener('click', () => this.reset());

            speedSlider.addEventListener('input', (e) => {
                this.playback.setSpeed(parseInt(e.target.value));
            });
        }

        solve() {
            if (this.playback.isPlaying()) return;

            // Apply any user selections as fixed cells before solving
            this.ui.applyUserSelections();

            const puzzle = this.ui.getPuzzle();
            const useAC3 = document.getElementById('use-ac3').checked;
            const useMRV = document.getElementById('use-mrv').checked;
            const useFC = document.getElementById('use-fc').checked;
            const useMAC = document.getElementById('use-mac').checked;

            this.ui.clearLog();
            this.ui.resetMetrics();
            this.treeViz.clear();

            const result = this.solver.solve(puzzle, useAC3, useMRV, useFC, useMAC);

            this.playback.load(result.steps);
            this.playback.play();
        }

        togglePause() {
            if (this.playback.isPlaying()) {
                this.playback.pause();
            } else if (this.playback.isPaused()) {
                this.playback.resume();
            }
        }

        step() {
            if (this.playback.isPlaying()) return;

            if (this.playback.steps.length === 0) {
                // Apply any user selections as fixed cells before solving
                this.ui.applyUserSelections();

                // Run solver first
                const puzzle = this.ui.getPuzzle();
                const useAC3 = document.getElementById('use-ac3').checked;
                const useMRV = document.getElementById('use-mrv').checked;
                const useFC = document.getElementById('use-fc').checked;
                const useMAC = document.getElementById('use-mac').checked;

                this.ui.clearLog();
                this.ui.resetMetrics();
                this.treeViz.clear();

                const result = this.solver.solve(puzzle, useAC3, useMRV, useFC, useMAC);
                this.playback.load(result.steps);
            }

            this.playback.step();
        }

        reset() {
            this.playback.stop();
            this.treeViz.clear();
            this.ui.loadPuzzle(this.ui.getPuzzle());
        }

        runComparison() {
            if (this.playback.isPlaying()) return;

            // Apply any user selections first
            this.ui.applyUserSelections();
            const puzzle = this.ui.getPuzzle();

            const results = [];

            for (const config of ALGORITHM_CONFIGS) {
                const solver = new ShidokuSolver();
                const startTime = performance.now();
                let result;

                if (config.ac3Only) {
                    // Special case: run AC-3 only without backtracking
                    result = solver.solveAC3Only(puzzle);
                } else {
                    result = solver.solve(puzzle, config.ac3, config.mrv, config.fc, config.mac);
                }
                const endTime = performance.now();

                // Determine which metrics are applicable for this algorithm
                const hasArcChecks = config.ac3Only || config.ac3 || config.mac;
                const hasSearch = !config.ac3Only;

                // Get the final snapshot from the last step
                const finalSnapshot = result.steps.length > 0
                    ? result.steps[result.steps.length - 1].snapshot
                    : null;

                // Build failed attempts map by processing BACKTRACK and AC3_PRUNE steps
                // Both represent values that were tested and found invalid
                const failedAttempts = {};
                for (const step of result.steps) {
                    // BACKTRACK: value was tried during search and caused conflict
                    // AC3_PRUNE: value was removed by AC-3 because it had no support
                    if ((step.type === StepType.BACKTRACK || step.type === StepType.AC3_PRUNE)
                        && step.metadata.cell && step.metadata.value) {
                        if (!failedAttempts[step.metadata.cell]) {
                            failedAttempts[step.metadata.cell] = new Set();
                        }
                        failedAttempts[step.metadata.cell].add(step.metadata.value);
                    }
                }

                results.push({
                    name: config.name,
                    success: result.success,
                    assignments: result.metrics.assignments,
                    backtracks: result.metrics.backtracks,
                    arcChecks: result.metrics.arcChecks,
                    prunings: result.metrics.prunings,
                    time: (endTime - startTime).toFixed(2),
                    // Flags for which metrics apply
                    hasArcChecks,
                    hasSearch,
                    // Store final state for visualization
                    finalSnapshot,
                    failedAttempts,
                    // Store config for setting checkboxes
                    config
                });
            }

            // Store results for click handler access
            this.comparisonResults = results;
            this.displayComparison(results);
        }

        displayComparison(results) {
            const tbody = document.querySelector('#comparison-table tbody');
            tbody.innerHTML = '';

            // Find best (minimum) values for each metric (only from successful results)
            const successfulResults = results.filter(r => r.success);
            const best = {
                assignments: Math.min(...successfulResults.filter(r => r.hasSearch).map(r => r.assignments)),
                backtracks: Math.min(...successfulResults.filter(r => r.hasSearch).map(r => r.backtracks)),
                arcChecks: Math.min(...successfulResults.filter(r => r.hasArcChecks && r.arcChecks > 0).map(r => r.arcChecks)) || 0,
                prunings: Math.min(...successfulResults.filter(r => r.prunings > 0).map(r => r.prunings)) || 0,
                time: Math.min(...successfulResults.map(r => parseFloat(r.time)))
            };

            results.forEach((r, index) => {
                const row = document.createElement('tr');

                if (!r.success) {
                    // Algorithm couldn't find solution - show message spanning all metric columns
                    row.innerHTML = `
                        <td class="algorithm-name" data-index="${index}">${r.name}</td>
                        <td colspan="5" class="no-solution-value">Failed to Find Solution</td>
                    `;
                } else {
                    const isBest = (val, metric) => val === best[metric] ? 'best-value' : '';

                    // Determine cell content and classes
                    const assignsContent = r.hasSearch ? r.assignments : 'N/A';
                    const assignsClass = r.hasSearch ? isBest(r.assignments, 'assignments') : 'na-value';

                    const backtracksContent = r.hasSearch ? r.backtracks : 'N/A';
                    const backtracksClass = r.hasSearch ? isBest(r.backtracks, 'backtracks') : 'na-value';

                    const arcChecksContent = r.hasArcChecks ? r.arcChecks : 'N/A';
                    const arcChecksClass = r.hasArcChecks ? (r.arcChecks > 0 && isBest(r.arcChecks, 'arcChecks') ? 'best-value' : '') : 'na-value';

                    const pruningsContent = r.hasArcChecks ? r.prunings : 'N/A';
                    const pruningsClass = r.hasArcChecks ? (r.prunings > 0 && isBest(r.prunings, 'prunings') ? 'best-value' : '') : 'na-value';

                    row.innerHTML = `
                        <td class="algorithm-name" data-index="${index}">${r.name}</td>
                        <td class="${assignsClass}">${assignsContent}</td>
                        <td class="${backtracksClass}">${backtracksContent}</td>
                        <td class="${arcChecksClass}">${arcChecksContent}</td>
                        <td class="${pruningsClass}">${pruningsContent}</td>
                        <td class="${isBest(parseFloat(r.time), 'time')}">${r.time}</td>
                    `;
                }
                tbody.appendChild(row);
            });

            // Add click handlers to algorithm names
            tbody.querySelectorAll('.algorithm-name').forEach(cell => {
                cell.addEventListener('click', () => {
                    const index = parseInt(cell.dataset.index);
                    this.showComparisonResult(index);
                });
            });
        }

        showComparisonResult(index) {
            if (!this.comparisonResults || !this.comparisonResults[index]) return;

            const result = this.comparisonResults[index];
            if (!result.finalSnapshot) return;

            // Stop any running playback
            this.playback.stop();

            // Clear tree visualization
            this.treeViz.clear();

            // Set up the failed attempts from this algorithm's run
            this.ui.failedAttempts = {};
            if (result.failedAttempts) {
                for (const cell in result.failedAttempts) {
                    this.ui.failedAttempts[cell] = new Set(result.failedAttempts[cell]);
                }
            }

            // Create a modified snapshot for comparison view:
            // - Fixed cells keep their single value
            // - Other cells show full domain [1,2,3,4] so we can display failed attempts
            // - We'll use metadata to indicate the assigned value
            const comparisonSnapshot = {};
            const assignedValues = {};

            for (const cell of CELLS) {
                const domain = result.finalSnapshot[cell];
                if (this.ui.fixedCells.has(cell)) {
                    // Fixed cells stay as-is
                    comparisonSnapshot[cell] = domain;
                } else if (domain.length === 1) {
                    // Cell was assigned - expand to full domain but track assigned value
                    comparisonSnapshot[cell] = [1, 2, 3, 4];
                    assignedValues[cell] = domain[0];
                } else {
                    // Cell wasn't fully assigned (e.g., AC-3 only case)
                    comparisonSnapshot[cell] = domain;
                }
            }

            // Render using comparison mode
            const stepType = result.success ? StepType.SOLVED : StepType.INIT;
            this.ui.renderComparisonState(comparisonSnapshot, assignedValues, stepType);

            // Update metrics display
            document.getElementById('metric-assignments').textContent = result.assignments;
            document.getElementById('metric-backtracks').textContent = result.backtracks;
            document.getElementById('metric-arcs').textContent = result.arcChecks;
            document.getElementById('metric-prunings').textContent = result.prunings;
            document.getElementById('metric-time').textContent = result.time + 'ms';

            // Highlight the selected row
            document.querySelectorAll('#comparison-table tbody tr').forEach((row, i) => {
                row.classList.toggle('selected-row', i === index);
            });

            // Set the algorithm checkboxes to match this configuration
            if (result.config) {
                const config = result.config;
                // For AC-3 Only, set AC-3 checkbox but it won't run search anyway
                document.getElementById('use-ac3').checked = config.ac3Only || config.ac3 || false;
                document.getElementById('use-mrv').checked = config.mrv || false;
                document.getElementById('use-fc').checked = config.fc || false;
                document.getElementById('use-mac').checked = config.mac || false;
            }
        }
    }

    // Initialize the application when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => new ShidokuApp());
    } else {
        new ShidokuApp();
    }
})();
