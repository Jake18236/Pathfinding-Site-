/**
 * A* Pathfinding Visualizer
 * Interactive visualization of the A* search algorithm
 */

(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================

    const GRID_WIDTH = 20;
    const GRID_HEIGHT = 15;
    const CELL_EMPTY = 0;
    const CELL_WALL = 1;
    const CELL_START = 2;
    const CELL_GOAL = 3;

    // Step types for playback
    const STEP_INIT = 'INIT';
    const STEP_EXPAND = 'EXPAND';
    const STEP_NEIGHBOR_CHECK = 'NEIGHBOR_CHECK';
    const STEP_ADD_TO_OPEN = 'ADD_TO_OPEN';
    const STEP_UPDATE_PATH = 'UPDATE_PATH';
    const STEP_GOAL_FOUND = 'GOAL_FOUND';
    const STEP_NO_SOLUTION = 'NO_SOLUTION';

    // Sample puzzles
    const SAMPLE_PUZZLES = {
        easy: {
            walls: [[5,3],[5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11]],
            start: [2, 7],
            goal: [17, 7]
        },
        medium: {
            walls: [
                [4,2],[4,3],[4,4],[4,5],[4,6],
                [8,8],[8,9],[8,10],[8,11],[8,12],
                [12,3],[12,4],[12,5],[12,6],[12,7],
                [15,7],[15,8],[15,9],[15,10],[15,11]
            ],
            start: [1, 7],
            goal: [18, 7]
        },
        hard: {
            walls: [
                // Vertical barriers
                [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                [7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13],
                [11,0],[11,1],[11,2],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8],[11,9],
                [15,5],[15,6],[15,7],[15,8],[15,9],[15,10],[15,11],[15,12],[15,13],[15,14]
            ],
            start: [1, 7],
            goal: [18, 7]
        },
        maze: {
            walls: [
                // Outer walls with gaps
                [2,0],[2,1],[2,2],[2,3],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],[2,11],[2,12],[2,13],[2,14],
                [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],
                [6,0],[6,1],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,12],[6,13],[6,14],
                [8,1],[8,2],[8,3],[8,4],[8,6],[8,7],[8,8],[8,9],[8,10],[8,11],[8,12],[8,13],
                [10,0],[10,1],[10,2],[10,4],[10,5],[10,6],[10,7],[10,8],[10,10],[10,11],[10,12],[10,13],[10,14],
                [12,1],[12,2],[12,3],[12,4],[12,5],[12,7],[12,8],[12,9],[12,10],[12,11],[12,12],[12,13],
                [14,0],[14,1],[14,2],[14,3],[14,5],[14,6],[14,7],[14,8],[14,9],[14,11],[14,12],[14,13],[14,14],
                [16,1],[16,2],[16,3],[16,4],[16,5],[16,6],[16,8],[16,9],[16,10],[16,11],[16,13]
            ],
            start: [0, 7],
            goal: [19, 7]
        },
        nosolution: {
            walls: [
                // Complete wall blocking the goal
                [14,0],[14,1],[14,2],[14,3],[14,4],[14,5],[14,6],[14,7],[14,8],[14,9],[14,10],[14,11],[14,12],[14,13],[14,14]
            ],
            start: [2, 7],
            goal: [17, 7]
        }
    };

    // ============================================
    // Priority Queue (Sorted Array - simpler, correct implementation)
    // ============================================

    class PriorityQueue {
        constructor(compareFn) {
            this.items = [];
            this.compare = compareFn || ((a, b) => a.f - b.f);
            this.nodeMap = new Map(); // key -> item for quick lookup
        }

        push(item) {
            const key = this._key(item);
            this.nodeMap.set(key, item);
            this.items.push(item);
            // Keep sorted by inserting in correct position
            this._sort();
        }

        pop() {
            if (this.items.length === 0) return null;
            const min = this.items.shift(); // Remove first (smallest) element
            this.nodeMap.delete(this._key(min));
            return min;
        }

        contains(x, y) {
            return this.nodeMap.has(`${x},${y}`);
        }

        get(x, y) {
            return this.nodeMap.get(`${x},${y}`) || null;
        }

        updatePriority(item) {
            const key = this._key(item);
            if (!this.nodeMap.has(key)) return;

            // Update in map
            this.nodeMap.set(key, item);

            // Find and update in array
            const idx = this.items.findIndex(n => this._key(n) === key);
            if (idx !== -1) {
                this.items[idx] = item;
                this._sort();
            }
        }

        isEmpty() {
            return this.items.length === 0;
        }

        size() {
            return this.items.length;
        }

        toArray() {
            return [...this.items]; // Already sorted
        }

        _key(item) {
            return `${item.x},${item.y}`;
        }

        _sort() {
            this.items.sort(this.compare);
        }
    }

    // ============================================
    // Grid State
    // ============================================

    class GridState {
        constructor(width = GRID_WIDTH, height = GRID_HEIGHT) {
            this.width = width;
            this.height = height;
            this.cells = [];
            this.start = null;
            this.goal = null;
            this.clear();
        }

        clear() {
            this.cells = [];
            for (let y = 0; y < this.height; y++) {
                this.cells[y] = [];
                for (let x = 0; x < this.width; x++) {
                    this.cells[y][x] = CELL_EMPTY;
                }
            }
            this.start = null;
            this.goal = null;
        }

        setCell(x, y, type) {
            if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;

            // Handle special cells
            if (type === CELL_START) {
                if (this.start) {
                    this.cells[this.start.y][this.start.x] = CELL_EMPTY;
                }
                this.start = { x, y };
            } else if (type === CELL_GOAL) {
                if (this.goal) {
                    this.cells[this.goal.y][this.goal.x] = CELL_EMPTY;
                }
                this.goal = { x, y };
            }

            // Clear old start/goal if overwriting
            if (this.start && this.start.x === x && this.start.y === y && type !== CELL_START) {
                this.start = null;
            }
            if (this.goal && this.goal.x === x && this.goal.y === y && type !== CELL_GOAL) {
                this.goal = null;
            }

            this.cells[y][x] = type;
        }

        getCell(x, y) {
            if (x < 0 || x >= this.width || y < 0 || y >= this.height) return CELL_WALL;
            return this.cells[y][x];
        }

        isWalkable(x, y) {
            return this.getCell(x, y) !== CELL_WALL;
        }

        getNeighbors(x, y, allowDiagonal = false) {
            const neighbors = [];
            const dirs = [
                { dx: 0, dy: -1, cost: 1 },  // up
                { dx: 1, dy: 0, cost: 1 },   // right
                { dx: 0, dy: 1, cost: 1 },   // down
                { dx: -1, dy: 0, cost: 1 }   // left
            ];

            if (allowDiagonal) {
                dirs.push(
                    { dx: 1, dy: -1, cost: Math.SQRT2 },  // up-right
                    { dx: 1, dy: 1, cost: Math.SQRT2 },   // down-right
                    { dx: -1, dy: 1, cost: Math.SQRT2 }, // down-left
                    { dx: -1, dy: -1, cost: Math.SQRT2 } // up-left
                );
            }

            for (const dir of dirs) {
                const nx = x + dir.dx;
                const ny = y + dir.dy;
                if (this.isWalkable(nx, ny)) {
                    // For diagonal, check that we can actually cut the corner
                    if (Math.abs(dir.dx) + Math.abs(dir.dy) === 2) {
                        if (!this.isWalkable(x + dir.dx, y) || !this.isWalkable(x, y + dir.dy)) {
                            continue; // Can't cut corner
                        }
                    }
                    neighbors.push({ x: nx, y: ny, cost: dir.cost });
                }
            }

            return neighbors;
        }

        clone() {
            const copy = new GridState(this.width, this.height);
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    copy.cells[y][x] = this.cells[y][x];
                }
            }
            copy.start = this.start ? { ...this.start } : null;
            copy.goal = this.goal ? { ...this.goal } : null;
            return copy;
        }

        loadSample(name) {
            this.clear();
            const sample = SAMPLE_PUZZLES[name];
            if (!sample) return;

            for (const [x, y] of sample.walls) {
                this.setCell(x, y, CELL_WALL);
            }
            this.setCell(sample.start[0], sample.start[1], CELL_START);
            this.setCell(sample.goal[0], sample.goal[1], CELL_GOAL);
        }
    }

    // ============================================
    // A* Solver
    // ============================================

    class AStarSolver {
        constructor() {
            this.heuristics = {
                manhattan: (x1, y1, x2, y2) => Math.abs(x1 - x2) + Math.abs(y1 - y2),
                euclidean: (x1, y1, x2, y2) => Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                chebyshev: (x1, y1, x2, y2) => Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2)),
                dijkstra: () => 0
            };
        }

        solve(grid, options = {}) {
            const {
                heuristic = 'manhattan',
                allowDiagonal = false,
                tiebreaker = 'none',
                weight = 1.0
            } = options;

            const hFunc = this.heuristics[heuristic] || this.heuristics.manhattan;
            const startTime = performance.now();
            const steps = [];
            const metrics = {
                nodesExpanded: 0,
                nodesGenerated: 0,
                pathLength: 0,
                pathCost: 0,
                time: 0
            };

            if (!grid.start || !grid.goal) {
                return { success: false, path: [], steps, metrics, error: 'Missing start or goal' };
            }

            const { x: sx, y: sy } = grid.start;
            const { x: gx, y: gy } = grid.goal;

            // Create comparator with tie-breaker
            const compareFn = (a, b) => {
                if (a.f !== b.f) return a.f - b.f;
                if (tiebreaker === 'highg') return b.g - a.g; // Prefer higher g
                if (tiebreaker === 'lowg') return a.g - b.g;  // Prefer lower g
                return 0;
            };

            const openSet = new PriorityQueue(compareFn);
            const closedSet = new Map();
            const cameFrom = new Map();
            const gScore = new Map();
            const fScore = new Map();

            const key = (x, y) => `${x},${y}`;

            // Initialize start node
            const startH = hFunc(sx, sy, gx, gy) * weight;
            gScore.set(key(sx, sy), 0);
            fScore.set(key(sx, sy), startH);
            openSet.push({ x: sx, y: sy, g: 0, h: startH, f: startH });
            metrics.nodesGenerated++;

            // Record initial state
            steps.push({
                type: STEP_INIT,
                openList: [{ x: sx, y: sy, g: 0, h: startH, f: startH }],
                closedList: [],
                current: null,
                message: `Starting A* from (${sx},${sy}) to (${gx},${gy})`,
                metrics: { nodesExpanded: 0, nodesGenerated: 1, pathLength: null, pathCost: null }
            });

            while (!openSet.isEmpty()) {
                const current = openSet.pop();
                const { x: cx, y: cy, g: cg, h: ch, f: cf } = current;
                const currentKey = key(cx, cy);

                // Skip if already in closed set (might have duplicate entries)
                if (closedSet.has(currentKey)) continue;

                // Add to closed set
                closedSet.set(currentKey, { x: cx, y: cy, g: cg, h: ch, f: cf });
                metrics.nodesExpanded++;

                // Record expansion
                steps.push({
                    type: STEP_EXPAND,
                    current: { x: cx, y: cy, g: cg, h: ch, f: cf },
                    openList: openSet.toArray().map(n => ({ x: n.x, y: n.y, g: n.g, h: n.h, f: n.f })),
                    closedList: Array.from(closedSet.values()),
                    message: `Expanding (${cx},${cy}) with f=${cf.toFixed(2)}, g=${cg.toFixed(2)}, h=${ch.toFixed(2)}`,
                    metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: null, pathCost: null }
                });

                // Check if goal reached
                if (cx === gx && cy === gy) {
                    const path = this._reconstructPath(cameFrom, cx, cy);
                    metrics.pathLength = path.length;
                    metrics.pathCost = cg;
                    metrics.time = performance.now() - startTime;

                    steps.push({
                        type: STEP_GOAL_FOUND,
                        path: path,
                        openList: openSet.toArray().map(n => ({ x: n.x, y: n.y, g: n.g, h: n.h, f: n.f })),
                        closedList: Array.from(closedSet.values()),
                        message: `Goal found! Path length: ${path.length}, cost: ${cg.toFixed(2)}`,
                        metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: path.length, pathCost: cg }
                    });

                    return { success: true, path, steps, metrics };
                }

                // Explore neighbors
                const neighbors = grid.getNeighbors(cx, cy, allowDiagonal);
                for (const neighbor of neighbors) {
                    const { x: nx, y: ny, cost } = neighbor;
                    const neighborKey = key(nx, ny);

                    // Skip if in closed set
                    if (closedSet.has(neighborKey)) continue;

                    const tentativeG = cg + cost;
                    const existingG = gScore.get(neighborKey);

                    // Record neighbor check
                    steps.push({
                        type: STEP_NEIGHBOR_CHECK,
                        current: { x: cx, y: cy, g: cg, h: ch, f: cf },
                        neighbor: { x: nx, y: ny },
                        tentativeG: tentativeG,
                        existingG: existingG !== undefined ? existingG : null,
                        openList: openSet.toArray().map(n => ({ x: n.x, y: n.y, g: n.g, h: n.h, f: n.f })),
                        closedList: Array.from(closedSet.values()),
                        message: `Checking neighbor (${nx},${ny}): tentative g=${tentativeG.toFixed(2)}`,
                        metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: null, pathCost: null }
                    });

                    if (existingG === undefined || tentativeG < existingG) {
                        // This path is better
                        const nh = hFunc(nx, ny, gx, gy) * weight;
                        const nf = tentativeG + nh;

                        cameFrom.set(neighborKey, { x: cx, y: cy });
                        gScore.set(neighborKey, tentativeG);
                        fScore.set(neighborKey, nf);

                        const newNode = { x: nx, y: ny, g: tentativeG, h: nh, f: nf };

                        if (existingG !== undefined) {
                            // Update existing node
                            openSet.updatePriority(newNode);
                            steps.push({
                                type: STEP_UPDATE_PATH,
                                node: newNode,
                                openList: openSet.toArray().map(n => ({ x: n.x, y: n.y, g: n.g, h: n.h, f: n.f })),
                                closedList: Array.from(closedSet.values()),
                                message: `Updated (${nx},${ny}): new g=${tentativeG.toFixed(2)}, f=${nf.toFixed(2)}`,
                                metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: null, pathCost: null }
                            });
                        } else {
                            // Add new node
                            openSet.push(newNode);
                            metrics.nodesGenerated++;
                            steps.push({
                                type: STEP_ADD_TO_OPEN,
                                node: newNode,
                                openList: openSet.toArray().map(n => ({ x: n.x, y: n.y, g: n.g, h: n.h, f: n.f })),
                                closedList: Array.from(closedSet.values()),
                                message: `Added (${nx},${ny}) to open: f=${nf.toFixed(2)}, g=${tentativeG.toFixed(2)}, h=${nh.toFixed(2)}`,
                                metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: null, pathCost: null }
                            });
                        }
                    }
                }
            }

            // No solution found
            metrics.time = performance.now() - startTime;
            steps.push({
                type: STEP_NO_SOLUTION,
                openList: [],
                closedList: Array.from(closedSet.values()),
                message: 'No path found - goal is unreachable',
                metrics: { nodesExpanded: metrics.nodesExpanded, nodesGenerated: metrics.nodesGenerated, pathLength: null, pathCost: null }
            });

            return { success: false, path: [], steps, metrics };
        }

        _reconstructPath(cameFrom, x, y) {
            const path = [{ x, y }];
            let current = `${x},${y}`;

            while (cameFrom.has(current)) {
                const prev = cameFrom.get(current);
                path.unshift(prev);
                current = `${prev.x},${prev.y}`;
            }

            return path;
        }
    }

    // ============================================
    // UI Controller
    // ============================================

    class UIController {
        constructor(grid) {
            this.grid = grid;
            this.editMode = 'wall';
            this.isDrawing = false;
            this.showValues = false;
            this.nodeValues = new Map(); // Store f/g/h values for cells
            this.currentHeuristic = 'manhattan';
            this.cellSize = 28; // Default cell size, updated on init
            this.gridGap = 1;
            this.initGrid();
            this.initEventListeners();
        }

        initGrid() {
            const container = document.getElementById('pathfinding-grid');
            if (!container) return;

            container.innerHTML = '';

            // Add corner spacer
            const corner = document.createElement('div');
            corner.className = 'grid-label grid-corner';
            container.appendChild(corner);

            // Add column headers (x coordinates)
            for (let x = 0; x < this.grid.width; x++) {
                const label = document.createElement('div');
                label.className = 'grid-label grid-col-label';
                label.textContent = x;
                container.appendChild(label);
            }

            // Add rows with row labels
            for (let y = 0; y < this.grid.height; y++) {
                // Row label (y coordinate)
                const rowLabel = document.createElement('div');
                rowLabel.className = 'grid-label grid-row-label';
                rowLabel.textContent = y;
                container.appendChild(rowLabel);

                // Grid cells for this row
                for (let x = 0; x < this.grid.width; x++) {
                    const cell = document.createElement('div');
                    cell.className = 'grid-cell';
                    cell.dataset.x = x;
                    cell.dataset.y = y;

                    // Add value display elements
                    const values = document.createElement('div');
                    values.className = 'cell-values';
                    values.innerHTML = `
                        <span class="f-value"></span>
                        <span class="g-value"></span>
                        <span class="h-value"></span>
                    `;
                    cell.appendChild(values);

                    container.appendChild(cell);
                }
            }

            this.renderGrid();
        }

        initEventListeners() {
            const container = document.getElementById('pathfinding-grid');
            if (!container) return;

            // Mouse events for drawing
            container.addEventListener('mousedown', (e) => {
                if (e.target.classList.contains('grid-cell') || e.target.closest('.grid-cell')) {
                    this.isDrawing = true;
                    this.handleCellClick(e);
                }
            });

            container.addEventListener('mousemove', (e) => {
                if (this.isDrawing) {
                    this.handleCellClick(e);
                }
            });

            document.addEventListener('mouseup', () => {
                this.isDrawing = false;
            });

            // Edit mode buttons
            document.querySelectorAll('.edit-mode-buttons .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.edit-mode-buttons .btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.editMode = btn.dataset.mode;
                });
            });

            // Clear grid button
            document.getElementById('btn-clear-grid')?.addEventListener('click', () => {
                this.grid.clear();
                this.nodeValues.clear();
                this.renderGrid();
            });

            // Show values checkbox
            document.getElementById('show-values')?.addEventListener('change', (e) => {
                this.showValues = e.target.checked;
                this.renderGrid();
            });
        }

        handleCellClick(e) {
            const cell = e.target.classList.contains('grid-cell') ? e.target : e.target.closest('.grid-cell');
            if (!cell) return;

            const x = parseInt(cell.dataset.x);
            const y = parseInt(cell.dataset.y);

            switch (this.editMode) {
                case 'wall':
                    if (this.grid.getCell(x, y) === CELL_EMPTY) {
                        this.grid.setCell(x, y, CELL_WALL);
                    }
                    break;
                case 'start':
                    this.grid.setCell(x, y, CELL_START);
                    break;
                case 'goal':
                    this.grid.setCell(x, y, CELL_GOAL);
                    break;
                case 'erase':
                    this.grid.setCell(x, y, CELL_EMPTY);
                    break;
            }

            this.renderGrid();
        }

        renderGrid() {
            const container = document.getElementById('pathfinding-grid');
            if (!container) return;

            const cells = container.querySelectorAll('.grid-cell');
            cells.forEach(cell => {
                const x = parseInt(cell.dataset.x);
                const y = parseInt(cell.dataset.y);
                const type = this.grid.getCell(x, y);

                // Reset classes
                cell.className = 'grid-cell';
                if (this.showValues) cell.classList.add('show-values');

                // Set type class
                switch (type) {
                    case CELL_WALL:
                        cell.classList.add('wall');
                        break;
                    case CELL_START:
                        cell.classList.add('start');
                        break;
                    case CELL_GOAL:
                        cell.classList.add('goal');
                        break;
                }

                // Update values display
                const key = `${x},${y}`;
                const values = this.nodeValues.get(key);
                const fSpan = cell.querySelector('.f-value');
                const gSpan = cell.querySelector('.g-value');
                const hSpan = cell.querySelector('.h-value');

                if (values && type !== CELL_WALL && type !== CELL_START && type !== CELL_GOAL) {
                    fSpan.textContent = values.f.toFixed(1);
                    gSpan.textContent = values.g.toFixed(1);
                    hSpan.textContent = values.h.toFixed(1);
                } else {
                    fSpan.textContent = '';
                    gSpan.textContent = '';
                    hSpan.textContent = '';
                }
            });
        }

        renderStep(step, path = [], goal = null) {
            const container = document.getElementById('pathfinding-grid');
            if (!container) return;

            // First reset to base grid state
            this.nodeValues.clear();

            // Clear heuristic line by default
            this.clearHeuristicLine();

            const cells = container.querySelectorAll('.grid-cell');
            cells.forEach(cell => {
                const x = parseInt(cell.dataset.x);
                const y = parseInt(cell.dataset.y);
                const type = this.grid.getCell(x, y);

                // Reset classes
                cell.className = 'grid-cell';
                if (this.showValues) cell.classList.add('show-values');

                // Set type class
                switch (type) {
                    case CELL_WALL:
                        cell.classList.add('wall');
                        break;
                    case CELL_START:
                        cell.classList.add('start');
                        break;
                    case CELL_GOAL:
                        cell.classList.add('goal');
                        break;
                }
            });

            // Apply step state
            if (step) {
                // Closed list
                if (step.closedList) {
                    for (const node of step.closedList) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                            cell.classList.add('closed');
                        }
                        this.nodeValues.set(`${node.x},${node.y}`, { f: node.f, g: node.g, h: node.h });
                    }
                }

                // Open list
                if (step.openList) {
                    for (const node of step.openList) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal') && !cell.classList.contains('closed')) {
                            cell.classList.add('open');
                        }
                        this.nodeValues.set(`${node.x},${node.y}`, { f: node.f, g: node.g, h: node.h });
                    }
                }

                // Current node
                if (step.current) {
                    const cell = container.querySelector(`[data-x="${step.current.x}"][data-y="${step.current.y}"]`);
                    if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                        cell.classList.remove('open', 'closed');
                        cell.classList.add('current');
                    }
                }

                // Neighbor being checked
                if (step.neighbor && step.type === STEP_NEIGHBOR_CHECK) {
                    const cell = container.querySelector(`[data-x="${step.neighbor.x}"][data-y="${step.neighbor.y}"]`);
                    if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                        cell.classList.add('neighbor');
                    }
                }

                // Path (for goal found)
                if (step.path) {
                    for (const node of step.path) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                            cell.classList.remove('open', 'closed', 'current');
                            cell.classList.add('path');
                        }
                    }
                }

                // Draw heuristic line from current node to goal
                if (step.current && goal && step.type !== STEP_GOAL_FOUND && step.type !== STEP_NO_SOLUTION) {
                    this.drawHeuristicLine(step.current, goal, step.current.h);
                }
            }

            // Render values
            this.renderGrid();

            // Re-apply visualization classes after renderGrid
            if (step) {
                if (step.closedList) {
                    for (const node of step.closedList) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal') && !cell.classList.contains('wall')) {
                            cell.classList.add('closed');
                        }
                    }
                }
                if (step.openList) {
                    for (const node of step.openList) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal') && !cell.classList.contains('wall') && !cell.classList.contains('closed')) {
                            cell.classList.add('open');
                        }
                    }
                }
                if (step.current) {
                    const cell = container.querySelector(`[data-x="${step.current.x}"][data-y="${step.current.y}"]`);
                    if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                        cell.classList.remove('open', 'closed');
                        cell.classList.add('current');
                    }
                }
                if (step.path) {
                    for (const node of step.path) {
                        const cell = container.querySelector(`[data-x="${node.x}"][data-y="${node.y}"]`);
                        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                            cell.classList.remove('open', 'closed', 'current');
                            cell.classList.add('path');
                        }
                    }
                    // Draw path line
                    this.drawPathLine(step.path);
                }
            }
        }

        updateMetrics(metrics) {
            document.getElementById('metric-expanded').textContent = metrics.nodesExpanded ?? 0;
            document.getElementById('metric-generated').textContent = metrics.nodesGenerated ?? 0;
            document.getElementById('metric-path-length').textContent = metrics.pathLength !== null && metrics.pathLength !== undefined ? metrics.pathLength : '-';
            document.getElementById('metric-path-cost').textContent = metrics.pathCost !== null && metrics.pathCost !== undefined ? metrics.pathCost.toFixed(2) : '-';
            document.getElementById('metric-time').textContent = metrics.time ? metrics.time.toFixed(2) + 'ms' : '-';
        }

        updateLists(openList, closedList, goal = null) {
            const openContainer = document.getElementById('open-list');
            const closedContainer = document.getElementById('closed-list');
            const openCount = document.getElementById('open-count');
            const closedCount = document.getElementById('closed-count');

            if (openContainer) {
                if (openList && openList.length > 0) {
                    openContainer.innerHTML = openList.map(n =>
                        `<span class="node-list-item" data-x="${n.x}" data-y="${n.y}" data-h="${n.h}">(${n.x},${n.y}) f=${n.f.toFixed(1)}</span>`
                    ).join('');
                } else {
                    openContainer.innerHTML = '<div class="list-empty">No nodes</div>';
                }
            }

            if (closedContainer) {
                if (closedList && closedList.length > 0) {
                    closedContainer.innerHTML = closedList.map(n =>
                        `<span class="node-list-item" data-x="${n.x}" data-y="${n.y}" data-h="${n.h}">(${n.x},${n.y})</span>`
                    ).join('');
                } else {
                    closedContainer.innerHTML = '<div class="list-empty">No nodes</div>';
                }
            }

            if (openCount) openCount.textContent = openList ? openList.length : 0;
            if (closedCount) closedCount.textContent = closedList ? closedList.length : 0;

            // Add hover listeners for highlighting
            this.setupListHoverListeners(goal);
        }

        setupListHoverListeners(goal) {
            const listItems = document.querySelectorAll('.node-list-item');
            const gridContainer = document.getElementById('pathfinding-grid');

            listItems.forEach(item => {
                item.addEventListener('mouseenter', () => {
                    const x = parseInt(item.dataset.x);
                    const y = parseInt(item.dataset.y);
                    const h = parseFloat(item.dataset.h);

                    // Highlight the cell in the grid
                    const cell = gridContainer?.querySelector(`[data-x="${x}"][data-y="${y}"]`);
                    if (cell) {
                        cell.classList.add('hover-highlight');
                    }

                    // Draw heuristic line if we have goal and h value
                    if (goal && !isNaN(h)) {
                        this.drawHeuristicLine({ x, y }, goal, h);
                    }

                    // Highlight the list item
                    item.classList.add('highlight');
                });

                item.addEventListener('mouseleave', () => {
                    const x = parseInt(item.dataset.x);
                    const y = parseInt(item.dataset.y);

                    // Remove highlight from cell
                    const cell = gridContainer?.querySelector(`[data-x="${x}"][data-y="${y}"]`);
                    if (cell) {
                        cell.classList.remove('hover-highlight');
                    }

                    // Clear heuristic line
                    this.clearHeuristicLine();

                    // Remove list item highlight
                    item.classList.remove('highlight');
                });
            });
        }

        addLogEntry(message, type = 'info') {
            const log = document.getElementById('execution-log');
            if (!log) return;

            const icons = {
                info: 'fa-info-circle',
                expand: 'fa-expand',
                neighbor: 'fa-arrows-alt',
                add: 'fa-plus',
                update: 'fa-pencil',
                success: 'fa-check-circle',
                failure: 'fa-times-circle'
            };

            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.innerHTML = `
                <span class="log-icon"><i class="fa ${icons[type] || icons.info}"></i></span>
                <span class="log-message">${message}</span>
            `;

            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }

        clearLog() {
            const log = document.getElementById('execution-log');
            if (log) {
                log.innerHTML = `
                    <div class="log-entry log-info">
                        <span class="log-icon"><i class="fa fa-info-circle"></i></span>
                        <span class="log-message">Draw walls, place start/goal, then click Solve to begin.</span>
                    </div>
                `;
            }
        }

        setHeuristic(heuristic) {
            this.currentHeuristic = heuristic;
        }

        // Calculate cell center position in pixels
        getCellCenter(x, y) {
            // Account for grid border (2px), gap (1px between cells), and label row/column
            const borderWidth = 2;
            const padding = 1;
            const labelSize = 20; // Size of label row/column
            const cellWithGap = this.cellSize + this.gridGap;

            return {
                px: borderWidth + padding + labelSize + this.gridGap + (x * cellWithGap) + (this.cellSize / 2),
                py: borderWidth + padding + labelSize + this.gridGap + (y * cellWithGap) + (this.cellSize / 2)
            };
        }

        // Draw heuristic visualization line from current node to goal
        drawHeuristicLine(current, goal, hValue) {
            const overlay = document.getElementById('heuristic-overlay');
            if (!overlay || !current || !goal || hValue === undefined || hValue === null) {
                this.clearHeuristicLine();
                return;
            }

            // Don't show line for Dijkstra (h=0)
            if (this.currentHeuristic === 'dijkstra') {
                this.clearHeuristicLine();
                return;
            }

            // Update cell size based on actual grid
            const gridEl = document.getElementById('pathfinding-grid');
            if (gridEl) {
                const firstCell = gridEl.querySelector('.grid-cell');
                if (firstCell) {
                    this.cellSize = firstCell.offsetWidth;
                }
            }

            const start = this.getCellCenter(current.x, current.y);
            const end = this.getCellCenter(goal.x, goal.y);

            // Build path based on heuristic type
            let pathData;
            let arrowAngle;
            let labelPos;

            if (this.currentHeuristic === 'manhattan') {
                // Manhattan: L-shaped path (horizontal then vertical)
                const corner = { px: end.px, py: start.py };

                // Shorten final segment for arrow
                const finalAngle = Math.atan2(end.py - corner.py, end.px - corner.px);
                const lineEnd = {
                    px: end.px - Math.cos(finalAngle) * 10,
                    py: end.py - Math.sin(finalAngle) * 10
                };

                pathData = `M ${start.px} ${start.py} L ${corner.px} ${corner.py} L ${lineEnd.px} ${lineEnd.py}`;
                arrowAngle = Math.PI / 2 * Math.sign(end.py - start.py) || Math.PI / 2; // Pointing down/up
                if (end.py === start.py) {
                    // Same row, just horizontal
                    arrowAngle = end.px > start.px ? 0 : Math.PI;
                }

                // Label at the corner
                labelPos = { x: corner.px, y: corner.py };

            } else if (this.currentHeuristic === 'chebyshev') {
                // Chebyshev: diagonal then straight (king's movement)
                const dx = end.px - start.px;
                const dy = end.py - start.py;
                const absDx = Math.abs(dx);
                const absDy = Math.abs(dy);

                // Move diagonally first, then straight
                const diagDist = Math.min(absDx, absDy);
                const diagX = start.px + diagDist * Math.sign(dx);
                const diagY = start.py + diagDist * Math.sign(dy);

                const corner = { px: diagX, py: diagY };

                // Calculate arrow angle for final segment
                const finalAngle = Math.atan2(end.py - corner.py, end.px - corner.px);
                const lineEnd = {
                    px: end.px - Math.cos(finalAngle) * 10,
                    py: end.py - Math.sin(finalAngle) * 10
                };

                if (absDx === absDy) {
                    // Pure diagonal
                    pathData = `M ${start.px} ${start.py} L ${lineEnd.px} ${lineEnd.py}`;
                    arrowAngle = Math.atan2(dy, dx);
                    labelPos = { x: (start.px + end.px) / 2, y: (start.py + end.py) / 2 };
                } else {
                    pathData = `M ${start.px} ${start.py} L ${corner.px} ${corner.py} L ${lineEnd.px} ${lineEnd.py}`;
                    arrowAngle = finalAngle;
                    labelPos = { x: corner.px, y: corner.py };
                }

            } else {
                // Euclidean: straight line
                const angle = Math.atan2(end.py - start.py, end.px - start.px);
                const lineEnd = {
                    px: end.px - Math.cos(angle) * 10,
                    py: end.py - Math.sin(angle) * 10
                };

                pathData = `M ${start.px} ${start.py} L ${lineEnd.px} ${lineEnd.py}`;
                arrowAngle = angle;
                labelPos = { x: (start.px + end.px) / 2, y: (start.py + end.py) / 2 };
            }

            // Arrow head calculations
            const arrowLength = 8;
            const arrowWidth = 5;
            const arrowTip = {
                x: end.px - Math.cos(arrowAngle) * 5,
                y: end.py - Math.sin(arrowAngle) * 5
            };
            const arrowLeft = {
                x: arrowTip.x - arrowLength * Math.cos(arrowAngle) - arrowWidth * Math.sin(arrowAngle),
                y: arrowTip.y - arrowLength * Math.sin(arrowAngle) + arrowWidth * Math.cos(arrowAngle)
            };
            const arrowRight = {
                x: arrowTip.x - arrowLength * Math.cos(arrowAngle) + arrowWidth * Math.sin(arrowAngle),
                y: arrowTip.y - arrowLength * Math.sin(arrowAngle) - arrowWidth * Math.cos(arrowAngle)
            };

            const heuristicClass = `heuristic-${this.currentHeuristic}`;

            overlay.innerHTML = `
                <g class="${heuristicClass}">
                    <!-- Glow effect -->
                    <path class="heuristic-line-glow" d="${pathData}" />

                    <!-- Main dashed line -->
                    <path class="heuristic-line" d="${pathData}" />

                    <!-- Arrow head -->
                    <polygon class="heuristic-arrow"
                             points="${arrowTip.x},${arrowTip.y} ${arrowLeft.x},${arrowLeft.y} ${arrowRight.x},${arrowRight.y}" />

                    <!-- Label background -->
                    <rect class="heuristic-label-bg"
                          x="${labelPos.x - 18}" y="${labelPos.y - 8}"
                          width="36" height="16" rx="3" />

                    <!-- H value label -->
                    <text class="heuristic-label"
                          x="${labelPos.x}" y="${labelPos.y + 4}"
                          text-anchor="middle">h=${hValue.toFixed(1)}</text>
                </g>
            `;
        }

        clearHeuristicLine() {
            const overlay = document.getElementById('heuristic-overlay');
            if (overlay) {
                overlay.innerHTML = '';
            }
        }

        drawPathLine(path) {
            const overlay = document.getElementById('heuristic-overlay');
            if (!overlay || !path || path.length < 2) return;

            // Update cell size based on actual grid
            const gridEl = document.getElementById('pathfinding-grid');
            if (gridEl) {
                const firstCell = gridEl.querySelector('.grid-cell');
                if (firstCell) {
                    this.cellSize = firstCell.offsetWidth;
                }
            }

            // Build path data
            const points = path.map(node => this.getCellCenter(node.x, node.y));
            const pathData = points.map((p, i) =>
                (i === 0 ? 'M' : 'L') + ` ${p.px} ${p.py}`
            ).join(' ');

            overlay.innerHTML = `
                <g class="path-line-group">
                    <!-- Glow effect -->
                    <path class="path-line-glow" d="${pathData}" />
                    <!-- Main path line -->
                    <path class="path-line" d="${pathData}" />
                </g>
            `;
        }
    }

    // ============================================
    // Playback Controller
    // ============================================

    class PlaybackController {
        constructor(ui) {
            this.ui = ui;
            this.steps = [];
            this.currentStepIndex = -1;
            this.isPlaying = false;
            this.playbackSpeed = 9;
            this.playbackTimer = null;
            this.onStepChange = null;
            this.goal = null;
        }

        loadSteps(steps, goal = null) {
            this.steps = steps;
            this.goal = goal;
            this.currentStepIndex = -1;
            this.isPlaying = false;
            this.updateStepDisplay();
        }

        play() {
            if (this.steps.length === 0) return;
            this.isPlaying = true;
            this.playNext();
        }

        pause() {
            this.isPlaying = false;
            if (this.playbackTimer) {
                clearTimeout(this.playbackTimer);
                this.playbackTimer = null;
            }
        }

        playNext() {
            if (!this.isPlaying) return;

            if (this.currentStepIndex < this.steps.length - 1) {
                this.stepForward();
                const delay = this.getDelay();
                this.playbackTimer = setTimeout(() => this.playNext(), delay);
            } else {
                this.isPlaying = false;
            }
        }

        stepForward() {
            if (this.currentStepIndex < this.steps.length - 1) {
                this.currentStepIndex++;
                this.renderCurrentStep();
                this.updateStepDisplay();
            }
        }

        stepBackward() {
            if (this.currentStepIndex > 0) {
                this.currentStepIndex--;
                this.renderCurrentStep();
                this.updateStepDisplay();
            }
        }

        goToStep(index) {
            if (index >= 0 && index < this.steps.length) {
                this.currentStepIndex = index;
                this.renderCurrentStep();
                this.updateStepDisplay();
            }
        }

        reset() {
            this.pause();
            this.currentStepIndex = -1;
            this.goal = null;
            this.ui.clearHeuristicLine();
            this.ui.renderGrid();
            this.updateStepDisplay();
        }

        setSpeed(speed) {
            this.playbackSpeed = speed;
        }

        getDelay() {
            // Speed 1 = 1000ms, Speed 10 = 50ms
            return 1050 - (this.playbackSpeed * 100);
        }

        renderCurrentStep() {
            const step = this.steps[this.currentStepIndex];
            if (step) {
                this.ui.renderStep(step, [], this.goal);
                this.ui.updateLists(step.openList, step.closedList, this.goal);

                // Update metrics from step
                if (step.metrics) {
                    this.ui.updateMetrics(step.metrics);
                }

                // Log the step
                const logType = this.getLogType(step.type);
                this.ui.addLogEntry(step.message, logType);

                if (this.onStepChange) {
                    this.onStepChange(step, this.currentStepIndex);
                }
            }
        }

        getLogType(stepType) {
            switch (stepType) {
                case STEP_INIT: return 'info';
                case STEP_EXPAND: return 'expand';
                case STEP_NEIGHBOR_CHECK: return 'neighbor';
                case STEP_ADD_TO_OPEN: return 'add';
                case STEP_UPDATE_PATH: return 'update';
                case STEP_GOAL_FOUND: return 'success';
                case STEP_NO_SOLUTION: return 'failure';
                default: return 'info';
            }
        }

        updateStepDisplay() {
            const display = document.getElementById('playback-step');
            if (display) {
                display.textContent = `Step: ${this.currentStepIndex + 1} / ${this.steps.length}`;
            }
        }
    }

    // ============================================
    // Main Application
    // ============================================

    class AStarApp {
        constructor() {
            this.grid = new GridState();
            this.solver = new AStarSolver();
            this.ui = new UIController(this.grid);
            this.playback = new PlaybackController(this.ui);
            this.currentResult = null;
            this.comparisonResults = new Map();

            this.initEventListeners();
            this.loadSample('easy');
        }

        initEventListeners() {
            // Sample input buttons
            document.querySelectorAll('.sample-input-tabs .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.sample-input-tabs .btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.loadSample(btn.dataset.sample);
                });
            });

            // Solve button
            document.getElementById('btn-solve')?.addEventListener('click', () => this.solve());

            // Pause button
            document.getElementById('btn-pause')?.addEventListener('click', () => {
                if (this.playback.isPlaying) {
                    this.playback.pause();
                    document.getElementById('btn-pause').innerHTML = '<i class="fa fa-play"></i>';
                } else {
                    this.playback.play();
                    document.getElementById('btn-pause').innerHTML = '<i class="fa fa-pause"></i>';
                }
            });

            // Step buttons
            document.getElementById('btn-step')?.addEventListener('click', () => {
                this.playback.pause();
                this.playback.stepForward();
                document.getElementById('btn-pause').innerHTML = '<i class="fa fa-play"></i>';
            });

            document.getElementById('btn-step-back')?.addEventListener('click', () => {
                this.playback.pause();
                this.playback.stepBackward();
                document.getElementById('btn-pause').innerHTML = '<i class="fa fa-play"></i>';
            });

            // Reset button
            document.getElementById('btn-reset')?.addEventListener('click', () => this.reset());

            // Speed slider
            document.getElementById('speed-slider')?.addEventListener('input', (e) => {
                this.playback.setSpeed(parseInt(e.target.value));
            });

            // Movement mode dropdown - auto-select appropriate heuristic
            document.getElementById('movement-mode')?.addEventListener('change', (e) => {
                const heuristicSelect = document.getElementById('heuristic-select');
                if (heuristicSelect) {
                    if (e.target.value === 'grid') {
                        heuristicSelect.value = 'manhattan';
                    } else {
                        heuristicSelect.value = 'euclidean';
                    }
                }
                // Re-run comparison with new movement mode
                this.runComparison();
            });

            // Heuristic dropdown - re-run comparison when changed
            document.getElementById('heuristic-select')?.addEventListener('change', () => {
                this.runComparison();
            });

            // Heuristic weight slider
            document.getElementById('heuristic-weight')?.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                document.getElementById('weight-value').textContent = value.toFixed(1);
                this.runComparison();
            });

            // Clear log button
            document.getElementById('btn-clear-log')?.addEventListener('click', () => {
                this.ui.clearLog();
            });

            // Info panel tabs
            document.querySelectorAll('.info-panel-tabs .btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.info-panel-tabs .btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');

                    document.querySelectorAll('.info-tab-content').forEach(tab => tab.classList.remove('active'));
                    const tabId = `tab-${btn.dataset.tab}`;
                    document.getElementById(tabId)?.classList.add('active');
                });
            });

            // Run comparison button
            document.getElementById('btn-run-comparison')?.addEventListener('click', () => this.runComparison());

            // Comparison table row click
            document.getElementById('comparison-table')?.addEventListener('click', (e) => {
                const row = e.target.closest('tr');
                if (row && row.dataset.heuristic) {
                    this.loadComparisonResult(row.dataset.heuristic);
                }
            });
        }

        loadSample(name) {
            this.reset();
            this.grid.loadSample(name);
            this.ui.renderGrid();
            // Auto-populate comparison table
            this.runComparison();
        }

        solve() {
            if (!this.grid.start || !this.grid.goal) {
                alert('Please place both a start and goal position.');
                return;
            }

            this.ui.clearLog();

            const movementMode = document.getElementById('movement-mode')?.value || 'grid';
            const options = {
                heuristic: document.getElementById('heuristic-select')?.value || 'manhattan',
                allowDiagonal: movementMode === 'free',
                tiebreaker: document.getElementById('tiebreaker-select')?.value || 'none',
                weight: parseFloat(document.getElementById('heuristic-weight')?.value) || 1.0
            };

            // Set heuristic on UI for visualization
            this.ui.setHeuristic(options.heuristic);

            this.currentResult = this.solver.solve(this.grid, options);
            this.playback.loadSteps(this.currentResult.steps, this.grid.goal);

            // Enable controls
            document.getElementById('btn-pause').disabled = false;
            document.getElementById('btn-step').disabled = false;
            document.getElementById('btn-step-back').disabled = false;

            // Start playback (metrics will update incrementally)
            document.getElementById('btn-pause').innerHTML = '<i class="fa fa-pause"></i>';
            this.playback.play();
        }

        reset() {
            this.playback.reset();
            this.ui.nodeValues.clear();
            this.ui.clearHeuristicLine();
            this.ui.renderGrid();
            this.ui.updateMetrics({});
            this.ui.updateLists([], []);
            this.ui.clearLog();

            // Disable controls
            document.getElementById('btn-pause').disabled = true;
            document.getElementById('btn-pause').innerHTML = '<i class="fa fa-pause"></i>';
            document.getElementById('btn-step').disabled = true;
            document.getElementById('btn-step-back').disabled = true;
        }

        runComparison() {
            if (!this.grid.start || !this.grid.goal) {
                // Clear comparison table if no start/goal
                const tbody = document.querySelector('#comparison-table tbody');
                if (tbody) tbody.innerHTML = '';
                return;
            }

            this.comparisonResults.clear();
            const heuristics = ['manhattan', 'euclidean', 'chebyshev', 'dijkstra'];
            const movementMode = document.getElementById('movement-mode')?.value || 'grid';
            const allowDiagonal = movementMode === 'free';
            const tiebreaker = document.getElementById('tiebreaker-select')?.value || 'none';
            const weight = parseFloat(document.getElementById('heuristic-weight')?.value) || 1.0;

            const results = [];

            for (const heuristic of heuristics) {
                const result = this.solver.solve(this.grid, { heuristic, allowDiagonal, tiebreaker, weight });
                this.comparisonResults.set(heuristic, result);

                results.push({
                    heuristic,
                    expanded: result.metrics.nodesExpanded,
                    generated: result.metrics.nodesGenerated,
                    pathLength: result.success ? result.metrics.pathLength : null,
                    pathCost: result.success ? result.metrics.pathCost : null,
                    time: result.metrics.time,
                    success: result.success
                });
            }

            this.renderComparisonTable(results);
        }

        renderComparisonTable(results) {
            const tbody = document.querySelector('#comparison-table tbody');
            if (!tbody) return;

            // Find best values (lowest expanded, shortest path, etc.)
            const best = {
                expanded: Math.min(...results.map(r => r.expanded)),
                generated: Math.min(...results.map(r => r.generated)),
                pathLength: Math.min(...results.filter(r => r.pathLength !== null).map(r => r.pathLength)),
                pathCost: Math.min(...results.filter(r => r.pathCost !== null).map(r => r.pathCost)),
                time: Math.min(...results.map(r => r.time))
            };

            tbody.innerHTML = results.map(r => {
                const isBestExpanded = r.expanded === best.expanded;
                const isBestGenerated = r.generated === best.generated;
                const isBestLength = r.pathLength !== null && r.pathLength === best.pathLength;
                const isBestCost = r.pathCost !== null && r.pathCost === best.pathCost;
                const isBestTime = r.time === best.time;

                const heuristicName = r.heuristic.charAt(0).toUpperCase() + r.heuristic.slice(1);

                return `
                    <tr data-heuristic="${r.heuristic}">
                        <td>${heuristicName}</td>
                        <td class="${isBestExpanded ? 'best-value' : ''}">${r.expanded}</td>
                        <td class="${isBestGenerated ? 'best-value' : ''}">${r.generated}</td>
                        <td class="${r.pathLength === null ? 'no-solution-value' : (isBestLength ? 'best-value' : '')}">${r.pathLength ?? 'No path'}</td>
                        <td class="${r.pathCost === null ? 'no-solution-value' : (isBestCost ? 'best-value' : '')}">${r.pathCost !== null ? r.pathCost.toFixed(2) : 'N/A'}</td>
                        <td class="${isBestTime ? 'best-value' : ''}">${r.time.toFixed(2)}</td>
                    </tr>
                `;
            }).join('');
        }

        loadComparisonResult(heuristic) {
            const result = this.comparisonResults.get(heuristic);
            if (!result) return;

            // Update selection in table
            document.querySelectorAll('#comparison-table tbody tr').forEach(row => {
                row.classList.toggle('selected-row', row.dataset.heuristic === heuristic);
            });

            // Update heuristic dropdown and UI
            const select = document.getElementById('heuristic-select');
            if (select) select.value = heuristic;
            this.ui.setHeuristic(heuristic);

            // Load the result
            this.currentResult = result;
            this.playback.loadSteps(result.steps, this.grid.goal);
            this.ui.clearLog();

            // Enable controls
            document.getElementById('btn-pause').disabled = false;
            document.getElementById('btn-step').disabled = false;
            document.getElementById('btn-step-back').disabled = false;

            // Go to last step to show final state (metrics will update from step)
            if (result.steps.length > 0) {
                this.playback.goToStep(result.steps.length - 1);
            }
        }
    }

    // ============================================
    // Initialize on DOM ready
    // ============================================

    document.addEventListener('DOMContentLoaded', () => {
        window.astarApp = new AStarApp();
    });

})();
