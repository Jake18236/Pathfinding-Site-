# AI Algorithm Visualization Design Principles

A guide for creating consistent, educational algorithm visualizations. Patterns derived from four production visualizations: A* Pathfinding, Minimax with Alpha-Beta Pruning, Dimensionality Reduction (MNIST/CIFAR-10), and Constraint Satisfaction (Shidoku).

---

## Core Philosophy

**Show the "why," not just the "what."** Students understand algorithms when they see *why* decisions are made, not just *what* the final answer is. Every visualization should expose the algorithm's reasoning process.

### Meta-Principles

These higher-order patterns emerge across all successful visualizations:

1. **Expose Tradeoffs, Not Just Results** - Every algorithm involves tradeoffs (speed vs. optimality, memory vs. time, local vs. global structure). Make these tradeoffs *visible and manipulable* through controls that let students experience the consequences of different choices.

2. **Dual-Mode Data Sources** - Offer both "raw" and "processed" views of the same problem. This teaches students that representation matters:
   - A*: Dijkstra (no heuristic) vs. heuristic-guided search
   - MNIST: Raw 784D pixels vs. learned 128D embeddings
   - Shidoku: Basic backtracking vs. AC-3 preprocessed
   - Minimax: Naive evaluation vs. alpha-beta pruning

3. **"Watch It Learn" Progressions** - Show transformation over time, making the "aha moment" visible. Whether it's training epochs, pruning efficiency growing, arc consistency propagating, or a frontier expanding—students should witness the algorithm *improving*.

4. **Failure as Pedagogy** - Failures aren't errors to hide; they're learning opportunities to highlight. Show *why* things fail, not just *that* they failed. The backtrack happened because the domain became empty. The path doesn't exist because walls create an impassable barrier.

---

## Layout Structure

### Two-Panel Design
```
┌─────────────────────────────────────────────────────────┐
│  LEFT PANEL (60-70%)    │   RIGHT PANEL (30-40%)        │
│                         │                               │
│  Main Visualization     │   Controls & Information      │
│  - The "thing" being    │   - Playback controls         │
│    computed on          │   - Algorithm options         │
│  - Interactive grid,    │   - Tabbed info panels        │
│    graph, tree, etc.    │   - Metrics display           │
│                         │                               │
└─────────────────────────────────────────────────────────┘
```

### Right Panel Organization
Use collapsible panels with clear headers:
1. **Playback** - Always at top, always visible
2. **Algorithm Options** - Checkboxes for toggling algorithm variants
3. **Sample Inputs** - Preloaded examples of varying difficulty
4. **Information Tabs** - About, data structures, logs, comparisons

---

## Color System

### Semantic Color Variables
Define colors by *meaning*, not appearance. Use CSS variables for theming:

```css
/* State colors - what the algorithm is doing */
--viz-fixed: /* Given/immutable input */
--viz-assigned: /* Algorithm made this choice */
--viz-examining: /* Currently being considered */
--viz-pruned: /* Eliminated from consideration */
--viz-conflict: /* Constraint violation detected */
--viz-success: /* Goal state reached */

/* UI colors */
--viz-bg: /* Background */
--viz-text: /* Primary text */
--viz-border: /* Borders and dividers */
--viz-highlight: /* Interactive hover states */
```

### Color Pairing
Each semantic state needs both a background AND a border/text color:
```css
--viz-assigned-bg: #e8f5e9;
--viz-assigned-border: #2e7d32;
```

### Theme Support
Support at least light and dark themes. Use `[data-theme="dark"]` selectors:
```css
[data-theme="dark"] {
    --viz-bg: #1d2021;
    --viz-text: #ebdbb2;
    /* ... */
}
```

---

## Playback System

### Essential Controls
- **Play/Pause** - Toggle continuous playback
- **Step Forward** - Single step advance
- **Step Backward** - Single step rewind (crucial for understanding)
- **Reset** - Return to initial state
- **Speed Slider** - Adjustable playback speed

### Step Recording
Record algorithm execution as discrete steps, each containing:
```javascript
{
    type: 'ASSIGN' | 'BACKTRACK' | 'PRUNE' | 'CHECK' | ...,
    target: /* what element is affected */,
    value: /* what changed */,
    reason: /* why this happened (optional but valuable) */,
    snapshot: /* full state for random access */
}
```

### Metrics Tracking
Track and display algorithm performance metrics:
- **Universal**: Time, iterations/steps
- **Search**: Nodes expanded, backtracks, depth reached
- **Constraint**: Arc checks, domain reductions, propagations
- **Optimization**: Best found, improvements, evaluations

---

## Information Display

### Tabbed Interface
Use tabs to organize different views without overwhelming:

| Tab | Purpose |
|-----|---------|
| **About** | Algorithm explanation, controls reference |
| **Data Structure** | Tree, queue, stack, etc. visualization |
| **Execution Log** | Step-by-step textual history |
| **Compare** | Side-by-side algorithm variant comparison |

### Domain/State Visualization
For constraint problems, show the *possibility space*:
- Display all possible values for each variable
- Cross off eliminated values (but keep them visible)
- Highlight the current assignment
- Show *why* values were eliminated when possible

### Search Tree/Graph
For search algorithms, show the exploration structure:
- Highlight current node
- Show expanded vs. frontier vs. unexplored
- Display path to current state
- Allow clicking nodes to jump to that state

---

## Interactivity

### Input Editing
Let users create custom inputs:
- Editable grid cells, graph nodes, etc.
- Clear visual distinction between user input and algorithm state
- Validation with helpful error messages

### State Exploration
Allow non-linear exploration:
- Click algorithm comparison rows to view end states
- Click search tree nodes to jump to that step
- Click log entries to jump to that moment

### Algorithm Configuration
Expose algorithm variants as toggles:
```
☑ Use heuristic X
☐ Enable optimization Y
☑ Preprocess with Z
```
This teaches which components matter and why.

---

## Comparison Mode

### Auto-Run Comparisons
When input changes, automatically run all algorithm variants and display results in a comparison table.

### Metrics Table
| Algorithm | Metric 1 | Metric 2 | Metric 3 | Time |
|-----------|----------|----------|----------|------|
| Variant A | 42 | 12 | 156 | 2.3ms |
| Variant B | **28** | 8 | **89** | **1.1ms** |

- **Bold/highlight** best values in each column
- Handle N/A gracefully (some metrics don't apply to all variants)
- Show failures clearly but not harshly ("Did not find solution")

### Click-to-View
Clicking an algorithm name should:
1. Display that algorithm's end state on the main visualization
2. Set the algorithm option checkboxes to match
3. Allow immediate re-run with animation

---

## Educational Scaffolding

### Sample Inputs by Difficulty
Provide curated examples:
- **Easy** - Completes quickly, demonstrates basic behavior
- **Medium** - Shows interesting algorithm decisions
- **Hard** - Demonstrates why optimizations matter
- **Edge Cases** - Unsolvable, multiple solutions, worst-case inputs

### Progressive Disclosure
- Default view shows essential information
- Collapsible panels reveal advanced details
- Tooltips explain technical terms
- "About" tab provides context for novices

### Failure Visualization
Show failures as learning opportunities:
- Failed attempts visible but distinguished (crossed off, grayed)
- Backtrack paths in search trees
- "Why did this fail?" should be answerable from the visualization

---

## Technical Implementation

### Framework Agnostic
The visualization should work with:
- Static site generators (11ty, Jekyll)
- Vanilla JS (no framework required)
- Optional framework integration

### File Organization
```
src/
  algorithm-name.njk     # HTML template
static/
  css/
    algorithm-name.css   # Visualization styles
  js/
    algorithm-name.js    # Solver + UI logic
```

### Class Structure
```javascript
// Solver: Pure algorithm, no UI dependencies
class AlgorithmSolver {
    solve(input, options) → { success, result, steps, metrics }
}

// UI Controller: Renders state to DOM
class UIController {
    render(state)
    highlightElement(id, type)
    updateMetrics(metrics)
}

// Playback Controller: Manages step navigation
class PlaybackController {
    play() / pause() / step() / reset()
    goToStep(n)
}

// App: Coordinates everything
class VisualizerApp {
    constructor()
    initControls()
    solve()
    runComparison()
}
```

### Responsive Design
- Panels should stack vertically on mobile
- Touch-friendly controls (larger tap targets)
- Readable font sizes at all breakpoints

### Performance Patterns for Scale

Visualizations may need to handle large datasets (1000s of points, 500K+ tree nodes). Use these patterns:

#### Web Workers for Heavy Computation
When computation exceeds ~100ms or involves >10K operations, offload to Web Workers:
```javascript
// minimax-worker.js pattern
self.onmessage = function(e) {
    const { game, depth, options } = e.data;
    const result = buildGameTree(game, depth, options);
    // Send progress updates for long operations
    self.postMessage({ type: 'progress', phase: 'building', percent: 50 });
    self.postMessage({ type: 'complete', result });
};
```
Benefits: UI remains responsive, progress indicators possible, can be cancelled.

#### Lazy Loading for Large Datasets
Load data on-demand rather than upfront:
- **MNIST**: Images fetch only on hover (quadtree finds point → load image)
- **Minimax**: Tree nodes render only when expanded (collapsed Set tracks state)
- **A***: Heuristic comparison runs only when algorithm selector clicked

#### Spatial Indexing for Interactive Point Queries
When users interact with dense visualizations (scatter plots, large grids):
```javascript
// Quadtree for O(log n) point lookup
class Quadtree {
    insert(point) { /* ... */ }
    query(x, y, radius) { /* ... */ }
}
// Used for: hover detection in MNIST scatter plot
```

#### Collapsible Hierarchies
For tree structures that could have thousands of nodes:
- Render only visible/expanded nodes
- Track expanded state with `Set<nodeId>`
- Provide "expand all to depth N" controls
- Show aggregate statistics for collapsed subtrees

### Binary Data Formats
For pre-computed visualizations with large datasets, consider binary formats:
```javascript
// Example: Minimax tree stored as binary for fast loading
const buffer = await fetch('tree.bin').then(r => r.arrayBuffer());
const view = new DataView(buffer);
// Parse header, then nodes on-demand
```

---

## Information Architecture

### Consistent Tab Ordering
All visualizations follow the same information progression:

| Order | Tab | Purpose | Example Content |
|-------|-----|---------|-----------------|
| 1 | **About** | Context for novices | Algorithm explanation, keyboard controls, "what am I looking at?" |
| 2 | **Algorithm/Data Structure** | Live internal state | Open/closed lists, search tree, current domains |
| 3 | **Log** | Temporal decision trace | Timestamped step-by-step history |
| 4 | **Compare** | Cross-variant analysis | Metrics table with click-to-load |

This progression moves from "what is this?" → "what's happening now?" → "what happened?" → "how do variants differ?"

### Legend Placement
- **Grid legends**: Below the visualization (doesn't compete with controls)
- **Scatter plot legends**: Below plot, clickable for filtering
- **Tree legends**: Corner overlay with zoom controls

---

## Checklist for New Visualizations

### Layout & Structure
- [ ] Two-panel layout (visualization left 60-70%, controls right 30-40%)
- [ ] Semantic color variables with light/dark themes
- [ ] Mobile-responsive layout (panels stack vertically)
- [ ] Legend placement follows conventions

### Playback & Interaction
- [ ] Playback controls (play, pause, step forward/back, reset, speed)
- [ ] Step recording with snapshots for random access
- [ ] Click-to-explore interactivity (lists, trees, comparison rows)
- [ ] Sample inputs by difficulty (easy → edge cases)

### Information Display
- [ ] Tabbed information panel (About → Data Structure → Log → Compare)
- [ ] Metrics tracking and display
- [ ] Algorithm variant toggles
- [ ] Comparison table with auto-run and click-to-load

### Educational Design
- [ ] Dual-mode data source (raw vs. processed/optimized)
- [ ] Tradeoffs exposed through manipulable controls
- [ ] Failure states clearly visualized with explanations
- [ ] "Watch it improve" progression visible

### Performance (if needed)
- [ ] Web Worker for computation >100ms
- [ ] Lazy loading for large datasets
- [ ] Spatial indexing for dense point interaction
- [ ] Collapsible hierarchies for large trees

---

## Example Algorithms to Visualize

### Implemented Visualizations

| Algorithm | Main Visualization | Key States | Tradeoff Exposed | Dual Mode |
|-----------|-------------------|------------|------------------|-----------|
| **A* Search** | Grid with path overlay | Open, closed, current, path | Optimality vs. speed (heuristic weight) | Dijkstra ↔ Heuristic-guided |
| **Minimax** | Collapsible game tree | Max, min, pruned, optimal | Nodes evaluated vs. decision quality | Naive ↔ Alpha-beta pruning |
| **Dimensionality Reduction** | Scatter plot | Class clusters, highlighted points | Local vs. global structure | Raw pixels ↔ CNN embeddings |
| **CSP (Shidoku)** | 4×4 grid + search tree | Fixed, assigned, conflict, backtrack | Completeness vs. efficiency | Backtracking ↔ MAC with AC-3 |

### Future Visualization Ideas

| Algorithm | Main Visualization | Key States | Interesting Metrics | Dual Mode Opportunity |
|-----------|-------------------|------------|---------------------|----------------------|
| K-Means | Scatter plot with centroids | Centroids, assignments, converged | Iterations, SSE, cluster sizes | Random init ↔ K-Means++ |
| Decision Tree | Tree + data table | Split nodes, leaf predictions | Depth, information gain, accuracy | No pruning ↔ Post-pruned |
| Neural Net | Network diagram | Activations, weights, gradients | Loss, gradient magnitudes, epochs | Untrained ↔ Trained |
| Genetic Algorithm | Population grid | Fitness levels, selected, mutated | Generation, best fitness, diversity | No crossover ↔ With crossover |
| BFS/DFS | Graph/tree | Frontier, explored, path | Nodes visited, memory, path optimality | BFS ↔ DFS (same problem) |
| Gradient Descent | 3D surface plot | Current position, gradient vector | Steps, loss, learning rate impact | Batch ↔ Stochastic |

---

## Architectural Patterns Across Visualizations

### Common Class Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                        VisualizerApp                            │
│  - Orchestrates all components                                  │
│  - Handles user input events                                    │
│  - Manages application state                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ coordinates
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ AlgorithmSolver│   │ UIController   │   │PlaybackController│
│               │   │               │   │               │
│ - Pure logic  │   │ - DOM updates │   │ - Step nav    │
│ - No UI deps  │   │ - Rendering   │   │ - Animation   │
│ - Returns:    │   │ - Event bind  │   │ - Speed ctrl  │
│   steps[]     │   │ - Highlights  │   │               │
│   metrics{}   │   │               │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Step Recording Pattern

All visualizations record execution as discrete steps. Each step captures:

```javascript
{
    type: 'EXPAND' | 'ASSIGN' | 'BACKTRACK' | 'PRUNE' | ...,
    target: cellId | nodeId | variableId,  // What was affected
    value: newValue,                        // What changed
    reason: 'domain empty' | 'goal found', // Why (optional but valuable)
    snapshot: { /* full state for random-access replay */ }
}
```

**Step type naming conventions:**
- `*_START` / `*_END` for phases (e.g., `AC3_START`, `AC3_END`)
- `TRY_*` for tentative actions (e.g., `TRY_VALUE`)
- Past tense for completed actions (e.g., `ASSIGNED`, `PRUNED`)
- `NO_*` for negative results (e.g., `NO_SOLUTION`)

---

## Animation & Timing

### Speed Control Philosophy

Speed sliders should map to meaningful ranges:
- **Slow (1-3)**: Learning mode, see every detail
- **Medium (4-6)**: Following along, understanding flow
- **Fast (7-10)**: Quick overview, comparing algorithms

### Visual Feedback Timing

| Action | Duration | Purpose |
|--------|----------|---------|
| Cell state change | 100-200ms | Visible but not slow |
| Conflict highlight | 300-500ms with pulse | Draw attention |
| Arc visualization | 150-250ms | Show constraint check |
| Node expansion | 50-100ms | Smooth tree growth |
| Backtrack | 200-300ms | Emphasize retreat |

### Hover Interactions

Hover should provide instant feedback (<50ms) with:
- Tooltip showing details
- Related elements highlighted (e.g., constraints, connected nodes)
- Visual guides (e.g., heuristic distance lines in A*)

---

## Final Thoughts

The best algorithm visualizations make the invisible visible. When a student can *see* why A* expands certain nodes, why backtracking retreats, or why a heuristic helps—they understand the algorithm at a deeper level than any textbook explanation provides.

**The meta-pattern**: Every visualization should answer "what would happen if I changed this?" Students learn by manipulating parameters and witnessing consequences. Build for exploration, not just demonstration.

Keep it simple. Keep it interactive. Show the reasoning.
