/**
 * TreeLayoutEngine - Computes tree node positions with overlap prevention
 *
 * Implements a variant of the Reingold-Tilford algorithm for drawing
 * trees with aesthetically pleasing layouts. Key features:
 * - Children centered under parents
 * - Contour-based overlap detection and fixing
 * - Bottom-up subtree shifting
 *
 * @example
 * const layout = new TreeLayoutEngine({
 *   horizontalSpacing: 100,
 *   verticalSpacing: 80,
 *   minNodeWidth: 60
 * });
 *
 * const positions = layout.computeLayout(rootNode, (node) => node.children);
 * // positions is a Map: nodeId -> { x, y }
 */
export class TreeLayoutEngine {
    /**
     * Create a TreeLayoutEngine
     * @param {Object} options - Configuration options
     * @param {number} [options.horizontalSpacing=100] - Horizontal spacing between sibling nodes
     * @param {number} [options.verticalSpacing=80] - Vertical spacing between levels
     * @param {number} [options.minNodeWidth=60] - Minimum width to prevent overlap
     */
    constructor(options = {}) {
        this.horizontalSpacing = options.horizontalSpacing ?? 100;
        this.verticalSpacing = options.verticalSpacing ?? 80;
        this.minNodeWidth = options.minNodeWidth ?? 60;
    }

    /**
     * Compute layout positions for all nodes in a tree
     *
     * @param {*} root - The root node or root node ID
     * @param {Function} getChildren - Function that returns array of child nodes/IDs: (node) => children[]
     * @param {Function} [getNodeId] - Function to get node ID: (node) => id (defaults to identity)
     * @returns {Map<string, {x: number, y: number}>} Map of node positions keyed by ID
     */
    computeLayout(root, getChildren, getNodeId = (n) => String(n)) {
        const positions = new Map();
        const rootId = getNodeId(root);

        // Build children map for quick lookup
        const childrenMap = new Map();
        const buildChildrenMap = (node, visited = new Set()) => {
            const nodeId = getNodeId(node);
            if (visited.has(nodeId)) return;
            visited.add(nodeId);

            const children = getChildren(node) || [];
            childrenMap.set(nodeId, children.map(c => getNodeId(c)));

            for (const child of children) {
                buildChildrenMap(child, visited);
            }
        };
        buildChildrenMap(root);

        // Pass 1: Initial positioning - children centered under parent with fixed spacing
        const positionNode = (nodeId, x, depth) => {
            positions.set(nodeId, { x, y: -depth * this.verticalSpacing });

            const children = childrenMap.get(nodeId) || [];
            if (children.length === 0) return;

            // Position children with fixed spacing, centered under parent
            const totalWidth = (children.length - 1) * this.horizontalSpacing;
            let currentX = x - totalWidth / 2;

            for (const childId of children) {
                positionNode(childId, currentX, depth + 1);
                currentX += this.horizontalSpacing;
            }
        };
        positionNode(rootId, 0, 0);

        // Pass 2: Fix overlaps by shifting subtrees
        this._fixOverlaps(rootId, 0, positions, childrenMap);

        return positions;
    }

    /**
     * Compute layout for a subset of visible nodes
     *
     * This is useful for collapsible trees where only some nodes are visible.
     *
     * @param {Set<string>} visibleNodes - Set of visible node IDs
     * @param {Function} getChildren - Function to get child IDs: (nodeId) => childIds[]
     * @param {Function} isExpanded - Function to check if node is expanded: (nodeId) => boolean
     * @param {string} [rootId='0'] - Root node ID
     * @returns {Map<string, {x: number, y: number}>} Map of node positions
     */
    computeVisibleLayout(visibleNodes, getChildren, isExpanded, rootId = '0') {
        const positions = new Map();

        // Build visible children map
        const visibleChildren = new Map();
        for (const nodeId of visibleNodes) {
            if (isExpanded(nodeId)) {
                const children = getChildren(nodeId).filter(c => visibleNodes.has(String(c)));
                visibleChildren.set(nodeId, children.map(String));
            } else {
                visibleChildren.set(nodeId, []);
            }
        }

        // Pass 1: Initial positioning
        const positionNode = (nodeId, x, depth) => {
            positions.set(nodeId, { x, y: -depth * this.verticalSpacing });

            const children = visibleChildren.get(nodeId) || [];
            if (children.length === 0) return;

            const totalWidth = (children.length - 1) * this.horizontalSpacing;
            let currentX = x - totalWidth / 2;

            for (const childId of children) {
                positionNode(childId, currentX, depth + 1);
                currentX += this.horizontalSpacing;
            }
        };
        positionNode(rootId, 0, 0);

        // Pass 2: Fix overlaps
        this._fixOverlapsWithMap(rootId, 0, positions, visibleChildren);

        return positions;
    }

    /**
     * Fix overlapping subtrees by shifting them apart
     * @private
     */
    _fixOverlaps(nodeId, depth, positions, childrenMap) {
        const children = childrenMap.get(nodeId) || [];

        // First, recursively fix children's subtrees
        for (const childId of children) {
            this._fixOverlaps(childId, depth + 1, positions, childrenMap);
        }

        // Then check for overlaps between adjacent siblings
        for (let i = 1; i < children.length; i++) {
            const leftChild = children[i - 1];
            const rightChild = children[i];

            // Get right contour of left subtree
            const leftContour = new Map();
            this._getContour(leftChild, depth + 1, leftContour, 'right', positions, childrenMap);

            // Get left contour of right subtree
            const rightContour = new Map();
            this._getContour(rightChild, depth + 1, rightContour, 'left', positions, childrenMap);

            // Find max overlap across all depths
            let maxOverlap = 0;
            for (const [d, leftX] of leftContour) {
                if (rightContour.has(d)) {
                    const rightX = rightContour.get(d);
                    const overlap = leftX + this.minNodeWidth - rightX;
                    maxOverlap = Math.max(maxOverlap, overlap);
                }
            }

            // Shift right subtree if there's overlap
            if (maxOverlap > 0) {
                this._shiftSubtree(rightChild, maxOverlap, positions, childrenMap);
            }
        }

        // Center parent over its children
        if (children.length > 0) {
            const firstChild = positions.get(children[0]);
            const lastChild = positions.get(children[children.length - 1]);
            const parentPos = positions.get(nodeId);
            if (firstChild && lastChild && parentPos) {
                parentPos.x = (firstChild.x + lastChild.x) / 2;
            }
        }
    }

    /**
     * Fix overlaps using a pre-built children map
     * @private
     */
    _fixOverlapsWithMap(nodeId, depth, positions, childrenMap) {
        const children = childrenMap.get(nodeId) || [];

        // First, recursively fix children's subtrees
        for (const childId of children) {
            this._fixOverlapsWithMap(childId, depth + 1, positions, childrenMap);
        }

        // Then check for overlaps between adjacent siblings
        for (let i = 1; i < children.length; i++) {
            const leftChild = children[i - 1];
            const rightChild = children[i];

            // Get right contour of left subtree
            const leftContour = new Map();
            this._getContourFromMap(leftChild, depth + 1, leftContour, 'right', positions, childrenMap);

            // Get left contour of right subtree
            const rightContour = new Map();
            this._getContourFromMap(rightChild, depth + 1, rightContour, 'left', positions, childrenMap);

            // Find max overlap across all depths
            let maxOverlap = 0;
            for (const [d, leftX] of leftContour) {
                if (rightContour.has(d)) {
                    const rightX = rightContour.get(d);
                    const overlap = leftX + this.minNodeWidth - rightX;
                    maxOverlap = Math.max(maxOverlap, overlap);
                }
            }

            // Shift right subtree if there's overlap
            if (maxOverlap > 0) {
                this._shiftSubtreeFromMap(rightChild, maxOverlap, positions, childrenMap);
            }
        }

        // Center parent over its children
        if (children.length > 0) {
            const firstChild = positions.get(children[0]);
            const lastChild = positions.get(children[children.length - 1]);
            const parentPos = positions.get(nodeId);
            if (firstChild && lastChild && parentPos) {
                parentPos.x = (firstChild.x + lastChild.x) / 2;
            }
        }
    }

    /**
     * Get the contour (leftmost or rightmost x at each depth) for a subtree
     * @private
     */
    _getContour(nodeId, depth, contour, side, positions, childrenMap) {
        const pos = positions.get(nodeId);
        if (!pos) return;

        if (!contour.has(depth)) {
            contour.set(depth, pos.x);
        } else {
            contour.set(depth, side === 'left'
                ? Math.min(contour.get(depth), pos.x)
                : Math.max(contour.get(depth), pos.x));
        }

        const children = childrenMap.get(nodeId) || [];
        for (const childId of children) {
            this._getContour(childId, depth + 1, contour, side, positions, childrenMap);
        }
    }

    /**
     * Get contour using pre-built children map
     * @private
     */
    _getContourFromMap(nodeId, depth, contour, side, positions, childrenMap) {
        const pos = positions.get(nodeId);
        if (!pos) return;

        if (!contour.has(depth)) {
            contour.set(depth, pos.x);
        } else {
            contour.set(depth, side === 'left'
                ? Math.min(contour.get(depth), pos.x)
                : Math.max(contour.get(depth), pos.x));
        }

        const children = childrenMap.get(nodeId) || [];
        for (const childId of children) {
            this._getContourFromMap(childId, depth + 1, contour, side, positions, childrenMap);
        }
    }

    /**
     * Shift a subtree horizontally
     * @private
     */
    _shiftSubtree(nodeId, dx, positions, childrenMap) {
        const pos = positions.get(nodeId);
        if (pos) {
            pos.x += dx;
        }
        const children = childrenMap.get(nodeId) || [];
        for (const childId of children) {
            this._shiftSubtree(childId, dx, positions, childrenMap);
        }
    }

    /**
     * Shift subtree using pre-built children map
     * @private
     */
    _shiftSubtreeFromMap(nodeId, dx, positions, childrenMap) {
        const pos = positions.get(nodeId);
        if (pos) {
            pos.x += dx;
        }
        const children = childrenMap.get(nodeId) || [];
        for (const childId of children) {
            this._shiftSubtreeFromMap(childId, dx, positions, childrenMap);
        }
    }

    /**
     * Calculate the bounding box of the tree layout
     * @param {Map<string, {x: number, y: number}>} positions - Node positions
     * @returns {Object} Bounding box { minX, minY, maxX, maxY, width, height }
     */
    getBounds(positions) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const { x, y } of positions.values()) {
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        return {
            minX,
            minY,
            maxX,
            maxY,
            width: maxX - minX,
            height: maxY - minY
        };
    }

    /**
     * Center the tree around origin (0, 0)
     * @param {Map<string, {x: number, y: number}>} positions - Node positions (modified in place)
     */
    centerLayout(positions) {
        const bounds = this.getBounds(positions);
        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;

        for (const pos of positions.values()) {
            pos.x -= centerX;
            pos.y -= centerY;
        }
    }
}

// Expose as global for backward compatibility
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.TreeLayoutEngine = TreeLayoutEngine;
}
