/**
 * Quadtree - Spatial indexing for efficient 2D point queries
 *
 * A quadtree recursively subdivides 2D space into four quadrants,
 * enabling O(log n) nearest neighbor and range queries.
 *
 * @example
 * const qt = new Quadtree({
 *   bounds: { x: 0, y: 0, width: 800, height: 600 },
 *   capacity: 8
 * });
 *
 * // Insert points
 * qt.insert({ x: 100, y: 200, data: { id: 1 } });
 *
 * // Query points in radius
 * const nearby = qt.queryRadius(150, 200, 50);
 */
export class Quadtree {
    /**
     * Create a new Quadtree
     * @param {Object} bounds - Bounding rectangle { x, y, width, height }
     * @param {number} [capacity=8] - Maximum points per node before subdividing
     */
    constructor(bounds, capacity) {
        this.bounds = bounds;
        this.capacity = capacity || 8;

        this.points = [];
        this.divided = false;
        this.ne = null;  // Northeast
        this.nw = null;  // Northwest
        this.se = null;  // Southeast
        this.sw = null;  // Southwest
    }

    /**
     * Insert a point into the quadtree
     * @param {Object} point - Point with x, y coordinates and optional data
     * @returns {boolean} True if inserted successfully
     */
    insert(point) {
        if (!this.contains(point)) {
            return false;
        }

        if (this.points.length < this.capacity && !this.divided) {
            this.points.push(point);
            return true;
        }

        if (!this.divided) {
            this.subdivide();
        }

        return this.ne.insert(point) ||
               this.nw.insert(point) ||
               this.se.insert(point) ||
               this.sw.insert(point);
    }

    /**
     * Check if a point is within this node's bounds
     * @param {Object} point - Point with x, y coordinates
     * @returns {boolean} True if point is within bounds
     */
    contains(point) {
        return point.x >= this.bounds.x &&
               point.x < this.bounds.x + this.bounds.width &&
               point.y >= this.bounds.y &&
               point.y < this.bounds.y + this.bounds.height;
    }

    /**
     * Subdivide this node into four quadrants
     * @private
     */
    subdivide() {
        const { x, y, width, height } = this.bounds;
        const w = width / 2;
        const h = height / 2;

        this.ne = new Quadtree({ x: x + w, y, width: w, height: h }, this.capacity);
        this.nw = new Quadtree({ x, y, width: w, height: h }, this.capacity);
        this.se = new Quadtree({ x: x + w, y: y + h, width: w, height: h }, this.capacity);
        this.sw = new Quadtree({ x, y: y + h, width: w, height: h }, this.capacity);
        this.divided = true;
    }

    /**
     * Query all points within a circular radius
     * @param {number} cx - Center X coordinate
     * @param {number} cy - Center Y coordinate
     * @param {number} radius - Search radius
     * @returns {Array} Array of points within the radius
     */
    queryRadius(cx, cy, radius) {
        const found = [];

        if (!this.intersectsCircle(cx, cy, radius)) {
            return found;
        }

        for (const p of this.points) {
            if (Math.hypot(p.x - cx, p.y - cy) <= radius) {
                found.push(p);
            }
        }

        if (this.divided) {
            found.push(...this.ne.queryRadius(cx, cy, radius));
            found.push(...this.nw.queryRadius(cx, cy, radius));
            found.push(...this.se.queryRadius(cx, cy, radius));
            found.push(...this.sw.queryRadius(cx, cy, radius));
        }

        return found;
    }

    /**
     * Query all points within a rectangular region
     * @param {Object} rect - Rectangle { x, y, width, height }
     * @returns {Array} Array of points within the rectangle
     */
    queryRect(rect) {
        const found = [];

        if (!this.intersectsRect(rect)) {
            return found;
        }

        for (const p of this.points) {
            if (p.x >= rect.x && p.x < rect.x + rect.width &&
                p.y >= rect.y && p.y < rect.y + rect.height) {
                found.push(p);
            }
        }

        if (this.divided) {
            found.push(...this.ne.queryRect(rect));
            found.push(...this.nw.queryRect(rect));
            found.push(...this.se.queryRect(rect));
            found.push(...this.sw.queryRect(rect));
        }

        return found;
    }

    /**
     * Find the nearest point to given coordinates
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} [maxRadius=Infinity] - Maximum search radius
     * @returns {Object|null} Nearest point or null if none found
     */
    findNearest(x, y, maxRadius = Infinity) {
        let nearest = null;
        let nearestDist = maxRadius;

        const search = (node) => {
            if (!node) return;

            // Check if this node could contain a closer point
            const { bounds } = node;
            const closestX = Math.max(bounds.x, Math.min(x, bounds.x + bounds.width));
            const closestY = Math.max(bounds.y, Math.min(y, bounds.y + bounds.height));
            if (Math.hypot(x - closestX, y - closestY) > nearestDist) {
                return;
            }

            // Check points in this node
            for (const p of node.points) {
                const dist = Math.hypot(p.x - x, p.y - y);
                if (dist < nearestDist) {
                    nearestDist = dist;
                    nearest = p;
                }
            }

            // Recursively search children
            if (node.divided) {
                search(node.ne);
                search(node.nw);
                search(node.se);
                search(node.sw);
            }
        };

        search(this);
        return nearest;
    }

    /**
     * Check if this node's bounds intersect a circle
     * @private
     */
    intersectsCircle(cx, cy, r) {
        const { x, y, width, height } = this.bounds;
        const closestX = Math.max(x, Math.min(cx, x + width));
        const closestY = Math.max(y, Math.min(cy, y + height));
        return Math.hypot(cx - closestX, cy - closestY) <= r;
    }

    /**
     * Check if this node's bounds intersect a rectangle
     * @private
     */
    intersectsRect(rect) {
        return !(rect.x > this.bounds.x + this.bounds.width ||
                 rect.x + rect.width < this.bounds.x ||
                 rect.y > this.bounds.y + this.bounds.height ||
                 rect.y + rect.height < this.bounds.y);
    }

    /**
     * Clear all points from the quadtree
     */
    clear() {
        this.points = [];
        this.divided = false;
        this.ne = null;
        this.nw = null;
        this.se = null;
        this.sw = null;
    }

    /**
     * Get total number of points in the tree
     * @returns {number} Total point count
     */
    size() {
        let count = this.points.length;
        if (this.divided) {
            count += this.ne.size() + this.nw.size() + this.se.size() + this.sw.size();
        }
        return count;
    }
}

// Expose as global for backward compatibility
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.Quadtree = Quadtree;
}
