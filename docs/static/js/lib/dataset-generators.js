/**
 * Dataset Generators - Reusable 2D dataset generators for classification visualizations
 *
 * All generators return arrays of {x, y, classLabel} points in [0, 1] space.
 *
 * @example
 * const points = DatasetGenerators.moons(200, 0.1);
 * const points = DatasetGenerators.blobs(150, 3);
 */

const MU = VizLib.MathUtils;

function noise(scale) {
    return (Math.random() - 0.5) * 2 * scale;
}

function gaussian() {
    return MU.gaussian();
}

function remap(val, min, max) {
    return MU.remapRange(val, min, max, 0, 1);
}

export const DatasetGenerators = {
    /**
     * Two interleaving half circles (sklearn make_moons style)
     * @param {number} n - Number of points
     * @param {number} [noiseLevel=0.1] - Noise magnitude
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    moons(n, noiseLevel = 0.1) {
        const points = [];
        const nPerClass = Math.floor(n / 2);

        for (let i = 0; i < nPerClass; i++) {
            const angle1 = Math.PI * i / nPerClass;
            const x1 = Math.cos(angle1) + noise(noiseLevel);
            const y1 = Math.sin(angle1) + noise(noiseLevel);
            points.push({
                x: remap(x1, -1.5, 2.5),
                y: remap(y1, -0.6, 1.6),
                classLabel: 0
            });

            const x2 = 1 - Math.cos(angle1) + noise(noiseLevel);
            const y2 = 0.5 - Math.sin(angle1) + noise(noiseLevel);
            points.push({
                x: remap(x2, -1.5, 2.5),
                y: remap(y2, -0.6, 1.6),
                classLabel: 1
            });
        }
        return points;
    },

    /**
     * Concentric circles
     * @param {number} n - Number of points
     * @param {number} [noiseLevel=0.05] - Noise magnitude
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    circles(n, noiseLevel = 0.05) {
        const points = [];
        const nPerClass = Math.floor(n / 2);
        const innerRadius = 0.25;
        const outerRadius = 0.5;

        for (let i = 0; i < nPerClass; i++) {
            const angle = 2 * Math.PI * Math.random();

            const r1 = innerRadius * (0.8 + 0.4 * Math.random()) + noise(noiseLevel);
            points.push({
                x: 0.5 + r1 * Math.cos(angle),
                y: 0.5 + r1 * Math.sin(angle),
                classLabel: 0
            });

            const r2 = outerRadius * (0.8 + 0.4 * Math.random()) + noise(noiseLevel);
            points.push({
                x: 0.5 + r2 * Math.cos(angle),
                y: 0.5 + r2 * Math.sin(angle),
                classLabel: 1
            });
        }
        return points;
    },

    /**
     * Gaussian blobs
     * @param {number} n - Number of points
     * @param {number} [numBlobs=3] - Number of clusters
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    blobs(n, numBlobs = 3) {
        const points = [];
        const centers = [
            { x: 0.25, y: 0.75 },
            { x: 0.75, y: 0.75 },
            { x: 0.5, y: 0.25 }
        ];
        const std = 0.08;
        const nPerBlob = Math.floor(n / numBlobs);

        for (let b = 0; b < numBlobs; b++) {
            for (let i = 0; i < nPerBlob; i++) {
                const x = centers[b].x + gaussian() * std;
                const y = centers[b].y + gaussian() * std;
                points.push({
                    x: Math.max(0.02, Math.min(0.98, x)),
                    y: Math.max(0.02, Math.min(0.98, y)),
                    classLabel: b
                });
            }
        }
        return points;
    },

    /**
     * Linearly separable points
     * @param {number} n - Number of points
     * @param {number} [noiseLevel=0.08] - Noise magnitude
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    linear(n, noiseLevel = 0.08) {
        const points = [];
        const nPerClass = Math.floor(n / 2);

        for (let i = 0; i < nPerClass; i++) {
            const x0 = 0.1 + 0.8 * Math.random();
            const maxY0 = x0 + 0.05;
            const y0 = Math.random() * maxY0 * 0.8 + noise(noiseLevel);
            points.push({
                x: x0,
                y: Math.max(0.05, Math.min(0.95, y0)),
                classLabel: 0
            });

            const x1 = 0.1 + 0.8 * Math.random();
            const minY1 = x1 + 0.15;
            const y1 = minY1 + (0.95 - minY1) * Math.random() + noise(noiseLevel);
            points.push({
                x: x1,
                y: Math.max(0.05, Math.min(0.95, y1)),
                classLabel: 1
            });
        }
        return points;
    },

    /**
     * XOR pattern (4 quadrants)
     * @param {number} n - Number of points
     * @param {number} [noiseLevel=0.05] - Noise magnitude
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    xor(n, noiseLevel = 0.05) {
        const points = [];
        const nPerQuadrant = Math.floor(n / 4);
        const quadrants = [
            { cx: 0.25, cy: 0.75, label: 0 },
            { cx: 0.75, cy: 0.75, label: 1 },
            { cx: 0.25, cy: 0.25, label: 1 },
            { cx: 0.75, cy: 0.25, label: 0 }
        ];

        for (const q of quadrants) {
            for (let i = 0; i < nPerQuadrant; i++) {
                points.push({
                    x: q.cx + (Math.random() - 0.5) * 0.35 + noise(noiseLevel),
                    y: q.cy + (Math.random() - 0.5) * 0.35 + noise(noiseLevel),
                    classLabel: q.label
                });
            }
        }
        return points;
    },

    /**
     * Two intertwining spirals
     * @param {number} n - Number of points
     * @param {number} [noiseLevel=0.03] - Noise magnitude
     * @returns {Array<{x: number, y: number, classLabel: number}>}
     */
    spiral(n, noiseLevel = 0.03) {
        const points = [];
        const nPerSpiral = Math.floor(n / 2);
        const turns = 1.5;

        for (let i = 0; i < nPerSpiral; i++) {
            const t = i / nPerSpiral;
            const angle = turns * 2 * Math.PI * t;
            const radius = 0.05 + 0.4 * t;

            points.push({
                x: 0.5 + radius * Math.cos(angle) + noise(noiseLevel),
                y: 0.5 + radius * Math.sin(angle) + noise(noiseLevel),
                classLabel: 0
            });

            points.push({
                x: 0.5 + radius * Math.cos(angle + Math.PI) + noise(noiseLevel),
                y: 0.5 + radius * Math.sin(angle + Math.PI) + noise(noiseLevel),
                classLabel: 1
            });
        }
        return points;
    }
};

// Expose as global for IIFE-based visualizations
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.DatasetGenerators = DatasetGenerators;
}
