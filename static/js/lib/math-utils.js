/**
 * Math Utilities - Shared mathematical functions for visualizations
 *
 * Extracted from entropy.js, cross-entropy.js, and knn.js to eliminate duplication.
 *
 * @example
 * import { normalize, clamp, euclideanDistance } from './math-utils.js';
 *
 * const probs = normalize([3, 1, 1]); // [0.6, 0.2, 0.2]
 * const val = clamp(1.5, 0, 1);       // 1
 */

/**
 * Normalize an array to sum to 1
 * @param {number[]} arr - Array of numbers
 * @returns {number[]} Normalized array summing to 1
 */
export function normalize(arr) {
    const sum = arr.reduce((a, b) => a + b, 0);
    if (sum === 0) return arr.map(() => 1 / arr.length);
    return arr.map(v => v / sum);
}

/**
 * Clamp a value between min and max
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/**
 * Euclidean distance between two 2D points
 * @param {Object} a - {x, y}
 * @param {Object} b - {x, y}
 * @returns {number}
 */
export function euclideanDistance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Manhattan distance between two 2D points
 * @param {Object} a - {x, y}
 * @param {Object} b - {x, y}
 * @returns {number}
 */
export function manhattanDistance(a, b) {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

/**
 * Chebyshev distance between two 2D points
 * @param {Object} a - {x, y}
 * @param {Object} b - {x, y}
 * @returns {number}
 */
export function chebyshevDistance(a, b) {
    return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
}

/**
 * Box-Muller transform for Gaussian random numbers
 * @returns {number} A sample from standard normal distribution
 */
export function gaussian() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Linear interpolation between two values
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Interpolation factor [0, 1]
 * @returns {number}
 */
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Remap a value from one range to another
 * @param {number} val - Value to remap
 * @param {number} inMin - Input range minimum
 * @param {number} inMax - Input range maximum
 * @param {number} outMin - Output range minimum
 * @param {number} outMax - Output range maximum
 * @returns {number}
 */
export function remapRange(val, inMin, inMax, outMin, outMax) {
    return outMin + ((val - inMin) / (inMax - inMin)) * (outMax - outMin);
}

// Expose as global for backward compatibility
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.MathUtils = {
        normalize,
        clamp,
        euclideanDistance,
        manhattanDistance,
        chebyshevDistance,
        gaussian,
        lerp,
        remapRange
    };
}
