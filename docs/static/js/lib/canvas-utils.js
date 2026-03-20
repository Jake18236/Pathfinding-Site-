/**
 * CanvasUtils - Utilities for working with HTML Canvas
 *
 * Provides helpers for high-DPI (Retina) canvas setup, common
 * drawing operations, and canvas state management.
 *
 * @example
 * import { setupHiDPICanvas, clearCanvas } from './lib/canvas-utils.js';
 *
 * const canvas = document.getElementById('my-canvas');
 * const { ctx, dpr, logicalWidth, logicalHeight } = setupHiDPICanvas(canvas);
 *
 * // Draw with crisp rendering on Retina displays
 * ctx.fillRect(0, 0, 100, 100);
 */

/**
 * Set up a canvas for high-DPI (Retina) displays
 *
 * This function scales the canvas internal resolution to match the device
 * pixel ratio, ensuring crisp rendering on high-DPI displays while
 * maintaining the visual CSS size.
 *
 * @param {HTMLCanvasElement} canvas - The canvas element to set up
 * @returns {Object} Canvas context and dimensions
 * @returns {CanvasRenderingContext2D} returns.ctx - The 2D rendering context
 * @returns {number} returns.dpr - Device pixel ratio
 * @returns {number} returns.logicalWidth - CSS width in pixels
 * @returns {number} returns.logicalHeight - CSS height in pixels
 */
export function setupHiDPICanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    // Store logical dimensions (CSS size) for calculations
    const logicalWidth = rect.width;
    const logicalHeight = rect.height;

    // Scale canvas internal resolution by device pixel ratio for crisp rendering
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Force the canvas to display at its original CSS size
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');

    return {
        ctx,
        dpr,
        logicalWidth,
        logicalHeight
    };
}

/**
 * Reset the canvas transform to account for device pixel ratio
 *
 * Call this at the start of each render frame to ensure proper scaling.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} dpr - Device pixel ratio
 */
export function resetCanvasTransform(ctx, dpr) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/**
 * Clear the canvas with an optional background color
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} width - Logical width
 * @param {number} height - Logical height
 * @param {string} [bgColor=null] - Optional background color (null for transparent)
 */
export function clearCanvas(ctx, width, height, bgColor = null) {
    ctx.clearRect(0, 0, width, height);
    if (bgColor) {
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, width, height);
    }
}

/**
 * Draw a rounded rectangle
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - X position
 * @param {number} y - Y position
 * @param {number} width - Rectangle width
 * @param {number} height - Rectangle height
 * @param {number} radius - Corner radius
 */
export function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

/**
 * Draw a line with optional arrow head
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x1 - Start X
 * @param {number} y1 - Start Y
 * @param {number} x2 - End X
 * @param {number} y2 - End Y
 * @param {Object} [options={}] - Drawing options
 * @param {boolean} [options.arrow=false] - Draw arrow at end
 * @param {number} [options.arrowSize=10] - Arrow head size
 */
export function drawLine(ctx, x1, y1, x2, y2, options = {}) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    if (options.arrow) {
        const arrowSize = options.arrowSize || 10;
        const angle = Math.atan2(y2 - y1, x2 - x1);

        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(
            x2 - arrowSize * Math.cos(angle - Math.PI / 6),
            y2 - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            x2 - arrowSize * Math.cos(angle + Math.PI / 6),
            y2 - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
    }
}

/**
 * Draw a circle
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - Center X
 * @param {number} y - Center Y
 * @param {number} radius - Circle radius
 * @param {Object} [options={}] - Drawing options
 * @param {boolean} [options.fill=true] - Fill the circle
 * @param {boolean} [options.stroke=false] - Stroke the circle
 */
export function drawCircle(ctx, x, y, radius, options = {}) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);

    if (options.fill !== false) {
        ctx.fill();
    }
    if (options.stroke) {
        ctx.stroke();
    }
}

/**
 * Get mouse position relative to canvas in logical coordinates
 *
 * @param {HTMLCanvasElement} canvas - The canvas element
 * @param {MouseEvent} event - The mouse event
 * @returns {Object} Mouse position { x, y }
 */
export function getMousePosition(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

/**
 * Create an offscreen canvas for caching or pre-rendering
 *
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @param {number} [dpr=1] - Device pixel ratio for high-DPI
 * @returns {Object} Canvas and context { canvas, ctx }
 */
export function createOffscreenCanvas(width, height, dpr = 1) {
    const canvas = document.createElement('canvas');
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    return { canvas, ctx };
}

/**
 * Map between data-space coordinates and canvas-space coordinates.
 *
 * Handles the common pattern of a plot area within a canvas with padding,
 * where data values in [dataMinX..dataMaxX] × [dataMinY..dataMaxY] map to
 * a rectangular region on the canvas.
 */
export class CoordinateMapper {
    /**
     * @param {number} plotLeft   - Canvas X of the plot area's left edge
     * @param {number} plotTop    - Canvas Y of the plot area's top edge
     * @param {number} plotWidth  - Width of the plot area in canvas pixels
     * @param {number} plotHeight - Height of the plot area in canvas pixels
     * @param {number} dataMinX   - Minimum data X value
     * @param {number} dataMaxX   - Maximum data X value
     * @param {number} [dataMinY] - Minimum data Y value (defaults to dataMinX)
     * @param {number} [dataMaxY] - Maximum data Y value (defaults to dataMaxX)
     */
    constructor(plotLeft, plotTop, plotWidth, plotHeight, dataMinX, dataMaxX, dataMinY, dataMaxY) {
        this.plotLeft = plotLeft;
        this.plotTop = plotTop;
        this.plotWidth = plotWidth;
        this.plotHeight = plotHeight;
        this.dataMinX = dataMinX;
        this.dataMaxX = dataMaxX;
        this.dataMinY = dataMinY !== undefined ? dataMinY : dataMinX;
        this.dataMaxY = dataMaxY !== undefined ? dataMaxY : dataMaxX;
    }

    /** Convert data coordinates to canvas coordinates (Y axis is flipped) */
    dataToCanvas(dx, dy) {
        const cx = this.plotLeft + ((dx - this.dataMinX) / (this.dataMaxX - this.dataMinX)) * this.plotWidth;
        const cy = this.plotTop + this.plotHeight - ((dy - this.dataMinY) / (this.dataMaxY - this.dataMinY)) * this.plotHeight;
        return { x: cx, y: cy };
    }

    /** Convert canvas coordinates to data coordinates (Y axis is flipped) */
    canvasToData(cx, cy) {
        const dx = this.dataMinX + ((cx - this.plotLeft) / this.plotWidth) * (this.dataMaxX - this.dataMinX);
        const dy = this.dataMinY + ((this.plotTop + this.plotHeight - cy) / this.plotHeight) * (this.dataMaxY - this.dataMinY);
        return { x: dx, y: dy };
    }
}

/**
 * Parse a hex color string to [r, g, b] array
 *
 * @param {string} hex - Hex color string (e.g. '#ff0000' or '#f00')
 * @returns {number[]} RGB values [0-255]
 */
export function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    const n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Linearly interpolate between two [r,g,b] color arrays
 *
 * @param {number[]} c1 - Start color [r, g, b]
 * @param {number[]} c2 - End color [r, g, b]
 * @param {number} t - Interpolation factor (0 = c1, 1 = c2)
 * @returns {number[]} Interpolated [r, g, b]
 */
export function colorLerp(c1, c2, t) {
    return [
        Math.round(c1[0] + (c2[0] - c1[0]) * t),
        Math.round(c1[1] + (c2[1] - c1[1]) * t),
        Math.round(c1[2] + (c2[2] - c1[2]) * t)
    ];
}

// Expose as global for backward compatibility
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.CanvasUtils = {
        setupHiDPICanvas,
        resetCanvasTransform,
        clearCanvas,
        roundRect,
        drawLine,
        drawCircle,
        getMousePosition,
        createOffscreenCanvas,
        CoordinateMapper,
        hexToRgb,
        colorLerp
    };
}
