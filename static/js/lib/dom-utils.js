/**
 * DomUtils - Common DOM helper functions for visualizations
 *
 * Provides reusable patterns for wiring up stepper controls, sliders,
 * and updating text content by element ID.
 */

/**
 * Set the textContent of an element by ID
 *
 * @param {string} id - Element ID
 * @param {string|number} value - Value to set
 */
export function setTextContent(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

/**
 * Wire a stepper control (minus button + value input + plus button)
 *
 * @param {string} minusId - ID of the minus button
 * @param {string} plusId - ID of the plus button
 * @param {string} valueId - ID of the value input
 * @param {Object} options
 * @param {number} options.min - Minimum value
 * @param {number} options.max - Maximum value
 * @param {number} [options.step=1] - Step size
 * @param {function} options.onChange - Callback with new value: onChange(newValue)
 */
export function wireStepper(minusId, plusId, valueId, { min, max, step = 1, onChange }) {
    const minusBtn = document.getElementById(minusId);
    const plusBtn = document.getElementById(plusId);
    const valueEl = document.getElementById(valueId);
    if (!minusBtn || !plusBtn || !valueEl) return;

    minusBtn.addEventListener('click', function() {
        const current = parseInt(valueEl.value) || min;
        if (current > min) {
            const next = Math.max(min, current - step);
            valueEl.value = next;
            if (onChange) onChange(next);
        }
    });

    plusBtn.addEventListener('click', function() {
        const current = parseInt(valueEl.value) || min;
        if (current < max) {
            const next = Math.min(max, current + step);
            valueEl.value = next;
            if (onChange) onChange(next);
        }
    });
}

/**
 * Wire a range slider with a display element that shows the current value
 *
 * @param {string} sliderId - ID of the range input
 * @param {string} displayId - ID of the element to display the value
 * @param {Object} [options]
 * @param {function} [options.format] - Format function: format(value) => string
 * @param {function} [options.onChange] - Callback with new value: onChange(value)
 */
export function wireSlider(sliderId, displayId, options = {}) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (!slider) return;

    slider.addEventListener('input', function() {
        const val = slider.value;
        if (display) {
            display.textContent = options.format ? options.format(val) : val;
        }
        if (options.onChange) options.onChange(val);
    });
}

// Expose as global for backward compatibility
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.DomUtils = {
        setTextContent,
        wireStepper,
        wireSlider
    };
}
