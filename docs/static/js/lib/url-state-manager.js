/**
 * URL State Manager
 *
 * Auto-discovers form controls in #main-content and syncs their values
 * to/from URL search params, enabling shareable visualization state.
 *
 * Usage:
 *   const urlState = new URLStateManager();
 *   urlState.applyFromURL();
 *   urlState.enableAutoSync();
 *   urlState.bindShareButton();
 */

const SUFFIX_PATTERN = /-(select|slider|input|checkbox|range|stepper|toggle)$/;
const EXCLUDED_IDS = ['speed-slider', 'theme-toggle'];

export class URLStateManager {
    constructor(containerSelector = '#main-content') {
        this.container = document.querySelector(containerSelector);
        this.controls = new Map();      // paramName -> {element, type}
        this.customHandlers = new Map(); // paramName -> {getter, setter}
        this._syncTimer = null;
        this._listening = false;

        if (this.container) {
            this._discoverControls();
        }
    }

    /**
     * Auto-discover standard form controls inside the container.
     */
    _discoverControls() {
        const selectors = 'select, input[type="range"], input[type="checkbox"], input[type="number"]';
        const elements = this.container.querySelectorAll(selectors);

        for (const el of elements) {
            if (!el.id) continue;
            if (EXCLUDED_IDS.includes(el.id)) continue;

            const paramName = el.id.replace(SUFFIX_PATTERN, '');
            const type = el.type === 'checkbox' ? 'checkbox' : 'value';

            this.controls.set(paramName, { element: el, type, id: el.id });
        }
    }

    /**
     * Register a custom handler for non-standard controls.
     * @param {string} paramName - URL param key
     * @param {Function} getter - Returns current value as string
     * @param {Function} setter - Receives string value, applies to control
     */
    registerCustom(paramName, getter, setter) {
        this.customHandlers.set(paramName, { getter, setter });
    }

    /**
     * Read URL search params and apply values to controls.
     * Dispatches native DOM events so existing viz listeners fire.
     */
    applyFromURL() {
        const params = new URLSearchParams(window.location.search);
        if (params.size === 0) return;

        for (const [paramName, value] of params) {
            // Check custom handlers first
            if (this.customHandlers.has(paramName)) {
                this.customHandlers.get(paramName).setter(value);
                continue;
            }

            // Check auto-discovered controls
            const control = this.controls.get(paramName);
            if (!control) continue;

            const { element, type } = control;

            if (type === 'checkbox') {
                const checked = value === '1' || value === 'true';
                if (element.checked !== checked) {
                    element.checked = checked;
                    element.dispatchEvent(new Event('change', { bubbles: true }));
                }
            } else {
                if (element.value !== value) {
                    element.value = value;
                    // Dispatch both input and change for range sliders
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }
    }

    /**
     * Read current control values and build URL search params.
     * @returns {URLSearchParams}
     */
    _buildParams() {
        const params = new URLSearchParams();

        for (const [paramName, control] of this.controls) {
            const { element, type } = control;
            if (type === 'checkbox') {
                if (element.checked) {
                    params.set(paramName, '1');
                }
            } else {
                const val = element.value;
                // Only include non-default / non-empty values
                if (val !== '' && val !== undefined) {
                    params.set(paramName, val);
                }
            }
        }

        for (const [paramName, handler] of this.customHandlers) {
            const val = handler.getter();
            if (val !== null && val !== undefined && val !== '') {
                params.set(paramName, val);
            }
        }

        return params;
    }

    /**
     * Update URL with current state using replaceState (no history pollution).
     */
    _pushState() {
        const params = this._buildParams();
        const qs = params.toString();
        const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
        history.replaceState(null, '', url);
    }

    /**
     * Debounced state push.
     */
    _debouncedPush() {
        clearTimeout(this._syncTimer);
        this._syncTimer = setTimeout(() => this._pushState(), 300);
    }

    /**
     * Start listening for control changes and syncing to URL.
     */
    enableAutoSync() {
        if (this._listening) return;
        this._listening = true;

        const handler = () => this._debouncedPush();

        for (const [, control] of this.controls) {
            control.element.addEventListener('change', handler);
            if (control.element.type === 'range') {
                control.element.addEventListener('input', handler);
            }
        }

        // Also push state immediately to capture initial state
        this._pushState();
    }

    /**
     * Get current shareable URL.
     * @returns {string}
     */
    getShareableURL() {
        const params = this._buildParams();
        const qs = params.toString();
        const base = `${window.location.origin}${window.location.pathname}`;
        return qs ? `${base}?${qs}` : base;
    }

    /**
     * Bind a share button to copy the shareable URL to clipboard.
     * Looks for #share-btn or accepts a custom selector.
     */
    bindShareButton(selector = '#share-btn') {
        const btn = document.querySelector(selector);
        if (!btn) return;

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const url = this.getShareableURL();

            navigator.clipboard.writeText(url).then(() => {
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fa fa-check"></i> Copied!';
                btn.classList.add('share-btn--copied');
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.classList.remove('share-btn--copied');
                }, 2000);
            }).catch(() => {
                // Fallback for older browsers
                const input = document.createElement('input');
                input.value = url;
                document.body.appendChild(input);
                input.select();
                document.execCommand('copy');
                document.body.removeChild(input);

                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fa fa-check"></i> Copied!';
                btn.classList.add('share-btn--copied');
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.classList.remove('share-btn--copied');
                }, 2000);
            });
        });
    }
}
