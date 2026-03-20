/**
 * PlaybackController - Unified step-through animation controller
 *
 * A reusable module for managing algorithm visualization playback.
 * Uses callbacks for decoupled rendering - no direct UI dependencies.
 *
 * @example
 * const playback = new PlaybackController({
 *   onRenderStep: (step, index) => ui.renderStep(step),
 *   onPlayStateChange: (isPlaying) => updatePlayButton(isPlaying),
 *   onStepChange: (index, total) => updateStepDisplay(index, total),
 *   onFinished: () => showCompletionMessage()
 * });
 *
 * playback.load(algorithmSteps);
 * playback.play();
 */
export class PlaybackController {
    /**
     * @param {Object} options - Configuration options
     * @param {number} [options.initialSpeed=5] - Initial playback speed (1-10)
     * @param {Function} [options.getDelayFn] - Custom delay function: (speed) => ms. Overrides default delay mapping.
     * @param {Function} [options.onRenderStep] - Called when a step should be rendered: (step, index, metadata) => void
     * @param {Function} [options.onPlayStateChange] - Called when play state changes: (isPlaying) => void
     * @param {Function} [options.onStepChange] - Called when step index changes: (index, total) => void
     * @param {Function} [options.onFinished] - Called when playback reaches the end: () => void
     * @param {Function} [options.onReset] - Called when playback is reset: () => void
     */
    constructor(options = {}) {
        this.steps = [];
        this.currentStepIndex = -1;
        this.isPlaying = false;
        this.speed = options.initialSpeed ?? 5;
        this.playbackTimer = null;
        this.metadata = null; // Optional metadata (e.g., goal position)
        this.getDelayFn = options.getDelayFn ?? null;

        // Callbacks for decoupled rendering
        this.onRenderStep = options.onRenderStep ?? (() => {});
        this.onPlayStateChange = options.onPlayStateChange ?? (() => {});
        this.onStepChange = options.onStepChange ?? (() => {});
        this.onFinished = options.onFinished ?? (() => {});
        this.onReset = options.onReset ?? (() => {});
    }

    /**
     * Load steps for playback
     * @param {Array} steps - Array of step objects to play through
     * @param {*} [metadata] - Optional metadata to associate with this playback session
     */
    load(steps, metadata = null) {
        this.pause();
        this.steps = steps || [];
        this.metadata = metadata;
        this.currentStepIndex = -1;
        this.onStepChange(this.currentStepIndex, this.steps.length);
    }

    /**
     * Start or resume playback
     */
    play() {
        if (this.steps.length === 0) return;

        const wasPlaying = this.isPlaying;
        this.isPlaying = true;

        if (!wasPlaying) {
            this.onPlayStateChange(true);
        }

        this._playNext();
    }

    /**
     * Pause playback
     */
    pause() {
        const wasPlaying = this.isPlaying;
        this.isPlaying = false;

        if (this.playbackTimer) {
            clearTimeout(this.playbackTimer);
            this.playbackTimer = null;
        }

        if (wasPlaying) {
            this.onPlayStateChange(false);
        }
    }

    /**
     * Toggle between play and pause
     * @returns {boolean} New playing state
     */
    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
        return this.isPlaying;
    }

    /**
     * Advance to the next step
     * @returns {boolean} True if advanced, false if at end
     */
    stepForward() {
        if (this.currentStepIndex < this.steps.length - 1) {
            this.currentStepIndex++;
            this._renderCurrentStep();
            this.onStepChange(this.currentStepIndex, this.steps.length);
            return true;
        }
        return false;
    }

    /**
     * Go back to the previous step
     * @returns {boolean} True if moved back, false if at start
     */
    stepBackward() {
        if (this.currentStepIndex > 0) {
            this.currentStepIndex--;
            this._renderCurrentStep();
            this.onStepChange(this.currentStepIndex, this.steps.length);
            return true;
        }
        return false;
    }

    /**
     * Jump to a specific step
     * @param {number} index - Step index to jump to
     * @returns {boolean} True if jumped, false if invalid index
     */
    goToStep(index) {
        if (index >= 0 && index < this.steps.length) {
            this.currentStepIndex = index;
            this._renderCurrentStep();
            this.onStepChange(this.currentStepIndex, this.steps.length);
            return true;
        }
        return false;
    }

    /**
     * Go to the first step
     */
    goToStart() {
        if (this.steps.length > 0) {
            this.goToStep(0);
        }
    }

    /**
     * Go to the last step
     */
    goToEnd() {
        if (this.steps.length > 0) {
            this.goToStep(this.steps.length - 1);
        }
    }

    /**
     * Reset playback to initial state
     */
    reset() {
        this.pause();
        this.currentStepIndex = -1;
        this.onStepChange(this.currentStepIndex, this.steps.length);
        this.onReset();
    }

    /**
     * Set playback speed
     * @param {number} speed - Speed value (1-10 for default, or custom range with getDelayFn)
     */
    setSpeed(speed) {
        this.speed = this.getDelayFn ? speed : Math.max(1, Math.min(10, speed));
    }

    /**
     * Get current playback speed
     * @returns {number} Current speed (1-10)
     */
    getSpeed() {
        return this.speed;
    }

    /**
     * Get delay between steps based on current speed
     * @returns {number} Delay in milliseconds
     */
    getDelay() {
        if (this.getDelayFn) {
            return this.getDelayFn(this.speed);
        }
        // Speed 1 = 1000ms, Speed 10 = 50ms
        return 1050 - (this.speed * 100);
    }

    /**
     * Get the current step object
     * @returns {*} Current step or null if no step selected
     */
    getCurrentStep() {
        if (this.currentStepIndex >= 0 && this.currentStepIndex < this.steps.length) {
            return this.steps[this.currentStepIndex];
        }
        return null;
    }

    /**
     * Get current step index
     * @returns {number} Current step index (-1 if not started)
     */
    getCurrentIndex() {
        return this.currentStepIndex;
    }

    /**
     * Get total number of steps
     * @returns {number} Total step count
     */
    getStepCount() {
        return this.steps.length;
    }

    /**
     * Check if at the end of playback
     * @returns {boolean} True if at last step
     */
    isAtEnd() {
        return this.currentStepIndex >= this.steps.length - 1;
    }

    /**
     * Check if at the start (before first step)
     * @returns {boolean} True if before first step
     */
    isAtStart() {
        return this.currentStepIndex <= 0;
    }

    /**
     * Check if playback is currently running
     * @returns {boolean} True if playing
     */
    getIsPlaying() {
        return this.isPlaying;
    }

    /**
     * Get metadata associated with current playback
     * @returns {*} Metadata object or null
     */
    getMetadata() {
        return this.metadata;
    }

    // Private methods

    /**
     * Internal: Play the next step with delay
     * @private
     */
    _playNext() {
        if (!this.isPlaying) return;

        if (this.stepForward()) {
            // More steps available, schedule next
            const delay = this.getDelay();
            this.playbackTimer = setTimeout(() => this._playNext(), delay);
        } else {
            // Reached the end
            this.isPlaying = false;
            this.onPlayStateChange(false);
            this.onFinished();
        }
    }

    /**
     * Internal: Render the current step via callback
     * @private
     */
    _renderCurrentStep() {
        const step = this.getCurrentStep();
        if (step) {
            this.onRenderStep(step, this.currentStepIndex, this.metadata);
        }
    }
}

// Expose as global for backward compatibility with IIFE-based visualizations
if (typeof window !== 'undefined') {
    window.VizLib = window.VizLib || {};
    window.VizLib.PlaybackController = PlaybackController;
}
