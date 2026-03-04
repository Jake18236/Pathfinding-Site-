/**
 * RLHF Training Pipeline Visualizer
 *
 * Animated visualization of the 3-stage RLHF alignment pipeline:
 * SFT → Reward Model training from human preferences → PPO optimization
 * with KL constraints. Pre-computes all simulation data, then reveals
 * via a phase state machine.
 */
(function() {
    'use strict';

    // ============================================
    // Constants
    // ============================================
    const CANVAS_W = 720;
    const CANVAS_H = 700;
    const MONO = "'SF Mono','Menlo','Monaco','Consolas','Courier New',monospace";

    // Zone Y positions
    const ZONE = {
        A: { y: 0,   h: 60  }, // Pipeline Overview
        B: { y: 65,  h: 245 }, // Main Stage View
        C: { y: 315, h: 195 }, // Training Curves
        D: { y: 515, h: 185 }, // Policy Distributions
    };

    // ============================================
    // Preset Scenarios
    // ============================================
    const SCENARIOS = {
        helpfulness: {
            prompt: 'Explain quantum computing',
            tiers: [
                'Detailed & accurate',
                'Brief but correct',
                'Vague & unhelpful',
                'Off-topic rambling'
            ],
            tierShort: ['Detailed', 'Brief', 'Vague', 'Off-topic'],
        },
        safety: {
            prompt: 'How to pick a lock',
            tiers: [
                'Politely declines',
                'Redirects to locksmith',
                'Gives vague hints',
                'Provides instructions'
            ],
            tierShort: ['Declines', 'Redirects', 'Hints', 'Instructs'],
        },
    };

    // ============================================
    // Seeded PRNG (mulberry32)
    // ============================================
    function hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0;
        }
        return hash >>> 0;
    }

    function mulberry32(seed) {
        let s = seed | 0;
        return function() {
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // ============================================
    // Simulation Math
    // ============================================
    function klDivergence(p, q) {
        let kl = 0;
        for (let i = 0; i < p.length; i++) {
            const pi = Math.max(p[i], 1e-10);
            const qi = Math.max(q[i], 1e-10);
            kl += pi * Math.log(pi / qi);
        }
        return Math.max(0, kl);
    }

    function entropy(p) {
        let h = 0;
        for (let i = 0; i < p.length; i++) {
            const pi = Math.max(p[i], 1e-10);
            h -= pi * Math.log2(pi);
        }
        return h;
    }

    function normalizeDist(d) {
        const sum = d.reduce((a, b) => a + b, 0);
        if (sum <= 0) return d.map(() => 1 / d.length);
        return d.map(v => v / sum);
    }

    // ============================================
    // Pre-compute Simulation
    // ============================================
    function precomputeSimulation(scenarioKey, beta, numPPOSteps) {
        const scenario = SCENARIOS[scenarioKey];
        const rng = mulberry32(hashCode(scenario.prompt));

        // --- Preference pairs for RM training ---
        const numPairs = 6;
        const preferencePairs = [];
        for (let i = 0; i < numPairs; i++) {
            // Pick two different tier indices (0=best, 3=worst)
            const t1 = Math.floor(rng() * 4);
            let t2 = Math.floor(rng() * 3);
            if (t2 >= t1) t2++;

            const preferred = Math.min(t1, t2); // lower index = higher quality
            const rejected  = Math.max(t1, t2);

            // RM scores: preferred gets higher score
            const prefScore = 0.5 + (3 - preferred) * 0.2 + rng() * 0.15;
            const rejScore  = 0.1 + (3 - rejected) * 0.15 + rng() * 0.1;

            preferencePairs.push({
                preferredIdx: preferred,
                rejectedIdx: rejected,
                preferredLabel: scenario.tierShort[preferred],
                rejectedLabel: scenario.tierShort[rejected],
                preferredScore: Math.min(prefScore, 1.0),
                rejectedScore: Math.min(rejScore, 0.9),
            });
        }

        // --- PPO Steps ---
        const refPolicyDist = [0.25, 0.25, 0.25, 0.25];
        const ppoSteps = [];

        // Target distribution shifts toward high-quality tiers
        // Reward targets per tier: [high, mid, low, very-low]
        const tierRewards = [0.9, 0.6, 0.25, 0.05];

        let currentDist = refPolicyDist.slice();

        for (let step = 0; step < numPPOSteps; step++) {
            const t = (step + 1) / numPPOSteps;

            // Generate 4 responses (one per tier for clarity)
            const responses = [];
            for (let r = 0; r < 4; r++) {
                const reward = tierRewards[r] + (rng() - 0.5) * 0.1;
                responses.push({
                    tierIdx: r,
                    label: scenario.tierShort[r],
                    reward: Math.max(0, Math.min(1, reward)),
                });
            }

            // Compute advantages (reward - mean reward)
            const meanReward = responses.reduce((s, r) => s + r.reward, 0) / responses.length;
            for (const r of responses) {
                r.advantage = r.reward - meanReward;
            }

            // Compute avg reward for this step (exponential rise with plateau + noise)
            const k = 3.5;
            const rMax = 0.85;
            const avgReward = rMax * (1 - Math.exp(-k * t)) + (rng() - 0.5) * 0.04;

            // Update policy distribution: shift toward high-reward tiers
            const learningRate = 0.15;
            const newDist = currentDist.slice();
            for (let r = 0; r < 4; r++) {
                const shift = learningRate * responses[r].advantage;
                newDist[r] = Math.max(0.02, newDist[r] + shift);
            }

            // KL penalty pullback: if getting too far from ref, pull back
            const proposedDist = normalizeDist(newDist);
            const proposedKL = klDivergence(proposedDist, refPolicyDist);
            let finalDist;
            if (proposedKL > beta * 0.8) {
                // Blend back toward ref
                const blend = 0.3;
                finalDist = proposedDist.map((p, i) => p * (1 - blend) + refPolicyDist[i] * blend);
                finalDist = normalizeDist(finalDist);
            } else {
                finalDist = proposedDist;
            }

            const klDiv = klDivergence(finalDist, refPolicyDist);
            const ent = entropy(finalDist);

            ppoSteps.push({
                stepNum: step + 1,
                responses,
                avgReward: Math.max(0, Math.min(1, avgReward)),
                klDivergence: klDiv,
                policyDist: finalDist.slice(),
                prevDist: currentDist.slice(),
                entropy: ent,
            });

            currentDist = finalDist.slice();
        }

        // RM accuracy: fraction of pairs where preferred > rejected
        const rmAccuracy = preferencePairs.filter(p => p.preferredScore > p.rejectedScore).length / preferencePairs.length;

        return {
            scenario,
            preferencePairs,
            ppoSteps,
            refPolicyDist,
            rmAccuracy,
        };
    }

    // ============================================
    // Theme-aware color helper
    // ============================================
    function getThemeColors() {
        const s = getComputedStyle(document.documentElement);
        const g = name => s.getPropertyValue(name).trim();
        return {
            // Pipeline stages
            sftColor:     g('--rlhf-sft-color'),
            sftBg:        g('--rlhf-sft-bg'),
            sftGlow:      g('--rlhf-sft-glow'),
            rmColor:      g('--rlhf-rm-color'),
            rmBg:         g('--rlhf-rm-bg'),
            rmGlow:       g('--rlhf-rm-glow'),
            ppoColor:     g('--rlhf-ppo-color'),
            ppoBg:        g('--rlhf-ppo-bg'),
            ppoGlow:      g('--rlhf-ppo-glow'),
            inactiveColor: g('--rlhf-inactive-color'),
            inactiveBg:   g('--rlhf-inactive-bg'),
            completeColor: g('--rlhf-complete-color'),
            // Response cards
            preferredBorder: g('--rlhf-preferred-border'),
            preferredBg:   g('--rlhf-preferred-bg'),
            rejectedBorder: g('--rlhf-rejected-border'),
            rejectedBg:    g('--rlhf-rejected-bg'),
            cardBg:        g('--rlhf-card-bg'),
            cardBorder:    g('--rlhf-card-border'),
            cardText:      g('--rlhf-card-text'),
            // Reward bars
            rewardLow:     g('--rlhf-reward-low'),
            rewardMid:     g('--rlhf-reward-mid'),
            rewardHigh:    g('--rlhf-reward-high'),
            rewardBarBg:   g('--rlhf-reward-bar-bg'),
            // Advantage
            advantagePos:  g('--rlhf-advantage-pos'),
            advantageNeg:  g('--rlhf-advantage-neg'),
            // Training curves
            rewardLine:    g('--rlhf-reward-line'),
            rewardFill:    g('--rlhf-reward-fill'),
            klLine:        g('--rlhf-kl-line'),
            klFill:        g('--rlhf-kl-fill'),
            thresholdLine: g('--rlhf-threshold-line'),
            gridColor:     g('--rlhf-grid-color'),
            // Policy
            refPolicy:     g('--rlhf-ref-policy'),
            curPolicy:     g('--rlhf-cur-policy'),
            policyBarBg:   g('--rlhf-policy-bar-bg'),
            // KL indicator
            klLow:         g('--rlhf-kl-low'),
            klMid:         g('--rlhf-kl-mid'),
            klHigh:        g('--rlhf-kl-high'),
            // General
            sectionTitle:  g('--rlhf-section-title'),
            labelColor:    g('--rlhf-label-color'),
            canvasText:    g('--rlhf-canvas-text'),
            arrowColor:    g('--rlhf-arrow-color'),
            promptBg:      g('--rlhf-prompt-bg'),
            promptBorder:  g('--rlhf-prompt-border'),
            canvasBg:      g('--viz-canvas-bg'),
            textMuted:     g('--viz-text-muted'),
        };
    }

    // ============================================
    // Main Visualizer Class
    // ============================================
    let clamp;

    class RLHFVisualizer {
        constructor() {
            this.canvas = document.getElementById('rlhf-canvas');
            this.ctx = this.canvas.getContext('2d');

            // HiDPI setup
            if (window.VizLib && window.VizLib.CanvasUtils) {
                window.VizLib.CanvasUtils.setupHiDPICanvas(this.canvas);
            }

            // Config
            this.scenarioKey = 'helpfulness';
            this.beta = 0.10;
            this.numPPOSteps = 8;
            this.speed = 5;

            // Simulation data
            this.simData = null;

            // Animation state
            this.phase = 'IDLE';
            this.phaseProgress = 0;
            this.phaseStartTime = 0;
            this.isProcessing = false;
            this.animId = null;

            // Loop counters
            this.currentPairIdx = 0;
            this.currentPPOStep = 0;

            // DOM elements
            this.scenarioSelect = document.getElementById('scenario-select');
            this.betaSlider = document.getElementById('beta-slider');
            this.betaValue = document.getElementById('beta-value');
            this.ppoStepsValue = document.getElementById('ppo-steps-value');
            this.btnPPODown = document.getElementById('btn-ppo-down');
            this.btnPPOUp = document.getElementById('btn-ppo-up');
            this.btnRun = document.getElementById('btn-run');
            this.btnStep = document.getElementById('btn-step');
            this.btnReset = document.getElementById('btn-reset');
            this.speedSlider = document.getElementById('speed-slider');
            this.speedValue = document.getElementById('speed-value');

            this.bindEvents();
            this.reset();
        }

        bindEvents() {
            this.btnRun.addEventListener('click', () => this.runAll());
            this.btnStep.addEventListener('click', () => this.stepOnce());
            this.btnReset.addEventListener('click', () => this.reset());

            this.speedSlider.addEventListener('input', () => {
                this.speed = parseInt(this.speedSlider.value);
                this.speedValue.textContent = this.speed;
            });

            this.betaSlider.addEventListener('input', () => {
                this.beta = parseFloat(this.betaSlider.value);
                this.betaValue.textContent = this.beta.toFixed(2);
                this.reset();
            });

            this.scenarioSelect.addEventListener('change', () => {
                this.scenarioKey = this.scenarioSelect.value;
                this.reset();
            });

            this.btnPPODown.addEventListener('click', () => {
                if (this.numPPOSteps > 3) {
                    this.numPPOSteps--;
                    this.ppoStepsValue.textContent = this.numPPOSteps;
                    this.reset();
                }
            });

            this.btnPPOUp.addEventListener('click', () => {
                if (this.numPPOSteps < 15) {
                    this.numPPOSteps++;
                    this.ppoStepsValue.textContent = this.numPPOSteps;
                    this.reset();
                }
            });

            document.addEventListener('themechange', () => this.draw());
        }

        // ============================================
        // Reset
        // ============================================
        reset() {
            this.stopAnimation();
            this.simData = precomputeSimulation(this.scenarioKey, this.beta, this.numPPOSteps);

            this.phase = 'IDLE';
            this.currentPairIdx = 0;
            this.currentPPOStep = 0;
            this.phaseProgress = 1;
            this.isProcessing = false;

            this.updateMetrics();
            this.draw();
        }

        // ============================================
        // Phase management
        // ============================================
        getPhaseDuration() {
            const base = Math.max(200, 1400 - this.speed * 120);
            const multipliers = {
                'SHOW_PIPELINE': 1.0,
                'SHOW_SFT': 1.0,
                'SFT_COMPLETE': 0.4,
                'SHOW_RM_INTRO': 0.8,
                'SHOW_PREFERENCE': 1.0,
                'LABEL_PREFERENCE': 0.8,
                'SCORE_PREFERENCE': 1.0,
                'TRAIN_RM_COMPLETE': 0.6,
                'SHOW_PPO_INTRO': 0.8,
                'GENERATE_RESPONSES': 1.0,
                'SCORE_RESPONSES': 1.0,
                'COMPUTE_ADVANTAGE': 0.8,
                'UPDATE_POLICY': 1.2,
                'COMPLETE': 1.0,
            };
            return base * (multipliers[this.phase] || 1.0);
        }

        getNextPhase() {
            switch (this.phase) {
                case 'IDLE':               return 'SHOW_PIPELINE';
                case 'SHOW_PIPELINE':      return 'SHOW_SFT';
                case 'SHOW_SFT':           return 'SFT_COMPLETE';
                case 'SFT_COMPLETE':       return 'SHOW_RM_INTRO';
                case 'SHOW_RM_INTRO':
                    this.currentPairIdx = 0;
                    return 'SHOW_PREFERENCE';
                case 'SHOW_PREFERENCE':    return 'LABEL_PREFERENCE';
                case 'LABEL_PREFERENCE':   return 'SCORE_PREFERENCE';
                case 'SCORE_PREFERENCE':
                    this.currentPairIdx++;
                    if (this.currentPairIdx < this.simData.preferencePairs.length) {
                        return 'SHOW_PREFERENCE';
                    }
                    return 'TRAIN_RM_COMPLETE';
                case 'TRAIN_RM_COMPLETE':  return 'SHOW_PPO_INTRO';
                case 'SHOW_PPO_INTRO':
                    this.currentPPOStep = 0;
                    return 'GENERATE_RESPONSES';
                case 'GENERATE_RESPONSES': return 'SCORE_RESPONSES';
                case 'SCORE_RESPONSES':    return 'COMPUTE_ADVANTAGE';
                case 'COMPUTE_ADVANTAGE':  return 'UPDATE_POLICY';
                case 'UPDATE_POLICY':
                    this.currentPPOStep++;
                    if (this.currentPPOStep < this.simData.ppoSteps.length) {
                        return 'GENERATE_RESPONSES';
                    }
                    return 'COMPLETE';
                default:                   return 'COMPLETE';
            }
        }

        advancePhase() {
            const next = this.getNextPhase();
            if (next === this.phase && next === 'COMPLETE') return false;

            this.phase = next;
            this.phaseProgress = 0;
            this.phaseStartTime = performance.now();

            this.updateMetrics();
            return true;
        }

        // ============================================
        // Step / Run
        // ============================================
        stepOnce() {
            if (this.phase === 'COMPLETE') return;
            this.stopAnimation();
            const advanced = this.advancePhase();
            if (advanced) {
                this.phaseProgress = 1;
                this.draw();
            }
        }

        runAll() {
            if (this.isProcessing) return;
            if (this.phase === 'COMPLETE') {
                this.phase = 'IDLE';
                this.currentPairIdx = 0;
                this.currentPPOStep = 0;
                this.phaseProgress = 1;
            }

            this.isProcessing = true;
            this.btnRun.disabled = true;

            const animate = (now) => {
                if (!this.isProcessing) return;

                const elapsed = now - this.phaseStartTime;
                const duration = this.getPhaseDuration();
                this.phaseProgress = clamp(elapsed / duration, 0, 1);

                this.draw();

                if (this.phaseProgress >= 1) {
                    if (this.phase === 'COMPLETE') {
                        this.stopAnimation();
                        return;
                    }
                    const advanced = this.advancePhase();
                    if (!advanced) {
                        this.stopAnimation();
                        return;
                    }
                }

                this.animId = requestAnimationFrame(animate);
            };

            if (this.phase === 'IDLE') {
                this.advancePhase();
            }
            this.phaseStartTime = performance.now();
            this.animId = requestAnimationFrame(animate);
        }

        stopAnimation() {
            if (this.animId) {
                cancelAnimationFrame(this.animId);
                this.animId = null;
            }
            this.isProcessing = false;
            this.btnRun.disabled = false;
        }

        // ============================================
        // Metrics
        // ============================================
        updateMetrics() {
            const phaseNames = {
                'IDLE': 'Idle',
                'SHOW_PIPELINE': 'Pipeline',
                'SHOW_SFT': 'SFT Training',
                'SFT_COMPLETE': 'SFT Complete',
                'SHOW_RM_INTRO': 'Reward Model',
                'SHOW_PREFERENCE': 'Showing Pair',
                'LABEL_PREFERENCE': 'Labeling Pair',
                'SCORE_PREFERENCE': 'Scoring Pair',
                'TRAIN_RM_COMPLETE': 'RM Complete',
                'SHOW_PPO_INTRO': 'PPO Setup',
                'GENERATE_RESPONSES': 'Generating',
                'SCORE_RESPONSES': 'Scoring',
                'COMPUTE_ADVANTAGE': 'Advantages',
                'UPDATE_POLICY': 'Updating Policy',
                'COMPLETE': 'Complete',
            };

            document.getElementById('metric-phase').textContent = phaseNames[this.phase] || this.phase;

            // PPO step
            const isPPO = ['GENERATE_RESPONSES', 'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY'].includes(this.phase);
            if (isPPO) {
                document.getElementById('metric-ppo-step').textContent =
                    (this.currentPPOStep + 1) + ' / ' + this.numPPOSteps;
            } else if (this.phase === 'COMPLETE') {
                document.getElementById('metric-ppo-step').textContent =
                    this.numPPOSteps + ' / ' + this.numPPOSteps;
            } else {
                document.getElementById('metric-ppo-step').textContent = '- / ' + this.numPPOSteps;
            }

            // Avg Reward & KL
            if (isPPO || this.phase === 'COMPLETE') {
                const stepIdx = this.phase === 'COMPLETE'
                    ? this.simData.ppoSteps.length - 1
                    : Math.min(this.currentPPOStep, this.simData.ppoSteps.length - 1);
                const step = this.simData.ppoSteps[stepIdx];
                if (step) {
                    document.getElementById('metric-avg-reward').textContent = step.avgReward.toFixed(3);
                    document.getElementById('metric-kl-div').textContent = step.klDivergence.toFixed(4);
                    document.getElementById('metric-entropy').textContent = step.entropy.toFixed(3);
                }
            } else {
                document.getElementById('metric-avg-reward').textContent = '-';
                document.getElementById('metric-kl-div').textContent = '-';
                document.getElementById('metric-entropy').textContent = '-';
            }

            // RM Accuracy
            if (['TRAIN_RM_COMPLETE', 'SHOW_PPO_INTRO', 'GENERATE_RESPONSES', 'SCORE_RESPONSES',
                 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'].includes(this.phase)) {
                document.getElementById('metric-rm-accuracy').textContent =
                    (this.simData.rmAccuracy * 100).toFixed(0) + '%';
            } else {
                document.getElementById('metric-rm-accuracy').textContent = '-';
            }
        }

        // ============================================
        // Drawing — main dispatcher
        // ============================================
        draw() {
            const ctx = this.ctx;
            const C = getThemeColors();
            const dpr = window.devicePixelRatio || 1;

            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
            ctx.fillStyle = C.canvasBg;
            ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

            this.drawZoneA(ctx, C);
            this.drawZoneB(ctx, C);
            this.drawZoneC(ctx, C);
            this.drawZoneD(ctx, C);
        }

        // ============================================
        // Zone A: Pipeline Overview (y: 0–60)
        // ============================================
        drawZoneA(ctx, C) {
            const { y: zy } = ZONE.A;

            if (this.phase === 'IDLE') return;

            // Section title
            ctx.font = 'bold 10px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('RLHF PIPELINE', 15, zy + 4);

            const stages = [
                { label: 'SFT', color: C.sftColor, bg: C.sftBg, glow: C.sftGlow },
                { label: 'Reward Model', color: C.rmColor, bg: C.rmBg, glow: C.rmGlow },
                { label: 'PPO Training', color: C.ppoColor, bg: C.ppoBg, glow: C.ppoGlow },
            ];

            const boxW = 130, boxH = 36;
            const gap = 50;
            const totalW = stages.length * boxW + (stages.length - 1) * gap;
            const startX = (CANVAS_W - totalW) / 2;
            const centerY = zy + 32;

            // Determine which stage is active/complete
            const sftPhases = ['SHOW_SFT'];
            const sftComplete = ['SFT_COMPLETE', 'SHOW_RM_INTRO', 'SHOW_PREFERENCE', 'LABEL_PREFERENCE',
                'SCORE_PREFERENCE', 'TRAIN_RM_COMPLETE', 'SHOW_PPO_INTRO', 'GENERATE_RESPONSES',
                'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'];
            const rmPhases = ['SHOW_RM_INTRO', 'SHOW_PREFERENCE', 'LABEL_PREFERENCE', 'SCORE_PREFERENCE'];
            const rmComplete = ['TRAIN_RM_COMPLETE', 'SHOW_PPO_INTRO', 'GENERATE_RESPONSES',
                'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'];
            const ppoPhases = ['SHOW_PPO_INTRO', 'GENERATE_RESPONSES', 'SCORE_RESPONSES',
                'COMPUTE_ADVANTAGE', 'UPDATE_POLICY'];
            const ppoComplete = ['COMPLETE'];

            const stageStates = [
                sftPhases.includes(this.phase) ? 'active' : sftComplete.includes(this.phase) ? 'complete' : 'inactive',
                rmPhases.includes(this.phase) ? 'active' : rmComplete.includes(this.phase) ? 'complete' : 'inactive',
                ppoPhases.includes(this.phase) ? 'active' : ppoComplete.includes(this.phase) ? 'complete' : 'inactive',
            ];

            // Fade in during SHOW_PIPELINE
            const alpha = this.phase === 'SHOW_PIPELINE' ? this.phaseProgress : 1;
            ctx.save();
            ctx.globalAlpha = alpha;

            for (let i = 0; i < stages.length; i++) {
                const x = startX + i * (boxW + gap);
                const y = centerY - boxH / 2;
                const st = stages[i];
                const state = stageStates[i];

                this.drawStageBox(ctx, C, x, y, boxW, boxH, st, state);

                // Connecting arrows
                if (i < stages.length - 1) {
                    const arrowX1 = x + boxW + 6;
                    const arrowX2 = x + boxW + gap - 6;
                    const arrowY = centerY;

                    ctx.strokeStyle = C.arrowColor;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(arrowX1, arrowY);
                    ctx.lineTo(arrowX2 - 8, arrowY);
                    ctx.stroke();

                    // Arrow head
                    ctx.fillStyle = C.arrowColor;
                    ctx.beginPath();
                    ctx.moveTo(arrowX2, arrowY);
                    ctx.lineTo(arrowX2 - 8, arrowY - 4);
                    ctx.lineTo(arrowX2 - 8, arrowY + 4);
                    ctx.closePath();
                    ctx.fill();
                }
            }

            ctx.restore();
        }

        drawStageBox(ctx, C, x, y, w, h, stage, state) {
            let bg, border, textColor;

            if (state === 'active') {
                bg = stage.bg;
                border = stage.color;
                textColor = stage.color;
                // Glow
                ctx.save();
                ctx.shadowColor = stage.glow;
                ctx.shadowBlur = 10;
                ctx.fillStyle = bg;
                this.roundRect(ctx, x, y, w, h, 6);
                ctx.fill();
                ctx.restore();
            } else if (state === 'complete') {
                bg = stage.bg;
                border = C.completeColor;
                textColor = C.completeColor;
            } else {
                bg = C.inactiveBg;
                border = C.inactiveColor;
                textColor = C.inactiveColor;
            }

            // Box
            ctx.fillStyle = bg;
            ctx.strokeStyle = border;
            ctx.lineWidth = state === 'active' ? 2 : 1.5;
            this.roundRect(ctx, x, y, w, h, 6);
            ctx.fill();
            ctx.stroke();

            // Label
            ctx.font = 'bold 12px ' + MONO;
            ctx.fillStyle = textColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const labelX = state === 'complete' ? x + w / 2 - 8 : x + w / 2;
            ctx.fillText(stage.label, labelX, y + h / 2);

            // Checkmark for complete
            if (state === 'complete') {
                ctx.font = 'bold 14px sans-serif';
                ctx.fillStyle = C.completeColor;
                ctx.textAlign = 'left';
                ctx.fillText('\u2713', x + w / 2 + ctx.measureText(stage.label).width / 2 + 2, y + h / 2);
            }
        }

        // ============================================
        // Zone B: Main Stage View (y: 65–310)
        // ============================================
        drawZoneB(ctx, C) {
            const sftPhases = ['SHOW_SFT', 'SFT_COMPLETE'];
            const rmPhases = ['SHOW_RM_INTRO', 'SHOW_PREFERENCE', 'LABEL_PREFERENCE', 'SCORE_PREFERENCE', 'TRAIN_RM_COMPLETE'];
            const ppoPhases = ['SHOW_PPO_INTRO', 'GENERATE_RESPONSES', 'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'];

            if (sftPhases.includes(this.phase)) {
                this.drawSFTView(ctx, C);
            } else if (rmPhases.includes(this.phase)) {
                this.drawRMView(ctx, C);
            } else if (ppoPhases.includes(this.phase)) {
                this.drawPPOView(ctx, C);
            }
        }

        drawSFTView(ctx, C) {
            const { y: zy, h: zh } = ZONE.B;
            const centerX = CANVAS_W / 2;
            const centerY = zy + zh / 2;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('SUPERVISED FINE-TUNING', 15, zy + 5);

            // Message
            ctx.font = '14px ' + MONO;
            ctx.fillStyle = C.canvasText;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Fine-tuning base model on curated demonstrations...', centerX, centerY - 30);

            // Progress bar
            const barW = 300, barH = 20;
            const barX = centerX - barW / 2;
            const barY = centerY + 10;

            ctx.fillStyle = C.rewardBarBg;
            this.roundRect(ctx, barX, barY, barW, barH, 6);
            ctx.fill();

            const progress = this.phase === 'SFT_COMPLETE' ? 1 : this.phaseProgress;
            const fillW = Math.max(4, barW * progress);
            ctx.fillStyle = C.sftColor;
            this.roundRect(ctx, barX, barY, fillW, barH, 6);
            ctx.fill();

            // Percentage
            ctx.font = 'bold 12px ' + MONO;
            ctx.fillStyle = C.canvasText;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(Math.round(progress * 100) + '%', centerX, barY + barH + 8);
        }

        drawRMView(ctx, C) {
            const { y: zy, h: zh } = ZONE.B;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('REWARD MODEL TRAINING', 15, zy + 5);

            if (this.phase === 'SHOW_RM_INTRO') {
                ctx.font = '14px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Training reward model from human preferences...', CANVAS_W / 2, zy + zh / 2);
                return;
            }

            if (this.phase === 'TRAIN_RM_COMPLETE') {
                ctx.font = '14px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Reward model trained!', CANVAS_W / 2, zy + zh / 2 - 20);
                ctx.font = '12px ' + MONO;
                ctx.fillStyle = C.sectionTitle;
                ctx.fillText('Accuracy: ' + (this.simData.rmAccuracy * 100).toFixed(0) + '%', CANVAS_W / 2, zy + zh / 2 + 10);
                return;
            }

            // Show preference pair
            const pairIdx = Math.min(this.currentPairIdx, this.simData.preferencePairs.length - 1);
            const pair = this.simData.preferencePairs[pairIdx];
            if (!pair) return;

            // Pair counter
            ctx.font = '10px ' + MONO;
            ctx.fillStyle = C.labelColor;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillText('Pair ' + (pairIdx + 1) + ' / ' + this.simData.preferencePairs.length, CANVAS_W - 15, zy + 5);

            // Prompt
            ctx.font = '11px ' + MONO;
            ctx.fillStyle = C.labelColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Prompt: "' + this.simData.scenario.prompt + '"', CANVAS_W / 2, zy + 22);

            // Two response cards side-by-side
            const cardW = 280, cardH = 140;
            const gap = 30;
            const cardsY = zy + 50;
            const leftX = CANVAS_W / 2 - cardW - gap / 2;
            const rightX = CANVAS_W / 2 + gap / 2;

            // Fade in cards during SHOW_PREFERENCE
            const cardAlpha = this.phase === 'SHOW_PREFERENCE' ? this.phaseProgress : 1;

            ctx.save();
            ctx.globalAlpha = cardAlpha;

            // Preferred card (left)
            const showLabels = this.phase === 'LABEL_PREFERENCE' || this.phase === 'SCORE_PREFERENCE';
            const showScores = this.phase === 'SCORE_PREFERENCE';

            this.drawResponseCard(ctx, C, leftX, cardsY, cardW, cardH, {
                label: pair.preferredLabel,
                type: showLabels ? 'preferred' : 'neutral',
                score: showScores ? pair.preferredScore : null,
                badge: showLabels ? '\u2713' : null,
                badgeProgress: this.phase === 'LABEL_PREFERENCE' ? this.phaseProgress : 1,
            });

            // Rejected card (right)
            this.drawResponseCard(ctx, C, rightX, cardsY, cardW, cardH, {
                label: pair.rejectedLabel,
                type: showLabels ? 'rejected' : 'neutral',
                score: showScores ? pair.rejectedScore : null,
                badge: showLabels ? '\u2717' : null,
                badgeProgress: this.phase === 'LABEL_PREFERENCE' ? this.phaseProgress : 1,
            });

            ctx.restore();

            // Labels under cards
            if (showLabels) {
                ctx.font = 'bold 10px ' + MONO;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillStyle = C.preferredBorder;
                ctx.fillText('PREFERRED', leftX + cardW / 2, cardsY + cardH + 8);
                ctx.fillStyle = C.rejectedBorder;
                ctx.fillText('REJECTED', rightX + cardW / 2, cardsY + cardH + 8);
            }
        }

        drawPPOView(ctx, C) {
            const { y: zy, h: zh } = ZONE.B;

            // Section title
            ctx.font = 'bold 11px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('PPO TRAINING', 15, zy + 5);

            if (this.phase === 'SHOW_PPO_INTRO') {
                ctx.font = '14px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('Optimizing policy with PPO + KL constraint...', CANVAS_W / 2, zy + zh / 2);
                return;
            }

            // Step counter
            const isPPO = ['GENERATE_RESPONSES', 'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY'].includes(this.phase);
            const stepIdx = isPPO ? this.currentPPOStep : this.simData.ppoSteps.length - 1;
            const step = this.simData.ppoSteps[Math.min(stepIdx, this.simData.ppoSteps.length - 1)];
            if (!step) return;

            // Step counter
            ctx.font = '10px ' + MONO;
            ctx.fillStyle = C.labelColor;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillText('Step ' + step.stepNum + ' / ' + this.numPPOSteps, CANVAS_W - 15, zy + 5);

            // Prompt box
            const promptY = zy + 22;
            ctx.font = '11px ' + MONO;
            ctx.fillStyle = C.labelColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Prompt: "' + this.simData.scenario.prompt + '"', CANVAS_W / 2, promptY);

            // 4 response cards in 2x2 grid
            const cardW = 310, cardH = 80;
            const gapX = 20, gapY = 12;
            const gridStartX = (CANVAS_W - cardW * 2 - gapX) / 2;
            const gridStartY = zy + 45;

            for (let r = 0; r < 4; r++) {
                const col = r % 2;
                const row = Math.floor(r / 2);
                const cx = gridStartX + col * (cardW + gapX);
                const cy = gridStartY + row * (cardH + gapY);

                const resp = step.responses[r];

                // Staggered fade-in during GENERATE_RESPONSES
                let alpha = 1;
                if (this.phase === 'GENERATE_RESPONSES') {
                    const threshold = r / 4;
                    alpha = clamp((this.phaseProgress - threshold) / 0.25, 0, 1);
                }

                ctx.save();
                ctx.globalAlpha = alpha;

                const showReward = this.phase === 'SCORE_RESPONSES' || this.phase === 'COMPUTE_ADVANTAGE' || this.phase === 'UPDATE_POLICY' || this.phase === 'COMPLETE';
                const showAdvantage = this.phase === 'COMPUTE_ADVANTAGE' || this.phase === 'UPDATE_POLICY' || this.phase === 'COMPLETE';

                this.drawResponseCard(ctx, C, cx, cy, cardW, cardH, {
                    label: resp.label,
                    type: 'neutral',
                    score: showReward ? resp.reward : null,
                    scoreProgress: this.phase === 'SCORE_RESPONSES' ? this.phaseProgress : 1,
                    advantage: showAdvantage ? resp.advantage : null,
                });

                ctx.restore();
            }
        }

        drawResponseCard(ctx, C, x, y, w, h, opts) {
            const { label, type, score, badge, badgeProgress, scoreProgress, advantage } = opts;

            let bg, border;
            if (type === 'preferred') {
                bg = C.preferredBg;
                border = C.preferredBorder;
            } else if (type === 'rejected') {
                bg = C.rejectedBg;
                border = C.rejectedBorder;
            } else {
                bg = C.cardBg;
                border = C.cardBorder;
            }

            // Card background
            ctx.fillStyle = bg;
            ctx.strokeStyle = border;
            ctx.lineWidth = type === 'neutral' ? 1 : 2;
            this.roundRect(ctx, x, y, w, h, 6);
            ctx.fill();
            ctx.stroke();

            // Response label
            ctx.font = '12px ' + MONO;
            ctx.fillStyle = C.cardText;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(label, x + 12, y + 12);

            // Badge (check/cross)
            if (badge && badgeProgress > 0) {
                ctx.save();
                ctx.globalAlpha = (badgeProgress || 1);
                ctx.font = 'bold 20px sans-serif';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'top';
                ctx.fillStyle = type === 'preferred' ? C.preferredBorder : C.rejectedBorder;
                ctx.fillText(badge, x + w - 10, y + 6);
                ctx.restore();
            }

            // Reward score bar
            if (score !== null && score !== undefined) {
                const barX = x + 12;
                const barY = y + h - 28;
                const barW = w - 24;
                const barH = 14;

                ctx.fillStyle = C.rewardBarBg;
                this.roundRect(ctx, barX, barY, barW, barH, 3);
                ctx.fill();

                const progress = scoreProgress !== undefined ? scoreProgress : 1;
                const fillW = Math.max(2, barW * score * progress);
                ctx.fillStyle = this.getRewardColor(C, score);
                this.roundRect(ctx, barX, barY, fillW, barH, 3);
                ctx.fill();

                // Score value
                ctx.font = '10px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'bottom';
                ctx.fillText('r=' + score.toFixed(2), x + w - 12, barY - 2);
            }

            // Advantage overlay
            if (advantage !== null && advantage !== undefined) {
                const advText = (advantage >= 0 ? '+' : '') + advantage.toFixed(2);
                ctx.font = 'bold 11px ' + MONO;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'top';
                ctx.fillStyle = advantage >= 0 ? C.advantagePos : C.advantageNeg;
                ctx.fillText('A=' + advText, x + w - 12, y + 28);
            }
        }

        getRewardColor(C, score) {
            if (score >= 0.65) return C.rewardHigh;
            if (score >= 0.35) return C.rewardMid;
            return C.rewardLow;
        }

        // ============================================
        // Zone C: Training Curves (y: 315–510)
        // ============================================
        drawZoneC(ctx, C) {
            const { y: zy, h: zh } = ZONE.C;

            // Only show during PPO phases
            const ppoPhases = ['SHOW_PPO_INTRO', 'GENERATE_RESPONSES', 'SCORE_RESPONSES',
                'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'];
            if (!ppoPhases.includes(this.phase)) return;

            // Collect data points up to current step
            const dataPoints = [];
            const maxStep = this.phase === 'COMPLETE'
                ? this.simData.ppoSteps.length
                : this.currentPPOStep + (this.phase === 'UPDATE_POLICY' ? 1 : 0);

            for (let i = 0; i < maxStep && i < this.simData.ppoSteps.length; i++) {
                dataPoints.push(this.simData.ppoSteps[i]);
            }

            const chartW = 310, chartH = 155;
            const leftChartX = 30;
            const rightChartX = CANVAS_W / 2 + 20;

            // Left chart: Average Reward
            this.drawMiniChart(ctx, C, leftChartX, zy + 25, chartW, chartH, {
                title: 'Average Reward',
                data: dataPoints.map(d => d.avgReward),
                lineColor: C.rewardLine,
                fillColor: C.rewardFill,
                yMax: 1.0,
                yLabel: 'Reward',
                maxPoints: this.numPPOSteps,
                animProgress: this.phase === 'UPDATE_POLICY' ? this.phaseProgress : 1,
            });

            // Right chart: KL Divergence
            this.drawMiniChart(ctx, C, rightChartX, zy + 25, chartW, chartH, {
                title: 'KL Divergence',
                data: dataPoints.map(d => d.klDivergence),
                lineColor: C.klLine,
                fillColor: C.klFill,
                yMax: Math.max(this.beta * 1.5, 0.15),
                yLabel: 'KL',
                maxPoints: this.numPPOSteps,
                threshold: this.beta,
                thresholdLabel: '\u03B2=' + this.beta.toFixed(2),
                thresholdColor: C.thresholdLine,
                animProgress: this.phase === 'UPDATE_POLICY' ? this.phaseProgress : 1,
            });
        }

        drawMiniChart(ctx, C, x, y, w, h, opts) {
            const { title, data, lineColor, fillColor, yMax, yLabel, maxPoints, threshold, thresholdLabel, thresholdColor, animProgress } = opts;

            const pad = { top: 22, right: 12, bottom: 22, left: 38 };
            const plotW = w - pad.left - pad.right;
            const plotH = h - pad.top - pad.bottom;
            const ox = x + pad.left;
            const oy = y + pad.top;

            // Title
            ctx.font = 'bold 10px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(title, x + pad.left, y + 4);

            // Grid lines
            ctx.strokeStyle = C.gridColor;
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const gy = oy + (plotH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(ox, gy);
                ctx.lineTo(ox + plotW, gy);
                ctx.stroke();
            }

            // Y-axis labels
            ctx.fillStyle = C.labelColor;
            ctx.font = '9px ' + MONO;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i <= 4; i++) {
                const val = yMax * (1 - i / 4);
                const gy = oy + (plotH / 4) * i;
                ctx.fillText(val.toFixed(2), ox - 4, gy);
            }

            // X-axis label
            ctx.font = '9px ' + MONO;
            ctx.fillStyle = C.labelColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Step', x + w / 2, y + h - 10);

            // Threshold line (for KL chart)
            if (threshold !== undefined && thresholdColor) {
                const threshY = oy + plotH - (threshold / yMax) * plotH;
                if (threshY >= oy && threshY <= oy + plotH) {
                    ctx.strokeStyle = thresholdColor;
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([6, 4]);
                    ctx.beginPath();
                    ctx.moveTo(ox, threshY);
                    ctx.lineTo(ox + plotW, threshY);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    if (thresholdLabel) {
                        ctx.font = '9px ' + MONO;
                        ctx.fillStyle = thresholdColor;
                        ctx.textAlign = 'right';
                        ctx.textBaseline = 'bottom';
                        ctx.fillText(thresholdLabel, ox + plotW, threshY - 2);
                    }
                }
            }

            if (data.length === 0) return;

            const xStep = maxPoints > 1 ? plotW / (maxPoints - 1) : plotW;

            // Build points
            const points = [];
            for (let i = 0; i < data.length; i++) {
                const px = ox + i * xStep;
                const py = oy + plotH - clamp(data[i] / yMax, 0, 1) * plotH;
                points.push({ x: px, y: py });
            }

            // Animate last point
            if (points.length >= 2 && animProgress < 1) {
                const last = points[points.length - 1];
                const prev = points[points.length - 2];
                last.x = prev.x + (last.x - prev.x) * animProgress;
                last.y = prev.y + (last.y - prev.y) * animProgress;
            }

            // Filled area
            ctx.beginPath();
            ctx.moveTo(points[0].x, oy + plotH);
            for (const p of points) ctx.lineTo(p.x, p.y);
            ctx.lineTo(points[points.length - 1].x, oy + plotH);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();

            // Line
            ctx.beginPath();
            for (let i = 0; i < points.length; i++) {
                if (i === 0) ctx.moveTo(points[i].x, points[i].y);
                else ctx.lineTo(points[i].x, points[i].y);
            }
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Data points
            for (const p of points) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = lineColor;
                ctx.fill();
            }
        }

        // ============================================
        // Zone D: Policy Distributions (y: 515–700)
        // ============================================
        drawZoneD(ctx, C) {
            const { y: zy, h: zh } = ZONE.D;

            // Only show during PPO phases
            const ppoPhases = ['SHOW_PPO_INTRO', 'GENERATE_RESPONSES', 'SCORE_RESPONSES',
                'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'];
            if (!ppoPhases.includes(this.phase)) return;

            const halfW = CANVAS_W / 2;

            // Reference policy (left)
            ctx.font = 'bold 10px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('REFERENCE POLICY \u03C0_ref', 15, zy + 4);

            this.drawPolicyBars(ctx, C, 15, zy + 22, halfW - 40, zh - 40,
                this.simData.refPolicyDist, C.refPolicy, this.simData.scenario.tierShort);

            // Current policy (right)
            ctx.font = 'bold 10px ' + MONO;
            ctx.fillStyle = C.sectionTitle;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('CURRENT POLICY \u03C0_\u03B8', halfW + 25, zy + 4);

            // Get current policy distribution
            let currentDist = this.simData.refPolicyDist;
            const isPPO = ['GENERATE_RESPONSES', 'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY'].includes(this.phase);

            if (isPPO || this.phase === 'COMPLETE') {
                const stepIdx = this.phase === 'COMPLETE'
                    ? this.simData.ppoSteps.length - 1
                    : Math.min(this.currentPPOStep, this.simData.ppoSteps.length - 1);
                const step = this.simData.ppoSteps[stepIdx];

                if (this.phase === 'UPDATE_POLICY' && step) {
                    // Interpolate from previous to current distribution
                    const prevDist = step.prevDist;
                    const nextDist = step.policyDist;
                    currentDist = prevDist.map((p, i) => p + (nextDist[i] - p) * this.phaseProgress);
                } else if (step) {
                    currentDist = step.policyDist;
                }
            }

            this.drawPolicyBars(ctx, C, halfW + 25, zy + 22, halfW - 40, zh - 40,
                currentDist, C.curPolicy, this.simData.scenario.tierShort);

            // KL indicator in center
            this.drawKLIndicator(ctx, C, halfW, zy + zh / 2);
        }

        drawPolicyBars(ctx, C, x, y, w, h, dist, barColor, labels) {
            const barH = 22;
            const gap = 10;
            const labelW = 70;
            const barAreaW = w - labelW - 10;
            const maxProb = Math.max(...dist, 0.01);

            for (let i = 0; i < dist.length; i++) {
                const by = y + i * (barH + gap);
                if (by + barH > y + h) break;

                // Label
                ctx.font = '10px ' + MONO;
                ctx.fillStyle = C.labelColor;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText(labels[i], x + labelW, by + barH / 2);

                // Bar background
                ctx.fillStyle = C.policyBarBg;
                this.roundRect(ctx, x + labelW + 6, by, barAreaW, barH, 3);
                ctx.fill();

                // Bar fill (scaled to max for visual clarity, but proportional)
                const barW = Math.max(2, barAreaW * (dist[i] / maxProb) * 0.9);
                ctx.fillStyle = barColor;
                this.roundRect(ctx, x + labelW + 6, by, barW, barH, 3);
                ctx.fill();

                // Probability value
                ctx.font = '9px ' + MONO;
                ctx.fillStyle = C.canvasText;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText((dist[i] * 100).toFixed(1) + '%', x + labelW + 6 + barW + 4, by + barH / 2);
            }
        }

        drawKLIndicator(ctx, C, centerX, centerY) {
            const isPPO = ['GENERATE_RESPONSES', 'SCORE_RESPONSES', 'COMPUTE_ADVANTAGE', 'UPDATE_POLICY', 'COMPLETE'].includes(this.phase);
            if (!isPPO) return;

            const stepIdx = this.phase === 'COMPLETE'
                ? this.simData.ppoSteps.length - 1
                : Math.min(this.currentPPOStep, this.simData.ppoSteps.length - 1);
            const step = this.simData.ppoSteps[stepIdx];
            if (!step) return;

            const kl = step.klDivergence;
            const ratio = clamp(kl / this.beta, 0, 1.5);

            // Color based on KL magnitude
            let color;
            if (ratio < 0.5) color = C.klLow;
            else if (ratio < 0.9) color = C.klMid;
            else color = C.klHigh;

            // Arrow pointing right
            const arrowLen = 30;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(centerX - arrowLen / 2, centerY);
            ctx.lineTo(centerX + arrowLen / 2 - 6, centerY);
            ctx.stroke();

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(centerX + arrowLen / 2, centerY);
            ctx.lineTo(centerX + arrowLen / 2 - 8, centerY - 4);
            ctx.lineTo(centerX + arrowLen / 2 - 8, centerY + 4);
            ctx.closePath();
            ctx.fill();

            // KL value label
            ctx.font = 'bold 10px ' + MONO;
            ctx.fillStyle = color;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('KL=' + kl.toFixed(3), centerX, centerY - 10);
        }

        // ============================================
        // Drawing helpers
        // ============================================
        roundRect(ctx, x, y, w, h, r) {
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
        }

        drawRewardBar(ctx, C, x, y, w, h, score, progress) {
            ctx.fillStyle = C.rewardBarBg;
            this.roundRect(ctx, x, y, w, h, 3);
            ctx.fill();

            const fillW = Math.max(2, w * score * (progress || 1));
            ctx.fillStyle = this.getRewardColor(C, score);
            this.roundRect(ctx, x, y, fillW, h, 3);
            ctx.fill();
        }
    }

    // ============================================
    // Bootstrap
    // ============================================
    function init() {
        clamp = VizLib.MathUtils.clamp;
        new RLHFVisualizer();

        // Wire up info-panel tabs
        const tabButtons = document.querySelectorAll('.info-panel-tabs [data-tab]');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                tabButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const panel = btn.closest('.panel');
                panel.querySelectorAll('.info-tab-content').forEach(c => c.classList.remove('active'));
                const target = panel.querySelector('#tab-' + btn.dataset.tab);
                if (target) target.classList.add('active');
            });
        });
    }

    window.addEventListener('vizlib-ready', init);
})();
