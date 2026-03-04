(function() {
    'use strict';

    // ========== State ==========
    var state = {
        regType: 'l1',
        lambda: 1.0,
        eccentricity: 4.0,
        cx: 1.5,         // loss center w1
        cy: 1.0,         // loss center w2
        alpha: 0.5,       // elastic net mixing
        rotation: 30,     // loss ellipse rotation in degrees
        viewMode: '3d',   // '3d' or 'contour'
        // Surface visibility
        showLoss: true,
        showPenalty: true,
        showTotal: false,
        // 3D camera
        yaw: -0.6,
        pitch: 0.7,
        dragging: false,
        lastMouse: null,
        // Gradient descent
        gdPath: [],
        gdAnimating: false,
        gdLR: 0.05,
        gdAnimFrame: null,
        // Computed
        optimum: null     // {w1, w2}
    };

    var canvas3d, ctx3d;
    var canvasContour, ctxContour;

    // ========== Color Helpers ==========
    function getCS() {
        var s = getComputedStyle(document.documentElement);
        return {
            l1: s.getPropertyValue('--reg-l1-color').trim(),
            l2: s.getPropertyValue('--reg-l2-color').trim(),
            elastic: s.getPropertyValue('--reg-elastic-color').trim(),
            none: s.getPropertyValue('--reg-none-color').trim(),
            loss: s.getPropertyValue('--reg-loss-color').trim(),
            optimal: s.getPropertyValue('--reg-optimal-color').trim(),
            path: s.getPropertyValue('--reg-path-color').trim(),
            contour: s.getPropertyValue('--reg-contour-color').trim(),
            constraintFill: s.getPropertyValue('--reg-constraint-fill').trim(),
            warm: s.getPropertyValue('--reg-surface-warm').trim(),
            cool: s.getPropertyValue('--reg-surface-cool').trim(),
            mid: s.getPropertyValue('--reg-surface-mid').trim(),
            gridLine: s.getPropertyValue('--reg-grid-line').trim(),
            axis: s.getPropertyValue('--reg-axis-color').trim(),
            zero: s.getPropertyValue('--reg-zero-marker').trim(),
            text: s.getPropertyValue('--viz-text').trim(),
            textMuted: s.getPropertyValue('--viz-text-muted').trim(),
            bg: s.getPropertyValue('--viz-canvas-bg').trim(),
            border: s.getPropertyValue('--viz-border').trim(),
            gd: s.getPropertyValue('--reg-gd-color').trim()
        };
    }

    function hexToRgb(hex) {
        hex = hex.replace('#', '');
        if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
        return [parseInt(hex.substring(0,2),16), parseInt(hex.substring(2,4),16), parseInt(hex.substring(4,6),16)];
    }

    function lerpColor(c1, c2, t) {
        var a = hexToRgb(c1), b = hexToRgb(c2);
        return 'rgb(' + Math.round(a[0]+(b[0]-a[0])*t) + ',' + Math.round(a[1]+(b[1]-a[1])*t) + ',' + Math.round(a[2]+(b[2]-a[2])*t) + ')';
    }

    // ========== Loss & Penalty Functions ==========
    function lossAt(w1, w2) {
        // Rotated elliptical quadratic loss centered at (cx, cy)
        var dx = w1 - state.cx;
        var dy = w2 - state.cy;
        var rad = state.rotation * Math.PI / 180;
        var cos = Math.cos(rad), sin = Math.sin(rad);
        var rx = cos * dx + sin * dy;
        var ry = -sin * dx + cos * dy;
        return state.eccentricity * rx * rx + ry * ry;
    }

    function penaltyAt(w1, w2) {
        switch (state.regType) {
            case 'l1': return Math.abs(w1) + Math.abs(w2);
            case 'l2': return w1 * w1 + w2 * w2;
            case 'elastic':
                return state.alpha * (Math.abs(w1) + Math.abs(w2)) + (1 - state.alpha) * (w1 * w1 + w2 * w2);
            default: return 0;
        }
    }

    function totalAt(w1, w2) {
        return lossAt(w1, w2) + state.lambda * penaltyAt(w1, w2);
    }

    // ========== Gradients for GD ==========
    function gradLoss(w1, w2) {
        var dx = w1 - state.cx;
        var dy = w2 - state.cy;
        var rad = state.rotation * Math.PI / 180;
        var cos = Math.cos(rad), sin = Math.sin(rad);
        var rx = cos * dx + sin * dy;
        var ry = -sin * dx + cos * dy;
        var dLdrx = 2 * state.eccentricity * rx;
        var dLdry = 2 * ry;
        return {
            dw1: dLdrx * cos + dLdry * (-sin),
            dw2: dLdrx * sin + dLdry * cos
        };
    }

    function gradPenalty(w1, w2) {
        switch (state.regType) {
            case 'l1': return { dw1: Math.sign(w1) || 0, dw2: Math.sign(w2) || 0 };
            case 'l2': return { dw1: 2 * w1, dw2: 2 * w2 };
            case 'elastic': return {
                dw1: state.alpha * (Math.sign(w1) || 0) + (1 - state.alpha) * 2 * w1,
                dw2: state.alpha * (Math.sign(w2) || 0) + (1 - state.alpha) * 2 * w2
            };
            default: return { dw1: 0, dw2: 0 };
        }
    }

    function gradTotal(w1, w2) {
        var gl = gradLoss(w1, w2);
        var gp = gradPenalty(w1, w2);
        return {
            dw1: gl.dw1 + state.lambda * gp.dw1,
            dw2: gl.dw2 + state.lambda * gp.dw2
        };
    }

    // ========== Gradient Descent Simulation ==========
    function startGD(w1, w2) {
        // Clamp to range
        w1 = Math.max(-3, Math.min(3, w1));
        w2 = Math.max(-3, Math.min(3, w2));
        state.gdPath = [{w1: w1, w2: w2}];
        state.gdAnimating = true;
        if (state.gdAnimFrame) cancelAnimationFrame(state.gdAnimFrame);
        animateGD();
    }

    function animateGD() {
        if (!state.gdAnimating || state.gdPath.length > 500) {
            state.gdAnimating = false;
            return;
        }
        var last = state.gdPath[state.gdPath.length - 1];
        // Take a few steps per frame for smoother animation
        var stepsPerFrame = 3;
        for (var s = 0; s < stepsPerFrame; s++) {
            var g = gradTotal(last.w1, last.w2);
            var gnorm = Math.sqrt(g.dw1 * g.dw1 + g.dw2 * g.dw2);
            if (gnorm < 1e-5) {
                state.gdAnimating = false;
                redraw();
                return;
            }
            var nw1 = last.w1 - state.gdLR * g.dw1;
            var nw2 = last.w2 - state.gdLR * g.dw2;
            nw1 = Math.max(-3, Math.min(3, nw1));
            nw2 = Math.max(-3, Math.min(3, nw2));
            last = {w1: nw1, w2: nw2};
            state.gdPath.push(last);
        }
        redraw();
        state.gdAnimFrame = requestAnimationFrame(animateGD);
    }

    // Smooth log compression: preserves shape near 0, no hard plateau
    function compress(val) {
        var c = 5;
        return c * Math.log(1 + val / c);
    }

    // ========== Find Optimum via Grid Search ==========
    function findOptimum() {
        var bestW1 = 0, bestW2 = 0, bestVal = Infinity;
        var range = 3, steps = 200;
        for (var i = 0; i <= steps; i++) {
            for (var j = 0; j <= steps; j++) {
                var w1 = -range + (2 * range * i / steps);
                var w2 = -range + (2 * range * j / steps);
                var val = totalAt(w1, w2);
                if (val < bestVal) {
                    bestVal = val;
                    bestW1 = w1;
                    bestW2 = w2;
                }
            }
        }
        // Refine with smaller grid around best
        var r2 = 2 * range / steps;
        for (var ii = 0; ii <= 40; ii++) {
            for (var jj = 0; jj <= 40; jj++) {
                var ww1 = bestW1 - r2 + (2 * r2 * ii / 40);
                var ww2 = bestW2 - r2 + (2 * r2 * jj / 40);
                var vv = totalAt(ww1, ww2);
                if (vv < bestVal) {
                    bestVal = vv;
                    bestW1 = ww1;
                    bestW2 = ww2;
                }
            }
        }
        state.optimum = {w1: bestW1, w2: bestW2, loss: lossAt(bestW1, bestW2), penalty: penaltyAt(bestW1, bestW2), total: bestVal};
    }

    // ========== 3D Projection ==========
    function project3d(x, y, z, pw, ph) {
        var cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
        var cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);

        // Rotate around Y axis (yaw)
        var x1 = cy * x + sy * z;
        var z1 = -sy * x + cy * z;
        var y1 = y;

        // Rotate around X axis (pitch)
        var y2 = cp * y1 - sp * z1;
        var z2 = sp * y1 + cp * z1;

        // Orthographic projection with scale
        var scale = pw * 0.14;
        var sx = pw / 2 + x1 * scale;
        var sy2 = ph / 2 - y2 * scale * 0.8 - z2 * scale * 0.05;
        return {x: sx, y: sy2, depth: z2};
    }

    // ========== Intersection Curve ==========
    function drawIntersectionCurve(ctx, pw, ph, heightScale, colors) {
        if (!state.showLoss || !state.showPenalty || state.regType === 'none' || state.lambda === 0) return;

        var range = 3;
        var res = 100;
        var step = 2 * range / res;

        // Find where loss(w1,w2) = lambda * penalty(w1,w2)
        function diffFn(w1, w2) {
            return lossAt(w1, w2) - state.lambda * penaltyAt(w1, w2);
        }

        var segments = [];
        for (var i = 0; i < res; i++) {
            for (var j = 0; j < res; j++) {
                var w1 = -range + i * step;
                var w2 = -range + j * step;

                var v00 = diffFn(w1, w2);
                var v10 = diffFn(w1 + step, w2);
                var v01 = diffFn(w1, w2 + step);
                var v11 = diffFn(w1 + step, w2 + step);

                var edges = [];
                if (v00 * v10 < 0) {
                    var t1 = v00 / (v00 - v10);
                    var ew1 = w1 + t1 * step, ew2 = w2;
                    var h1 = compress(lossAt(ew1, ew2)) * heightScale;
                    edges.push(project3d(ew1, h1, ew2, pw, ph));
                }
                if (v10 * v11 < 0) {
                    var t2 = v10 / (v10 - v11);
                    var ew1b = w1 + step, ew2b = w2 + t2 * step;
                    var h2 = compress(lossAt(ew1b, ew2b)) * heightScale;
                    edges.push(project3d(ew1b, h2, ew2b, pw, ph));
                }
                if (v01 * v11 < 0) {
                    var t3 = v01 / (v01 - v11);
                    var ew1c = w1 + t3 * step, ew2c = w2 + step;
                    var h3 = compress(lossAt(ew1c, ew2c)) * heightScale;
                    edges.push(project3d(ew1c, h3, ew2c, pw, ph));
                }
                if (v00 * v01 < 0) {
                    var t4 = v00 / (v00 - v01);
                    var ew1d = w1, ew2d = w2 + t4 * step;
                    var h4 = compress(lossAt(ew1d, ew2d)) * heightScale;
                    edges.push(project3d(ew1d, h4, ew2d, pw, ph));
                }

                if (edges.length >= 2) {
                    segments.push([edges[0], edges[1]]);
                    if (edges.length === 4) {
                        segments.push([edges[2], edges[3]]);
                    }
                }
            }
        }
        if (segments.length === 0) return;

        // White glow outline
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        for (var s = 0; s < segments.length; s++) {
            ctx.moveTo(segments[s][0].x, segments[s][0].y);
            ctx.lineTo(segments[s][1].x, segments[s][1].y);
        }
        ctx.stroke();

        // Main intersection line
        ctx.strokeStyle = colors.optimal;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (var s2 = 0; s2 < segments.length; s2++) {
            ctx.moveTo(segments[s2][0].x, segments[s2][0].y);
            ctx.lineTo(segments[s2][1].x, segments[s2][1].y);
        }
        ctx.stroke();
        ctx.lineCap = 'butt';
    }

    // ========== 3D Surface Rendering ==========
    function draw3d() {
        var colors = getCS();
        var dpr = window.devicePixelRatio || 1;
        var pw = canvas3d.width / dpr;
        var ph = canvas3d.height / dpr;

        ctx3d.save();
        ctx3d.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx3d.clearRect(0, 0, pw, ph);

        var gridSize = 40;
        var range = 3;
        var step = 2 * range / gridSize;

        // Compute grid values for each surface type
        var lossVals = [], penVals = [], totalVals = [];
        var maxLoss = 0, maxPen = 0, maxTotal = 0;
        for (var i = 0; i <= gridSize; i++) {
            lossVals[i] = [];
            penVals[i] = [];
            totalVals[i] = [];
            for (var j = 0; j <= gridSize; j++) {
                var w1 = -range + i * step;
                var w2 = -range + j * step;
                var l = compress(lossAt(w1, w2));
                var p = compress(state.lambda * penaltyAt(w1, w2));
                var t = compress(lossAt(w1, w2) + state.lambda * penaltyAt(w1, w2));
                lossVals[i][j] = l;
                penVals[i][j] = p;
                totalVals[i][j] = t;
                if (l > maxLoss) maxLoss = l;
                if (p > maxPen) maxPen = p;
                if (t > maxTotal) maxTotal = t;
            }
        }

        // Shared height scale across all surfaces so relative sizes are accurate
        var maxVal = Math.max(maxLoss, maxPen, maxTotal, 1);
        var heightScale = 3.0 / maxVal;
        var penColor = colors[state.regType] || colors.l1;

        // Collect faces from all visible surfaces
        var faces = [];
        function addSurfaceFaces(vals, surfaceType, surfaceMax) {
            for (var fi = 0; fi < gridSize; fi++) {
                for (var fj = 0; fj < gridSize; fj++) {
                    var w1a = -range + fi * step;
                    var w2a = -range + fj * step;
                    var w1b = w1a + step;
                    var w2b = w2a + step;

                    var v00 = vals[fi][fj] * heightScale;
                    var v10 = vals[fi+1][fj] * heightScale;
                    var v11 = vals[fi+1][fj+1] * heightScale;
                    var v01 = vals[fi][fj+1] * heightScale;

                    var p00 = project3d(w1a, v00, w2a, pw, ph);
                    var p10 = project3d(w1b, v10, w2a, pw, ph);
                    var p11 = project3d(w1b, v11, w2b, pw, ph);
                    var p01 = project3d(w1a, v01, w2b, pw, ph);

                    var avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) / 4;
                    // Normalize color intensity per surface's own max
                    var avgVal = (vals[fi][fj] + vals[fi+1][fj] + vals[fi+1][fj+1] + vals[fi][fj+1]) / 4;

                    faces.push({
                        pts: [p00, p10, p11, p01],
                        depth: avgDepth,
                        val: avgVal / surfaceMax,
                        type: surfaceType
                    });
                }
            }
        }

        if (state.showLoss) addSurfaceFaces(lossVals, 'loss', maxLoss);
        if (state.showPenalty && state.regType !== 'none') addSurfaceFaces(penVals, 'penalty', maxPen);
        if (state.showTotal) addSurfaceFaces(totalVals, 'total', maxTotal);

        // Sort by depth (far to near) for painter's algorithm
        faces.sort(function(a, b) { return a.depth - b.depth; });

        // Count visible surfaces for alpha adjustment
        var visCount = (state.showLoss ? 1 : 0) + (state.showPenalty && state.regType !== 'none' ? 1 : 0) + (state.showTotal ? 1 : 0);
        var multiSurface = visCount > 1;

        // Draw faces
        for (var fi2 = 0; fi2 < faces.length; fi2++) {
            var f = faces[fi2];
            var t = f.val;
            var fillColor;

            if (f.type === 'total') {
                // Blue → white → red diverging
                if (t < 0.5) fillColor = lerpColor(colors.cool, colors.mid, t * 2);
                else fillColor = lerpColor(colors.mid, colors.warm, (t - 0.5) * 2);
            } else if (f.type === 'loss') {
                // White/mid → green
                fillColor = lerpColor(colors.mid, colors.loss, Math.min(t * 1.8, 1));
            } else {
                // White/mid → reg-type color (red/blue/purple)
                fillColor = lerpColor(colors.mid, penColor, Math.min(t * 1.8, 1));
            }

            ctx3d.globalAlpha = multiSurface ? 0.55 : 0.9;
            ctx3d.beginPath();
            ctx3d.moveTo(f.pts[0].x, f.pts[0].y);
            ctx3d.lineTo(f.pts[1].x, f.pts[1].y);
            ctx3d.lineTo(f.pts[2].x, f.pts[2].y);
            ctx3d.lineTo(f.pts[3].x, f.pts[3].y);
            ctx3d.closePath();
            ctx3d.fillStyle = fillColor;
            ctx3d.fill();
            ctx3d.strokeStyle = colors.gridLine;
            ctx3d.lineWidth = 0.5;
            ctx3d.stroke();
        }
        ctx3d.globalAlpha = 1;

        // Draw intersection curve where loss = lambda * penalty
        drawIntersectionCurve(ctx3d, pw, ph, heightScale, colors);

        // Draw axes
        var axisLen = range;
        var origin = project3d(0, 0, 0, pw, ph);
        var xEnd = project3d(axisLen, 0, 0, pw, ph);
        var zEnd = project3d(0, 0, axisLen, pw, ph);
        var yEnd = project3d(0, 3, 0, pw, ph);

        ctx3d.strokeStyle = colors.axis;
        ctx3d.lineWidth = 1.5;
        ctx3d.globalAlpha = 0.4;

        // w1 axis
        ctx3d.beginPath();
        ctx3d.moveTo(origin.x, origin.y);
        ctx3d.lineTo(xEnd.x, xEnd.y);
        ctx3d.stroke();

        // w2 axis
        ctx3d.beginPath();
        ctx3d.moveTo(origin.x, origin.y);
        ctx3d.lineTo(zEnd.x, zEnd.y);
        ctx3d.stroke();

        // vertical L axis
        ctx3d.beginPath();
        ctx3d.moveTo(origin.x, origin.y);
        ctx3d.lineTo(yEnd.x, yEnd.y);
        ctx3d.stroke();

        ctx3d.globalAlpha = 1;

        // Axis labels
        ctx3d.font = 'bold ' + Math.max(11, pw * 0.022) + 'px sans-serif';
        ctx3d.fillStyle = colors.text;
        ctx3d.textAlign = 'center';
        ctx3d.fillText('w₀', xEnd.x + 10, xEnd.y);
        ctx3d.fillText('w₁', zEnd.x + 10, zEnd.y);
        ctx3d.fillText('L', yEnd.x - 10, yEnd.y - 2);

        // Draw optimum marker on surface
        if (state.optimum && state.showTotal) {
            var optH = compress(state.optimum.total) * heightScale;
            var optP = project3d(state.optimum.w1, optH, state.optimum.w2, pw, ph);
            ctx3d.beginPath();
            ctx3d.arc(optP.x, optP.y, 5, 0, Math.PI * 2);
            ctx3d.fillStyle = colors.optimal;
            ctx3d.fill();
            ctx3d.strokeStyle = '#fff';
            ctx3d.lineWidth = 2;
            ctx3d.stroke();

            var optGround = project3d(state.optimum.w1, 0, state.optimum.w2, pw, ph);
            ctx3d.beginPath();
            ctx3d.setLineDash([3, 3]);
            ctx3d.strokeStyle = colors.optimal;
            ctx3d.lineWidth = 1;
            ctx3d.moveTo(optP.x, optP.y);
            ctx3d.lineTo(optGround.x, optGround.y);
            ctx3d.stroke();
            ctx3d.setLineDash([]);
        }

        // Draw GD path on surface
        if (state.gdPath.length > 0) {
            // Trail
            if (state.gdPath.length > 1) {
            ctx3d.strokeStyle = colors.gd;
            ctx3d.lineWidth = 2;
            ctx3d.globalAlpha = 0.8;
            ctx3d.beginPath();
            for (var gi = 0; gi < state.gdPath.length; gi++) {
                var gp = state.gdPath[gi];
                var gh = compress(totalAt(gp.w1, gp.w2)) * heightScale;
                var gpt = project3d(gp.w1, gh, gp.w2, pw, ph);
                if (gi === 0) ctx3d.moveTo(gpt.x, gpt.y);
                else ctx3d.lineTo(gpt.x, gpt.y);
            }
            ctx3d.stroke();
            ctx3d.globalAlpha = 1;
            }

            // 3D ball at current position (with shadow)
            var cur = state.gdPath[state.gdPath.length - 1];
            var curH = compress(totalAt(cur.w1, cur.w2)) * heightScale;
            var curP = project3d(cur.w1, curH, cur.w2, pw, ph);
            // Shadow
            ctx3d.beginPath();
            ctx3d.ellipse(curP.x + 2, curP.y + 3, 7, 4, 0, 0, Math.PI * 2);
            ctx3d.fillStyle = 'rgba(0,0,0,0.2)';
            ctx3d.fill();
            var ballR = 8;
            var ballGrad = ctx3d.createRadialGradient(
                curP.x - ballR * 0.3, curP.y - ballR * 0.3, ballR * 0.1,
                curP.x, curP.y, ballR
            );
            ballGrad.addColorStop(0, '#fff');
            ballGrad.addColorStop(0.3, colors.gd);
            ballGrad.addColorStop(1, 'rgba(0,0,0,0.4)');
            ctx3d.beginPath();
            ctx3d.arc(curP.x, curP.y, ballR, 0, Math.PI * 2);
            ctx3d.fillStyle = ballGrad;
            ctx3d.fill();
        }

        // Title label
        var regLabels = {none: 'No Regularization', l1: 'L1 (Lasso)', l2: 'L2 (Ridge)', elastic: 'Elastic Net'};
        ctx3d.font = 'bold ' + Math.max(12, pw * 0.026) + 'px sans-serif';
        ctx3d.fillStyle = colors.textMuted;
        ctx3d.textAlign = 'left';
        ctx3d.textBaseline = 'top';
        ctx3d.fillText(regLabels[state.regType], 8, 8);

        // Drag/click hint
        ctx3d.font = Math.max(10, pw * 0.02) + 'px sans-serif';
        ctx3d.fillStyle = colors.textMuted;
        ctx3d.globalAlpha = 0.5;
        ctx3d.textAlign = 'right';
        ctx3d.fillText('click to drop \u00b7 drag to rotate', pw - 8, 10);
        ctx3d.globalAlpha = 1;

        ctx3d.restore();
    }

    // ========== 2D Contour Rendering ==========
    function drawContour() {
        var colors = getCS();
        var dpr = window.devicePixelRatio || 1;
        var pw = canvasContour.width / dpr;
        var ph = canvasContour.height / dpr;

        ctxContour.save();
        ctxContour.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctxContour.clearRect(0, 0, pw, ph);

        var range = 3;
        var pad = pw * 0.08;
        var plotW = pw - 2 * pad;
        var plotH = ph - 2 * pad;

        function toScreenX(w) { return pad + (w + range) / (2 * range) * plotW; }
        function toScreenY(w) { return pad + (range - w) / (2 * range) * plotH; }

        // Grid background
        ctxContour.strokeStyle = colors.zero;
        ctxContour.lineWidth = 1;
        // Zero lines
        ctxContour.beginPath();
        ctxContour.moveTo(toScreenX(0), pad);
        ctxContour.lineTo(toScreenX(0), pad + plotH);
        ctxContour.moveTo(pad, toScreenY(0));
        ctxContour.lineTo(pad + plotW, toScreenY(0));
        ctxContour.stroke();

        // Draw loss contours
        var contourLevels = [0.5, 1, 2, 3, 5, 8, 12, 18];
        ctxContour.strokeStyle = colors.contour;
        ctxContour.lineWidth = 1.5;

        for (var li = 0; li < contourLevels.length; li++) {
            var level = contourLevels[li];
            // Sample points on contour using marching
            drawContourLevel(ctxContour, function(w1, w2) { return lossAt(w1, w2); }, level, range, pad, plotW, plotH, colors.contour);
        }

        // Draw constraint region
        if (state.regType !== 'none' && state.lambda > 0) {
            var regColor = colors[state.regType] || colors.l1;

            // For the constraint interpretation: penalty(w) <= t
            // We draw the boundary of the feasible region at a few levels
            // The "budget" level that corresponds to the optimum
            if (state.optimum) {
                var budgetLevel = penaltyAt(state.optimum.w1, state.optimum.w2);
                drawConstraintRegion(ctxContour, budgetLevel, range, pad, plotW, plotH, regColor, colors.constraintFill);
            }
        }

        // Draw total loss contours (thin, dashed)
        if (state.regType !== 'none' && state.lambda > 0) {
            ctxContour.setLineDash([4, 3]);
            var totalColor = colors[state.regType] || colors.l1;
            for (var tli = 0; tli < contourLevels.length; tli++) {
                drawContourLevel(ctxContour, totalAt, contourLevels[tli] + state.lambda, range, pad, plotW, plotH, totalColor);
            }
            ctxContour.setLineDash([]);
        }

        // Draw loss center
        ctxContour.beginPath();
        ctxContour.arc(toScreenX(state.cx), toScreenY(state.cy), 4, 0, Math.PI * 2);
        ctxContour.fillStyle = colors.loss;
        ctxContour.fill();
        ctxContour.strokeStyle = '#fff';
        ctxContour.lineWidth = 1.5;
        ctxContour.stroke();

        // Draw optimum
        if (state.optimum) {
            var ox = toScreenX(state.optimum.w1);
            var oy = toScreenY(state.optimum.w2);

            // Star marker
            ctxContour.beginPath();
            for (var si = 0; si < 10; si++) {
                var a = si * Math.PI / 5 - Math.PI / 2;
                var r = si % 2 === 0 ? 7 : 3;
                if (si === 0) ctxContour.moveTo(ox + r * Math.cos(a), oy + r * Math.sin(a));
                else ctxContour.lineTo(ox + r * Math.cos(a), oy + r * Math.sin(a));
            }
            ctxContour.closePath();
            ctxContour.fillStyle = colors.optimal;
            ctxContour.fill();
            ctxContour.strokeStyle = '#fff';
            ctxContour.lineWidth = 1.5;
            ctxContour.stroke();

            // Coordinate label
            ctxContour.font = '600 10px ' + getComputedStyle(document.documentElement).getPropertyValue('--viz-mono-font').trim();
            ctxContour.fillStyle = colors.optimal;
            ctxContour.textAlign = 'left';
            ctxContour.fillText('(' + state.optimum.w1.toFixed(2) + ', ' + state.optimum.w2.toFixed(2) + ')', ox + 10, oy - 4);
        }

        // Draw GD path
        if (state.gdPath.length > 0) {
            if (state.gdPath.length > 1) {
            ctxContour.strokeStyle = colors.gd;
            ctxContour.lineWidth = 2;
            ctxContour.globalAlpha = 0.8;
            ctxContour.beginPath();
            for (var gi = 0; gi < state.gdPath.length; gi++) {
                var gpx = toScreenX(state.gdPath[gi].w1);
                var gpy = toScreenY(state.gdPath[gi].w2);
                if (gi === 0) ctxContour.moveTo(gpx, gpy);
                else ctxContour.lineTo(gpx, gpy);
            }
            ctxContour.stroke();
            ctxContour.globalAlpha = 1;

            // Start marker
            ctxContour.beginPath();
            ctxContour.arc(toScreenX(state.gdPath[0].w1), toScreenY(state.gdPath[0].w2), 3, 0, Math.PI * 2);
            ctxContour.fillStyle = colors.textMuted;
            ctxContour.fill();
            }

            // Current ball
            var gcur = state.gdPath[state.gdPath.length - 1];
            ctxContour.beginPath();
            ctxContour.arc(toScreenX(gcur.w1), toScreenY(gcur.w2), 5, 0, Math.PI * 2);
            ctxContour.fillStyle = colors.gd;
            ctxContour.fill();
            ctxContour.strokeStyle = '#fff';
            ctxContour.lineWidth = 1.5;
            ctxContour.stroke();
        }

        // Axis labels
        ctxContour.font = 'bold ' + Math.max(11, pw * 0.025) + 'px sans-serif';
        ctxContour.fillStyle = colors.text;
        ctxContour.textAlign = 'center';
        ctxContour.textBaseline = 'top';
        ctxContour.fillText('w₀', pad + plotW / 2, pad + plotH + 4);
        ctxContour.save();
        ctxContour.translate(pad - 6, pad + plotH / 2);
        ctxContour.rotate(-Math.PI / 2);
        ctxContour.textBaseline = 'bottom';
        ctxContour.fillText('w₁', 0, 0);
        ctxContour.restore();

        // Legend
        var legendY = ph - 6;
        ctxContour.font = '10px sans-serif';
        ctxContour.textBaseline = 'bottom';
        ctxContour.textAlign = 'left';

        // Loss contour swatch
        ctxContour.fillStyle = colors.contour;
        ctxContour.fillRect(pad, legendY - 8, 14, 3);
        ctxContour.fillStyle = colors.textMuted;
        ctxContour.fillText('Loss contours', pad + 18, legendY);

        // Optimum swatch
        ctxContour.fillStyle = colors.optimal;
        ctxContour.fillRect(pad + 105, legendY - 8, 8, 8);
        ctxContour.fillStyle = colors.textMuted;
        ctxContour.fillText('Optimum', pad + 117, legendY);

        // Unregularized swatch
        ctxContour.fillStyle = colors.loss;
        ctxContour.beginPath();
        ctxContour.arc(pad + 186, legendY - 4, 4, 0, Math.PI * 2);
        ctxContour.fill();
        ctxContour.fillStyle = colors.textMuted;
        ctxContour.fillText('Unreg. min', pad + 194, legendY);

        ctxContour.restore();
    }

    // Draw a single contour level using marching squares (simplified)
    function drawContourLevel(ctx, fn, level, range, pad, plotW, plotH, color) {
        var res = 80;
        var step = 2 * range / res;

        function toSX(w) { return pad + (w + range) / (2 * range) * plotW; }
        function toSY(w) { return pad + (range - w) / (2 * range) * plotH; }

        ctx.strokeStyle = color;
        ctx.beginPath();

        for (var i = 0; i < res; i++) {
            for (var j = 0; j < res; j++) {
                var w1 = -range + i * step;
                var w2 = -range + j * step;

                var v00 = fn(w1, w2) - level;
                var v10 = fn(w1 + step, w2) - level;
                var v01 = fn(w1, w2 + step) - level;
                var v11 = fn(w1 + step, w2 + step) - level;

                // Simple edge crossing detection
                var edges = [];
                if (v00 * v10 < 0) {
                    var t = v00 / (v00 - v10);
                    edges.push({x: toSX(w1 + t * step), y: toSY(w2)});
                }
                if (v10 * v11 < 0) {
                    var t2 = v10 / (v10 - v11);
                    edges.push({x: toSX(w1 + step), y: toSY(w2 + t2 * step)});
                }
                if (v01 * v11 < 0) {
                    var t3 = v01 / (v01 - v11);
                    edges.push({x: toSX(w1 + t3 * step), y: toSY(w2 + step)});
                }
                if (v00 * v01 < 0) {
                    var t4 = v00 / (v00 - v01);
                    edges.push({x: toSX(w1), y: toSY(w2 + t4 * step)});
                }

                if (edges.length >= 2) {
                    ctx.moveTo(edges[0].x, edges[0].y);
                    ctx.lineTo(edges[1].x, edges[1].y);
                    if (edges.length === 4) {
                        ctx.moveTo(edges[2].x, edges[2].y);
                        ctx.lineTo(edges[3].x, edges[3].y);
                    }
                }
            }
        }
        ctx.stroke();
    }

    // Draw filled constraint region
    function drawConstraintRegion(ctx, budget, range, pad, plotW, plotH, strokeColor, fillColor) {
        function toSX(w) { return pad + (w + range) / (2 * range) * plotW; }
        function toSY(w) { return pad + (range - w) / (2 * range) * plotH; }

        ctx.save();
        ctx.beginPath();

        if (state.regType === 'l1') {
            // Diamond: |w1| + |w2| = budget
            var r = budget;
            ctx.moveTo(toSX(r), toSY(0));
            ctx.lineTo(toSX(0), toSY(r));
            ctx.lineTo(toSX(-r), toSY(0));
            ctx.lineTo(toSX(0), toSY(-r));
            ctx.closePath();
        } else if (state.regType === 'l2') {
            // Circle: w1^2 + w2^2 = budget
            var rad = Math.sqrt(budget);
            var cx = toSX(0), cy2 = toSY(0);
            var rx = rad / (2 * range) * plotW;
            var ry = rad / (2 * range) * plotH;
            ctx.ellipse(cx, cy2, rx, ry, 0, 0, Math.PI * 2);
        } else if (state.regType === 'elastic') {
            // Parametric: sample boundary
            var pts = 200;
            var first = true;
            for (var ai = 0; ai <= pts; ai++) {
                var angle = ai * 2 * Math.PI / pts;
                // Binary search for radius at this angle where penalty = budget
                var lo = 0, hi = 5;
                for (var bs = 0; bs < 30; bs++) {
                    var mid = (lo + hi) / 2;
                    var tw1 = mid * Math.cos(angle);
                    var tw2 = mid * Math.sin(angle);
                    if (penaltyAt(tw1, tw2) < budget) lo = mid;
                    else hi = mid;
                }
                var rr = (lo + hi) / 2;
                var px = rr * Math.cos(angle);
                var py = rr * Math.sin(angle);
                if (first) { ctx.moveTo(toSX(px), toSY(py)); first = false; }
                else ctx.lineTo(toSX(px), toSY(py));
            }
            ctx.closePath();
        }

        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.restore();
    }

    // ========== Math Panel ==========
    function updateMathPanel() {
        var panel = document.getElementById('reg-math-panel');
        if (!state.optimum) {
            panel.innerHTML = '<span class="formula-note">Adjust parameters to see the computation.</span>';
            return;
        }
        var o = state.optimum;
        var regLabels = {none: 'None', l1: 'L1', l2: 'L2', elastic: 'Elastic Net'};

        var html = '';
        html += '<div class="reg-calc-row"><span>Reg Type:</span><span>' + regLabels[state.regType] + '</span></div>';
        html += '<div class="reg-calc-row"><span>&lambda;:</span><span>' + state.lambda.toFixed(2) + '</span></div>';

        html += '<div class="reg-calc-row"><span>Unreg. min:</span><span class="reg-val-loss">(' + state.cx.toFixed(2) + ', ' + state.cy.toFixed(2) + ')</span></div>';
        html += '<div class="reg-calc-row"><span>Reg. optimum:</span><span class="reg-val-optimal">(' + o.w1.toFixed(3) + ', ' + o.w2.toFixed(3) + ')</span></div>';

        html += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--viz-border)"></div>';
        html += '<div class="reg-calc-row"><span>Loss L(w):</span><span class="reg-val-loss">' + o.loss.toFixed(4) + '</span></div>';
        html += '<div class="reg-calc-row"><span>Penalty R(w):</span><span class="reg-val-l1">' + o.penalty.toFixed(4) + '</span></div>';
        html += '<div class="reg-calc-row"><span>&lambda; &middot; R(w):</span><span>' + (state.lambda * o.penalty).toFixed(4) + '</span></div>';
        html += '<div class="reg-calc-row reg-calc-result"><span>Total:</span><span>' + o.total.toFixed(4) + '</span></div>';

        // Sparsity check
        var thresh = 0.03;
        var w1Zero = Math.abs(o.w1) < thresh;
        var w2Zero = Math.abs(o.w2) < thresh;
        if (w1Zero || w2Zero) {
            html += '<div style="margin-top:8px;padding:6px 8px;border-radius:4px;background:var(--viz-success-bg);border:1px solid var(--viz-success-border);font-size:11px;color:var(--viz-success-border);">';
            html += '<i class="fa fa-check-circle"></i> Sparse solution: ';
            if (w1Zero && w2Zero) html += 'both weights &asymp; 0';
            else if (w1Zero) html += 'w<sub>0</sub> &asymp; 0';
            else html += 'w<sub>1</sub> &asymp; 0';
            html += '</div>';
        }

        panel.innerHTML = html;
    }

    // ========== Canvas Setup ==========
    function setupCanvas(canvas, aspectRatio) {
        var dpr = window.devicePixelRatio || 1;
        var parentW = canvas.parentElement.clientWidth;
        var h = parentW * (aspectRatio || 0.75);
        canvas.style.width = parentW + 'px';
        canvas.style.height = h + 'px';
        canvas.width = parentW * dpr;
        canvas.height = h * dpr;
    }

    function updatePenaltySwatch() {
        var colors = getCS();
        var swatch = document.getElementById('reg-penalty-swatch');
        if (swatch) swatch.style.background = colors[state.regType] || colors.l1;
    }

    function redraw() {
        findOptimum();
        var togglesDiv = document.getElementById('reg-surface-toggles');
        if (state.viewMode === '3d') {
            canvas3d.style.display = 'block';
            canvasContour.style.display = 'none';
            if (togglesDiv) togglesDiv.style.display = 'flex';
            setupCanvas(canvas3d, 0.75);
            updatePenaltySwatch();
            draw3d();
        } else {
            canvas3d.style.display = 'none';
            canvasContour.style.display = 'block';
            if (togglesDiv) togglesDiv.style.display = 'none';
            setupCanvas(canvasContour, 0.65);
            drawContour();
        }
        updateMathPanel();
        updateLambdaBadge();
    }

    function updateLambdaBadge() {
        document.getElementById('reg-lambda-badge').innerHTML = '&lambda; = ' + state.lambda.toFixed(2);
    }

    // ========== Event Wiring ==========
    function init() {
        canvas3d = document.getElementById('reg-3d-canvas');
        canvasContour = document.getElementById('reg-contour-canvas');
        ctx3d = canvas3d.getContext('2d');
        ctxContour = canvasContour.getContext('2d');

        redraw();

        // Theme changes
        if (window.VizLib && window.VizLib.ThemeManager) {
            window.VizLib.ThemeManager.onThemeChange(function() { redraw(); });
        }

        // Reg type buttons
        var regBtns = document.querySelectorAll('.reg-type-toggle .btn');
        regBtns.forEach(function(btn) {
            btn.addEventListener('click', function() {
                regBtns.forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                state.regType = btn.getAttribute('data-reg');
                redraw();
            });
        });

        // Surface toggle buttons
        document.querySelectorAll('.reg-surface-toggle').forEach(function(label) {
            label.addEventListener('click', function() {
                var surface = label.getAttribute('data-surface');
                label.classList.toggle('active');
                if (surface === 'loss') state.showLoss = label.classList.contains('active');
                else if (surface === 'penalty') state.showPenalty = label.classList.contains('active');
                else if (surface === 'total') state.showTotal = label.classList.contains('active');
                redraw();
            });
        });

        // View toggle buttons
        var viewBtns = document.querySelectorAll('.reg-view-toggle .btn');
        viewBtns.forEach(function(btn) {
            btn.addEventListener('click', function() {
                viewBtns.forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                state.viewMode = btn.getAttribute('data-view');
                redraw();
            });
        });

        // 3D drag rotation + click-to-drop
        var dragDist = 0;
        canvas3d.addEventListener('mousedown', function(e) {
            state.dragging = true;
            dragDist = 0;
            state.lastMouse = {x: e.clientX, y: e.clientY};
        });

        window.addEventListener('mousemove', function(e) {
            if (!state.dragging) return;
            var dx = e.clientX - state.lastMouse.x;
            var dy = e.clientY - state.lastMouse.y;
            dragDist += Math.abs(dx) + Math.abs(dy);
            state.yaw += dx * 0.01;
            state.pitch = Math.max(0.15, Math.min(1.4, state.pitch + dy * 0.01));
            state.lastMouse = {x: e.clientX, y: e.clientY};
            draw3d();
        });

        window.addEventListener('mouseup', function() {
            state.dragging = false;
        });

        // Click to drop ball (3D) — inverse projection onto ground plane
        canvas3d.addEventListener('click', function(e) {
            if (dragDist > 5) return; // was a drag, not a click
            var rect = canvas3d.getBoundingClientRect();
            var sx = e.clientX - rect.left;
            var sy = e.clientY - rect.top;
            var pw = rect.width, ph = rect.height;
            var scale = pw * 0.14;
            var cy = Math.cos(state.yaw), sy2 = Math.sin(state.yaw);
            var cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
            // Invert screen → rotated coords on ground plane (y=0)
            var x1 = (sx - pw / 2) / scale;
            // sy = ph/2 - y2*scale*0.8 - z2*scale*0.05
            // with y=0: y2 = -sp*z1, z2 = cp*z1
            // sy = ph/2 + sp*z1*scale*0.8 - cp*z1*scale*0.05
            var zCoeff = scale * (sp * 0.8 - cp * 0.05);
            var z1 = (zCoeff !== 0) ? (sy - ph / 2) / (-zCoeff) : 0;
            // Invert yaw rotation: x1 = cy*w0 + sy*w1, z1 = -sy*w0 + cy*w1
            var w0 = cy * x1 - sy2 * z1;
            var w1 = sy2 * x1 + cy * z1;
            startGD(w0, w1);
        });

        // Click to drop ball (2D contour) — precise mapping
        canvasContour.addEventListener('click', function(e) {
            var rect = canvasContour.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            var range = 3;
            var padFrac = 0.08;
            var pad = rect.width * padFrac;
            var plotW = rect.width - 2 * pad;
            var plotH = rect.height - 2 * pad;
            var w0 = (x - pad) / plotW * (2 * range) - range;
            var w1 = range - (y - pad) / plotH * (2 * range);
            startGD(w0, w1);
        });

        // Touch support for 3D rotation
        var touchDragDist = 0;
        canvas3d.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                state.dragging = true;
                touchDragDist = 0;
                state.lastMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
                e.preventDefault();
            }
        }, {passive: false});

        window.addEventListener('touchmove', function(e) {
            if (!state.dragging || e.touches.length !== 1) return;
            var dx = e.touches[0].clientX - state.lastMouse.x;
            var dy = e.touches[0].clientY - state.lastMouse.y;
            touchDragDist += Math.abs(dx) + Math.abs(dy);
            state.yaw += dx * 0.01;
            state.pitch = Math.max(0.15, Math.min(1.4, state.pitch + dy * 0.01));
            state.lastMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
            draw3d();
        }, {passive: true});

        window.addEventListener('touchend', function() {
            state.dragging = false;
        });

        // Sliders
        function wireSlider(id, key, displayId, formatter) {
            var slider = document.getElementById(id);
            var display = document.getElementById(displayId);
            if (!slider) return;
            slider.addEventListener('input', function() {
                var v = parseFloat(slider.value);
                state[key] = v;
                display.innerHTML = formatter ? formatter(v) : v.toFixed(2);
                redraw();
            });
        }

        wireSlider('reg-lambda-slider', 'lambda', 'reg-lambda-value');
        wireSlider('reg-ecc-slider', 'eccentricity', 'reg-ecc-value', function(v) { return v.toFixed(1); });
        wireSlider('reg-cx-slider', 'cx', 'reg-cx-value');
        wireSlider('reg-cy-slider', 'cy', 'reg-cy-value');
        wireSlider('reg-alpha-slider', 'alpha', 'reg-alpha-value');
        wireSlider('reg-rot-slider', 'rotation', 'reg-rot-value', function(v) { return v + '&deg;'; });
        wireSlider('reg-lr-slider', 'gdLR', 'reg-lr-value', function(v) { return v.toFixed(3); });

        // Clear GD path
        document.getElementById('reg-clear-path-btn').addEventListener('click', function() {
            state.gdPath = [];
            state.gdAnimating = false;
            if (state.gdAnimFrame) cancelAnimationFrame(state.gdAnimFrame);
            redraw();
        });

        // Reset
        document.getElementById('reg-reset-btn').addEventListener('click', function() {
            state.regType = 'l1'; state.lambda = 1.0; state.eccentricity = 4.0;
            state.cx = 1.5; state.cy = 1.0; state.alpha = 0.5; state.rotation = 30;
            state.yaw = -0.6; state.pitch = 0.7;

            document.getElementById('reg-lambda-slider').value = 1;
            document.getElementById('reg-lambda-value').textContent = '1.00';
            document.getElementById('reg-ecc-slider').value = 4;
            document.getElementById('reg-ecc-value').textContent = '4.0';
            document.getElementById('reg-cx-slider').value = 1.5;
            document.getElementById('reg-cx-value').textContent = '1.50';
            document.getElementById('reg-cy-slider').value = 1;
            document.getElementById('reg-cy-value').textContent = '1.00';
            document.getElementById('reg-alpha-slider').value = 0.5;
            document.getElementById('reg-alpha-value').textContent = '0.50';
            document.getElementById('reg-rot-slider').value = 30;
            document.getElementById('reg-rot-value').innerHTML = '30&deg;';

            state.viewMode = '3d';
            state.showLoss = true;
            state.showPenalty = true;
            state.showTotal = false;
            state.gdPath = [];
            state.gdAnimating = false;
            state.gdLR = 0.05;
            if (state.gdAnimFrame) cancelAnimationFrame(state.gdAnimFrame);

            document.getElementById('reg-lr-slider').value = 0.05;
            document.getElementById('reg-lr-value').textContent = '0.050';

            var regBtns2 = document.querySelectorAll('.reg-type-toggle .btn');
            regBtns2.forEach(function(b) {
                b.classList.toggle('active', b.getAttribute('data-reg') === 'l1');
            });
            var viewBtns2 = document.querySelectorAll('.reg-view-toggle .btn');
            viewBtns2.forEach(function(b) {
                b.classList.toggle('active', b.getAttribute('data-view') === '3d');
            });
            document.querySelectorAll('.reg-surface-toggle').forEach(function(label) {
                var s = label.getAttribute('data-surface');
                label.classList.toggle('active', s === 'loss' || s === 'penalty');
            });

            redraw();
        });

        // Resize
        var resizeTimer;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(redraw, 100);
        });

        // Info tab switching
        document.querySelectorAll('.info-panel-tabs .btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var tabId = btn.getAttribute('data-tab');
                btn.closest('.info-panel-tabs').querySelectorAll('.btn').forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                var panel = btn.closest('.panel');
                panel.querySelectorAll('.info-tab-content').forEach(function(t) { t.classList.remove('active'); });
                var target = panel.querySelector('#tab-' + tabId);
                if (target) target.classList.add('active');
            });
        });
    }

    // ========== Entry Point ==========
    if (window.VizLib && window.VizLib._ready) {
        init();
    } else {
        window.addEventListener('vizlib-ready', init);
    }
})();
