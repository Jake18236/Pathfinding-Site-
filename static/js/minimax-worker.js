/**
 * Minimax Tree Parsing Web Worker
 *
 * Handles CPU-intensive operations off the main thread:
 * - Binary data decompression
 * - Tree structure parsing
 * - Building parent-child relationships
 *
 * Layout is computed dynamically in main thread based on visible nodes.
 * Uses typed arrays for efficient data transfer back to main thread.
 */

'use strict';

// ============================================
// Message Handler
// ============================================

self.onmessage = async function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'parse':
            try {
                const result = await parseTreeData(data);
                // Transfer typed arrays without copying
                self.postMessage({ type: 'complete', data: result }, [
                    result.nodeDepths.buffer,
                    result.nodeFlags.buffer,
                    result.nodeOptimalBits.buffer,
                    result.nodeBoards.buffer,
                    result.nodeMoves.buffer,
                    result.nodeValuesOffsets.buffer,
                    result.edgeSources.buffer,
                    result.edgeTargets.buffer,
                    result.childOffsets.buffer,
                    result.allChildren.buffer
                ]);
            } catch (error) {
                self.postMessage({ type: 'error', error: error.message });
            }
            break;
    }
};

// ============================================
// Main Parsing (No Layout Computation)
// ============================================

async function parseTreeData({ buffer }) {
    const dataView = new DataView(buffer);
    let offset = 0;

    // Read header
    const nodeCount = dataView.getUint32(offset, true); offset += 4;
    const edgeCount = dataView.getUint32(offset, true); offset += 4;
    const maxDepth = dataView.getUint32(offset, true); offset += 4;
    offset += 8; // reserved

    const bytesPerNode = 4 + 4 + 4 + 1 + 1 + 1 + 9 + maxDepth * 2 + 2;
    const nodeDataOffset = offset;
    const edgeDataOffset = nodeDataOffset + nodeCount * bytesPerNode;

    self.postMessage({ type: 'progress', phase: 'parse', progress: 5, detail: 'Header read' });

    // Allocate output typed arrays
    const nodeDepths = new Uint8Array(nodeCount);
    const nodeFlags = new Uint8Array(nodeCount);  // bit 0: isMaximizing, bit 1: isTerminal
    const nodeOptimalBits = new Uint16Array(nodeCount);
    const nodeBoards = new Uint8Array(nodeCount * 9);  // 9 cells per board
    const nodeMoves = new Uint8Array(nodeCount);  // move that led to this node (255 = none)
    const nodeValuesOffsets = new Uint32Array(nodeCount);  // offset to values in original buffer

    const edgeSources = new Uint32Array(edgeCount);
    const edgeTargets = new Uint32Array(edgeCount);

    // Build children index from edges
    const childCounts = new Uint32Array(nodeCount);
    const childOffsets = new Uint32Array(nodeCount + 1);
    const allChildren = new Uint32Array(edgeCount);

    self.postMessage({ type: 'progress', phase: 'parse', progress: 10, detail: 'Reading edges...' });

    // First pass: count children per node and store edges
    let edgeOffset = edgeDataOffset;
    for (let i = 0; i < edgeCount; i++) {
        const source = dataView.getUint32(edgeOffset, true);
        edgeSources[i] = source;
        edgeTargets[i] = dataView.getUint32(edgeOffset + 4, true);
        childCounts[source]++;
        edgeOffset += 8;

        if (i > 0 && i % 100000 === 0) {
            self.postMessage({
                type: 'progress',
                phase: 'parse',
                progress: 10 + (i / edgeCount) * 20,
                detail: `Edges: ${(i/1000).toFixed(0)}K/${(edgeCount/1000).toFixed(0)}K`
            });
        }
    }

    // Build offsets (prefix sum)
    childOffsets[0] = 0;
    for (let i = 0; i < nodeCount; i++) {
        childOffsets[i + 1] = childOffsets[i] + childCounts[i];
    }

    // Reset counts to use as insertion indices
    childCounts.fill(0);

    self.postMessage({ type: 'progress', phase: 'parse', progress: 35, detail: 'Building children index...' });

    // Second pass: fill children array
    for (let i = 0; i < edgeCount; i++) {
        const source = edgeSources[i];
        const target = edgeTargets[i];
        const insertIdx = childOffsets[source] + childCounts[source];
        allChildren[insertIdx] = target;
        childCounts[source]++;
    }

    self.postMessage({ type: 'progress', phase: 'nodes', progress: 50, detail: 'Reading node data...' });

    // Read all node data
    offset = nodeDataOffset;
    for (let i = 0; i < nodeCount; i++) {
        const id = dataView.getUint32(offset, true);
        const depth = dataView.getUint8(offset + 12);
        const flags = dataView.getUint8(offset + 13);
        const moveRaw = dataView.getUint8(offset + 14);

        nodeDepths[id] = depth;
        nodeFlags[id] = flags;
        nodeMoves[id] = moveRaw;

        // Store values offset for on-demand reading
        nodeValuesOffsets[id] = offset + 24;

        // Read optimal bits
        nodeOptimalBits[id] = dataView.getUint16(offset + 24 + maxDepth * 2, true);

        // Read board - use id for indexing since that's where data should go
        // The binary stores nodes sequentially, but we index by id
        const boardOffset = id * 9;
        for (let j = 0; j < 9; j++) {
            nodeBoards[boardOffset + j] = dataView.getUint8(offset + 15 + j);
        }

        offset += bytesPerNode;

        if (i > 0 && i % 100000 === 0) {
            self.postMessage({
                type: 'progress',
                phase: 'nodes',
                progress: 50 + (i / nodeCount) * 45,
                detail: `Nodes: ${(i/1000).toFixed(0)}K/${(nodeCount/1000).toFixed(0)}K`
            });
        }
    }

    self.postMessage({ type: 'progress', phase: 'nodes', progress: 95, detail: 'Finalizing...' });

    return {
        nodeCount,
        edgeCount,
        maxDepth,
        nodeDepths,
        nodeFlags,
        nodeOptimalBits,
        nodeBoards,
        nodeMoves,
        nodeValuesOffsets,
        edgeSources,
        edgeTargets,
        childOffsets,
        allChildren,
        bufferByteLength: buffer.byteLength
    };
}
