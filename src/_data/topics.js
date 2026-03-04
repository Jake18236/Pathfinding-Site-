const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

module.exports = async function() {
  const csvPath = path.join(process.cwd(), 'static', 'csv', 'topics.csv');

  if (!fs.existsSync(csvPath)) {
    console.warn('[11ty] topics.csv not found at', csvPath);
    return { nodes: [], links: [], groups: [] };
  }

  const csvText = fs.readFileSync(csvPath, 'utf8');
  const records = parse(csvText, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  });

  const nodeMap = new Map();
  const nodes = [];

  for (const row of records) {
    const id = (row.id || '').trim();
    const name = (row.name || '').trim();
    if (!id || !name) continue;

    const node = {
      id,
      name,
      group: (row.group || 'Other').trim(),
      size: parseInt(row.size, 10) || 1,
      description: (row.description || '').trim(),
      resources: (row.resources || '').trim() ? row.resources.trim().split('|').map(s => s.trim()).filter(Boolean) : [],
      visualization: (row.visualization || '').trim() || null
    };

    nodes.push(node);
    nodeMap.set(id, node);
  }

  // Build deduplicated links from connections column
  const linkSet = new Set();
  const links = [];

  for (const row of records) {
    const sourceId = (row.id || '').trim();
    const connections = (row.connections || '').trim();
    if (!sourceId || !connections) continue;

    for (const targetId of connections.split('|').map(s => s.trim()).filter(Boolean)) {
      if (!nodeMap.has(targetId)) {
        console.warn(`[11ty] topics: node "${sourceId}" references unknown connection "${targetId}"`);
        continue;
      }

      // Deduplicate: sort the pair so A-B and B-A produce the same key
      const key = [sourceId, targetId].sort().join('::');
      if (linkSet.has(key)) continue;
      linkSet.add(key);

      links.push({ source: sourceId, target: targetId });
    }
  }

  // Collect unique sorted groups
  const groups = [...new Set(nodes.map(n => n.group))].sort();

  console.log(`[11ty] Topics graph: ${nodes.length} nodes, ${links.length} links, ${groups.length} groups`);

  return { nodes, links, groups };
};
