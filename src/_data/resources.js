const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

/**
 * Load resources from CSV for the landing page
 * Returns a mapping: { export_id: [ {name, type, link, author, date_added_*, ...}, ... ] }
 *
 * Special handling for:
 * - learning resources (export_id contains 'learning')
 * - library items (export_id starts with 'lib-' or contains 'library')
 */

// Helper functions
function normalizeHeaders(headers) {
  return headers.map(h => h.trim().toLowerCase().replace(/ /g, '_'));
}

function bestOf(row, ...candidates) {
  for (const c of candidates) {
    if (row[c] && row[c].trim()) {
      return row[c].trim();
    }
  }
  return '';
}

function toBool(s) {
  if (!s) return false;
  s = s.trim().toLowerCase();
  return ['1', 'true', 'yes', 'y', 'checked out', 'checked_out'].includes(s);
}

// Parse date string and return date info object
function parseDateAdded(dateStr) {
  if (!dateStr) return null;

  try {
    // Try various date formats
    let date;

    // ISO format: 2024-01-15
    if (/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
      date = new Date(dateStr);
    }
    // US format: 01/15/2024 or 1/15/2024
    else if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
      const parts = dateStr.split('/');
      date = new Date(parseInt(parts[2]), parseInt(parts[0]) - 1, parseInt(parts[1]));
    }
    // Year only: 2024
    else if (/^\d{4}$/.test(dateStr)) {
      date = new Date(parseInt(dateStr), 0, 1);
    }
    // Try native parsing
    else {
      date = new Date(dateStr);
    }

    if (isNaN(date.getTime())) return null;

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    return {
      date_added_raw: dateStr,
      date_added_display: `${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`,
      date_added_sort_key: date.toISOString().slice(0, 10), // YYYY-MM-DD for sorting
      date_added_year: date.getFullYear().toString()
    };
  } catch (err) {
    return null;
  }
}

// Find book cover thumbnail
function findCoverImage(name, coverField) {
  if (coverField) return coverField;

  // Try to find in book_cover_thumbnails directory
  const thumbnailDir = path.join(process.cwd(), 'static', 'img', 'book_cover_thumbnails');
  if (!fs.existsSync(thumbnailDir)) return '';

  // Slugify the name for matching
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

  const extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif'];
  for (const ext of extensions) {
    const tryPath = path.join(thumbnailDir, slug + ext);
    if (fs.existsSync(tryPath)) {
      return `static/img/book_cover_thumbnails/${slug}${ext}`;
    }
  }

  return '';
}

function loadResourcesFromCSV(text) {
  const records = parse(text, {
    columns: (headers) => normalizeHeaders(headers),
    skip_empty_lines: true,
    trim: true
  });

  const out = {};
  let totalRows = 0;
  let keptRows = 0;

  for (const row of records) {
    totalRows++;

    // Get export_id (determines grouping)
    const exportId = bestOf(row, 'export_id', 'lecture_id', 'id', 'category');
    const name = bestOf(row, 'name', 'title', 'resource', 'resource_name');
    const link = bestOf(row, 'link', 'url', 'href');
    const rtype = bestOf(row, 'type', 'category', 'format') || 'reading';
    const author = bestOf(row, 'author', 'creator', 'by');

    // Student credit (for student-submitted resources)
    const student = bestOf(row, 'student', 'student_name', 'submitted_by', 'attribution', 'credit');

    // Date added
    const dateAddedRaw = bestOf(row, 'date_added', 'date', 'added_date', 'created');
    const dateInfo = parseDateAdded(dateAddedRaw);

    // Library-specific fields
    const cover = bestOf(row, 'cover', 'image', 'thumbnail', 'cover_image');
    const checkedOut = toBool(bestOf(row, 'checked_out', 'checkedout', 'status'));
    const rating = bestOf(row, 'rating', 'stars', 'score');
    const notes = bestOf(row, 'notes', 'description', 'summary');
    const media = bestOf(row, 'media');

    // Skip rows without export_id or name
    if (!exportId || !name) continue;

    const entry = {
      name,
      type: rtype
    };

    if (link) entry.link = link;
    if (author) entry.author = author;
    if (student) entry.student = student;

    // Add date info
    if (dateInfo) {
      Object.assign(entry, dateInfo);
    }

    // Library-specific fields
    const resolvedCover = findCoverImage(name, cover);
    if (resolvedCover) entry.cover = resolvedCover;
    if (checkedOut) entry.checked_out = true;
    if (rating) entry.rating = rating;
    if (notes) entry.notes = notes;
    if (media) entry.media = media;

    if (!out[exportId]) out[exportId] = [];
    out[exportId].push(entry);
    keptRows++;
  }

  // Deduplicate entries within each export_id
  for (const [k, items] of Object.entries(out)) {
    const seen = new Set();
    const deduped = [];
    for (const it of items) {
      const sig = `${it.type || ''}|${it.name || ''}|${it.link || ''}`;
      if (seen.has(sig)) continue;
      seen.add(sig);
      deduped.push(it);
    }
    out[k] = deduped;
  }

  console.log(`[11ty] CSV resources: rows=${totalRows}, kept=${keptRows}, groups=${Object.keys(out).length}`);
  return out;
}

module.exports = async function() {
  // Try local CSV file
  const localCsvPath = path.join(process.cwd(), 'static', 'csv', 'resources.csv');

  try {
    if (fs.existsSync(localCsvPath)) {
      const csvText = fs.readFileSync(localCsvPath, 'utf8');
      return loadResourcesFromCSV(csvText);
    }
  } catch (err) {
    console.error('Warning: Could not read local resources.csv:', err.message);
  }

  console.warn('Warning: No resources CSV found at', localCsvPath);
  return {};
};
