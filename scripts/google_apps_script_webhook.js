/**
 * Google Apps Script - Tweet Queue Webhook
 *
 * SETUP INSTRUCTIONS:
 * 1. Create a new Google Sheet
 * 2. Rename the first sheet to "Queue"
 * 3. Add headers in row 1: id | url | status | created_at | claimed_at | completed_at | meta
 * 4. Go to Extensions > Apps Script
 * 5. Paste this entire script
 * 6. Click Deploy > New deployment
 * 7. Select type: Web app
 * 8. Execute as: Me
 * 9. Who has access: Anyone
 * 10. Copy the web app URL - this is your SHEETS_WEBHOOK_URL
 */

// Configuration
// IMPORTANT: Replace with your Google Sheet ID (from the URL: docs.google.com/spreadsheets/d/SHEET_ID/edit)
const SHEET_ID = 'YOUR_SHEET_ID_HERE';
const SHEET_NAME = 'Queue';
const RESOURCES_SHEET_NAME = 'Learning'; // Tab for archived tweets (same as learning resources)
const VALID_URL_PATTERNS = [
  /^https?:\/\/(www\.)?(twitter\.com|x\.com)\/\w+\/status\/\d+/i,
  /^https?:\/\/(mobile\.)?(twitter\.com|x\.com)\/\w+\/status\/\d+/i
];

/**
 * Handle GET requests (for testing)
 */
function doGet(e) {
  return ContentService.createTextOutput(JSON.stringify({
    status: 'ok',
    message: 'Tweet Queue Webhook is running. Use POST to add URLs.'
  })).setMimeType(ContentService.MimeType.JSON);
}

/**
 * Handle POST requests
 */
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const action = data.action || 'add';

    switch (action) {
      case 'add':
        return addToQueue(data.url);
      case 'claim':
        return claimItem(data.secret);
      case 'complete':
        return completeItem(data.id, data.meta, data.secret);
      case 'release':
        return releaseItem(data.id, data.secret);
      case 'fail':
        return failItem(data.id, data.error, data.secret);
      case 'archive':
        return archiveToResources(data);
      default:
        return jsonResponse({ error: 'Unknown action' }, 400);
    }
  } catch (err) {
    return jsonResponse({ error: err.message }, 500);
  }
}

/**
 * Add a URL to the queue
 */
function addToQueue(url) {
  if (!url) {
    return jsonResponse({ error: 'Missing url parameter' }, 400);
  }

  // Clean URL (remove tracking params)
  url = cleanTwitterUrl(url);

  // Validate URL
  const isValid = VALID_URL_PATTERNS.some(pattern => pattern.test(url));
  if (!isValid) {
    return jsonResponse({ error: 'Invalid Twitter/X URL' }, 400);
  }

  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Queue sheet not found' }, 500);
  }

  // Check for duplicates
  const data = sheet.getDataRange().getValues();
  for (let i = 1; i < data.length; i++) {
    if (data[i][1] === url) {
      return jsonResponse({
        status: 'duplicate',
        message: 'URL already in queue',
        id: data[i][0]
      });
    }
  }

  // Generate ID and add row
  const id = Utilities.getUuid().substring(0, 8);
  const now = new Date().toISOString();

  sheet.appendRow([id, url, 'pending', now, '', '', '']);

  return jsonResponse({
    status: 'queued',
    id: id,
    url: url
  });
}

/**
 * Claim the next pending item for processing
 */
function claimItem(secret) {
  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Queue sheet not found' }, 500);
  }

  const data = sheet.getDataRange().getValues();
  const now = new Date().toISOString();

  // Find first pending item
  for (let i = 1; i < data.length; i++) {
    if (data[i][2] === 'pending') {
      const row = i + 1; // 1-indexed
      const id = data[i][0];
      const url = data[i][1];

      // Mark as claimed
      sheet.getRange(row, 3).setValue('claimed');
      sheet.getRange(row, 5).setValue(now);

      return jsonResponse({
        item: {
          id: id,
          url: url,
          row: row
        }
      });
    }
  }

  return jsonResponse({ item: null, message: 'No pending items' });
}

/**
 * Mark an item as completed and remove from queue
 */
function completeItem(id, meta, secret) {
  if (!id) {
    return jsonResponse({ error: 'Missing id parameter' }, 400);
  }

  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Queue sheet not found' }, 500);
  }

  const data = sheet.getDataRange().getValues();

  for (let i = 1; i < data.length; i++) {
    if (data[i][0] === id) {
      const row = i + 1;
      // Delete the row since it's now archived in the Learning sheet
      sheet.deleteRow(row);
      return jsonResponse({ status: 'completed', id: id, deleted: true });
    }
  }

  return jsonResponse({ error: 'Item not found' }, 404);
}

/**
 * Release a claimed item back to pending
 */
function releaseItem(id, secret) {
  if (!id) {
    return jsonResponse({ error: 'Missing id parameter' }, 400);
  }

  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Queue sheet not found' }, 500);
  }

  const data = sheet.getDataRange().getValues();

  for (let i = 1; i < data.length; i++) {
    if (data[i][0] === id) {
      const row = i + 1;
      sheet.getRange(row, 3).setValue('pending');
      sheet.getRange(row, 5).setValue(''); // Clear claimed_at
      return jsonResponse({ status: 'released', id: id });
    }
  }

  return jsonResponse({ error: 'Item not found' }, 404);
}

/**
 * Mark an item as failed (won't be retried)
 */
function failItem(id, errorMsg, secret) {
  if (!id) {
    return jsonResponse({ error: 'Missing id parameter' }, 400);
  }

  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Queue sheet not found' }, 500);
  }

  const data = sheet.getDataRange().getValues();
  const now = new Date().toISOString();

  for (let i = 1; i < data.length; i++) {
    if (data[i][0] === id) {
      const row = i + 1;
      sheet.getRange(row, 3).setValue('failed');
      sheet.getRange(row, 6).setValue(now);
      if (errorMsg) {
        sheet.getRange(row, 7).setValue(JSON.stringify({ error: errorMsg }));
      }
      return jsonResponse({ status: 'failed', id: id });
    }
  }

  return jsonResponse({ error: 'Item not found' }, 404);
}

/**
 * Archive a tweet to the Learning sheet
 * Expected data: { name, author, type, link, date_added, notes, media }
 * Learning sheet columns: id | Read | Date Added | Name | Author | Type | Link | Rating | Notes | Checked Out | Student | Media
 */
function archiveToResources(data) {
  const sheet = SpreadsheetApp.openById(SHEET_ID).getSheetByName(RESOURCES_SHEET_NAME);
  if (!sheet) {
    return jsonResponse({ error: 'Learning sheet not found' }, 500);
  }

  const link = data.link || '';
  if (!link) {
    return jsonResponse({ error: 'Missing link parameter' }, 400);
  }

  // Check for duplicates by link (column G, index 6)
  const existingData = sheet.getDataRange().getValues();
  for (let i = 1; i < existingData.length; i++) {
    if (existingData[i][6] === link) {
      return jsonResponse({ status: 'duplicate', message: 'Tweet already archived' });
    }
  }

  // Generate slug-style id from name
  const name = data.name || 'tweet';
  const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 30);

  // Append row: id | Read | Date Added | Name | Author | Type | Link | Rating | Notes | Checked Out | Student | Media
  const row = [
    id,                      // A: id
    false,                   // B: Read checkbox
    data.date_added || '',   // C: Date Added
    name,                    // D: Name
    data.author || '',       // E: Author
    data.type || 'tweet',    // F: Type
    link,                    // G: Link
    '',                      // H: Rating (empty for tweets)
    data.notes || '',        // I: Notes
    false,                   // J: Checked Out (empty for tweets)
    '',                      // K: Student (empty for tweets)
    data.media || ''         // L: Media
  ];

  sheet.appendRow(row);

  return jsonResponse({ status: 'archived', link: link });
}

/**
 * Clean Twitter URL - remove tracking parameters
 */
function cleanTwitterUrl(url) {
  try {
    const parsed = new URL(url);
    // Keep only the base URL with status ID
    const match = url.match(/(https?:\/\/(?:www\.|mobile\.)?(twitter\.com|x\.com)\/\w+\/status\/\d+)/i);
    return match ? match[1] : url;
  } catch {
    return url;
  }
}

/**
 * Helper to create JSON response
 */
function jsonResponse(data, status = 200) {
  const output = ContentService.createTextOutput(JSON.stringify(data));
  output.setMimeType(ContentService.MimeType.JSON);
  return output;
}
