const fs = require('fs');
const path = require('path');

/**
 * Load course changelog from JSON file
 * Returns object with course keys (e.g., cse10124, cse30124)
 */
module.exports = function() {
  const jsonPath = path.join(process.cwd(), 'static', 'json', 'course_changelog.json');

  try {
    if (fs.existsSync(jsonPath)) {
      const content = fs.readFileSync(jsonPath, 'utf8');
      const data = JSON.parse(content);
      console.log('[11ty] Loaded course changelog');
      return data;
    }
  } catch (err) {
    console.error('Warning: Could not load course_changelog.json:', err.message);
  }

  return {};
};
