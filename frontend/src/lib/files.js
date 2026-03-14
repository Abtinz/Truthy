/**
 * Convert one browser file to a base64 string.
 *
 * @param {File} file Browser file object.
 * @returns {Promise<string>} Base64-encoded file contents without the data URL prefix.
 */
export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const result = String(reader.result || "");
      const [, base64Data = ""] = result.split(",");
      resolve(base64Data);
    };

    reader.onerror = () => {
      reject(new Error(`Failed to read file: ${file.name}`));
    };

    reader.readAsDataURL(file);
  });
}

/**
 * Build the API review payload from uploaded browser files.
 *
 * @param {string} applicationName Human-entered application name.
 * @param {{id: string, file: File}[]} selectedFiles Uploaded file entries.
 * @returns {Promise<object>} Serialized review payload for the FastAPI service.
 */
export async function buildReviewPayload(applicationName, selectedFiles) {
  const files = await Promise.all(
    selectedFiles.map(async (entry) => ({
      file_name: entry.file.name,
      content_type: entry.file.type || "application/octet-stream",
      base64_data: await fileToBase64(entry.file),
    })),
  );

  return {
    application_name: applicationName,
    files,
  };
}

/**
 * Create object URLs so uploaded files can be previewed in the UI.
 *
 * @param {{id: string, file: File}[]} selectedFiles Uploaded file entries.
 * @returns {Record<string, string>} Map from file identifier to object URL.
 */
export function createObjectUrlMap(selectedFiles) {
  return selectedFiles.reduce((map, entry) => {
    map[entry.id] = URL.createObjectURL(entry.file);
    return map;
  }, {});
}

/**
 * Revoke all preview URLs created for the file-preview panel.
 *
 * @param {Record<string, string>} objectUrlMap Map from file identifier to object URL.
 * @returns {void}
 */
export function revokeObjectUrlMap(objectUrlMap) {
  Object.values(objectUrlMap).forEach((objectUrl) => URL.revokeObjectURL(objectUrl));
}

/**
 * Compute a single overall case result from the review stage outputs.
 *
 * @param {object} reviewResult API review response payload.
 * @returns {string} `PASS`, `FAIL`, or `MANUAL REVIEW`.
 */
export function computeOverallResult(reviewResult) {
  const statuses = (reviewResult.stage_outcomes || []).map((stage) => stage.status);
  if (statuses.length === 0) {
    return "MANUAL REVIEW";
  }
  if (statuses.every((status) => status === "passed")) {
    return "PASS";
  }
  if (statuses.includes("failed")) {
    return "FAIL";
  }
  return "MANUAL REVIEW";
}

/**
 * Format the review response into a compact officer-facing report block.
 *
 * @param {object} reviewResult API review response payload.
 * @returns {string} Multiline summary string.
 */
export function formatResultSummary(reviewResult) {
  const lines = [
    `Application Name: ${reviewResult.application_name || ""}`,
    `Overall Result: ${computeOverallResult(reviewResult)}`,
    "",
    "Stage Outcomes:",
  ];

  (reviewResult.stage_outcomes || []).forEach((stage) => {
    lines.push(`- ${stage.stage_name}: ${stage.status}`);
    lines.push(`  Explanation: ${stage.explanation}`);
  });

  lines.push("");
  lines.push("Final Report:");
  lines.push(reviewResult.final_report_text || "");

  return lines.join("\n");
}
