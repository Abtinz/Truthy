const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);
const INDEXER_BASE_URL = (
  import.meta.env.VITE_INDEXER_BASE_URL || "http://localhost:8001"
).replace(/\/$/, "");

/**
 * Submit the review payload to the FastAPI gateway.
 *
 * @param {object} payload Serialized review request body.
 * @returns {Promise<object>} Parsed API review response.
 */
export async function submitReview(payload) {
  const response = await fetch(`${API_BASE_URL}/review`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Review request failed.");
  }

  return response.json();
}

/**
 * Submit a direct indexing request to the indexer service.
 *
 * @param {object} payload Serialized direct-indexing request body.
 * @returns {Promise<object>} Parsed API indexing response.
 */
export async function submitIndexRequest(payload) {
  const response = await fetch(`${INDEXER_BASE_URL}/index`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Index request failed.");
  }

  return response.json();
}
