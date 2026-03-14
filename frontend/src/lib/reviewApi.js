const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);

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
