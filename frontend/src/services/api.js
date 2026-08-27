/**
 * Small fetch wrapper for the MedOrchestrate FastAPI backend (Phase 9).
 *
 * Every function throws a plain Error with a clinician-readable message
 * on failure -- network failure, 422 validation errors, and 500s are
 * all normalized to `error.message`, with `error.status` set when the
 * server actually responded (absent for a pure network failure).
 */

const API_BASE_URL = "http://localhost:8000";

const NETWORK_ERROR_MESSAGE =
  "Could not reach the MedOrchestrate server. Check your connection and try again.";

function formatDetail(detail) {
  if (typeof detail === "string") return detail;
  // FastAPI's default 422 shape is a list of {loc, msg, type} objects.
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.at(-1) : undefined;
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function handleResponse(response) {
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        message = formatDetail(body.detail);
      }
    } catch {
      // Response body wasn't JSON (or was empty) -- fall back to the
      // generic status-based message above.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function safeFetch(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(NETWORK_ERROR_MESSAGE);
  }
  return handleResponse(response);
}

/** POST /diagnose -- formData must already be a FormData instance. */
export function diagnoseCase(formData) {
  return safeFetch(`${API_BASE_URL}/diagnose`, {
    method: "POST",
    body: formData,
  });
}

/** GET /diagnose/{case_id} */
export function getDiagnosis(caseId) {
  return safeFetch(`${API_BASE_URL}/diagnose/${encodeURIComponent(caseId)}`);
}

/** POST /review/{case_id} -- review = { decision, annotation, reviewer } */
export function submitReview(caseId, review) {
  return safeFetch(`${API_BASE_URL}/review/${encodeURIComponent(caseId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(review),
  });
}
