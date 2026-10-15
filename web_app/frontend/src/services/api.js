// All network calls live here so components stay purely visual.
// Endpoints (implemented in web_app/backend/app.py):
//   GET  /api/schema   -> dropdown options + training ranges, read from the trained model
//   POST /api/predict  -> fraud score for one claim

export class ApiError extends Error {
  constructor(message, { fieldErrors = {}, network = false } = {}) {
    super(message);
    this.fieldErrors = fieldErrors;
    this.network = network;
  }
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new ApiError('The analysis service is not reachable.', { network: true });
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON response */
  }
  if (!res.ok) {
    if (res.status >= 500 || !body) {
      throw new ApiError('The analysis service hit an error. Try again, or check the backend terminal.', { network: !body && res.status === 404 });
    }
    throw new ApiError(body.error || 'The claim could not be analyzed.', { fieldErrors: body.field_errors || {} });
  }
  return body;
}

export const getSchema = () => request('/api/schema');

export const predictClaim = (claim) =>
  request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(claim),
  });
