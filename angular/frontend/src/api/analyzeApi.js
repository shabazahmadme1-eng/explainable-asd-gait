// Base URL comes from .env (VITE_API_URL); local dev falls back to a local backend.
// A production build without VITE_API_URL (the GitHub Pages demo while no backend is
// hosted) has no analysis server, and the UI says so instead of failing silently.
const API_URL = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : ''))
  .replace(/\/$/, '');
export const BACKEND_CONNECTED = Boolean(API_URL);
export const NO_BACKEND_MESSAGE =
  'The analysis server for this online demo is not connected yet, so files cannot be analysed here. ' +
  'To try it, run the backend locally (see angular/README.md in the GitHub repository).';

export const analyzePatientData = async (file, ageGroup = 'child') => {
  if (!BACKEND_CONNECTED) throw new Error(NO_BACKEND_MESSAGE);
  const formData = new FormData();
  formData.append('file', file);

  // Tell the FastAPI backend whether this is a CSV or a video.
  const isCSV = /\.(csv|xlsx)$/i.test(file.name);  // coordinate CSV or Excel
  formData.append('file_type', isCSV ? 'csv' : 'video');

  // Child (default) uses the full HC + movement-model fusion; Adult disables the
  // child-trained movement model and screens on handcrafted biomechanics (provisional).
  formData.append('age_group', ageGroup === 'adult' ? 'adult' : 'child');

  try {
    const response = await fetch(`${API_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
    });

    // The backend uses 400 (bad format) and 422 (quality gate) with a `detail`
    // message intended for the clinician - surface it verbatim.
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error (${response.status}). Is the backend running?`);
    }

    const data = await response.json();
    if (data.status === 'success') return data;
    throw new Error(data.message || 'Analysis failed.');
  } catch (error) {
    console.error('API Fetch Error:', error);
    throw new Error(error.message || 'Cannot connect to the analysis backend.');
  }
};
