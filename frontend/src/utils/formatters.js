/**
 * Utility functions for formatting clinical and system data 
 * for the React frontend interface.
 */

// 1. Converts AI probability decimals into clean percentages (0.95623 -> "95.6%")
export const formatConfidence = (decimal) => {
  if (decimal === null || decimal === undefined) return '0.0%';
  return `${(decimal * 100).toFixed(1)}%`;
};

// 2. Cleans up raw XGBoost feature names (e.g., "mean-x-HandTipRightA" -> "Hand Tip Right A (X-Axis)")
export const formatBiomarkerName = (rawName) => {
  if (!rawName) return 'Unknown Marker';
  
  // Split the string by dashes
  const parts = rawName.split('-');
  if (parts.length < 3) return rawName; 

  const metric = parts[0]; // e.g., "mean" or "std"
  const axis = parts[1].toUpperCase(); // e.g., "X", "Y", "Z"
  const joint = parts[2]; // e.g., "HandTipRightA"

  // Adds spaces before capital letters (HandTipRightA -> Hand Tip Right A)
  const cleanJoint = joint.replace(/([A-Z])/g, ' $1').trim();
  
  return `${cleanJoint} (${axis}-Axis ${metric})`;
};

// 3. Converts raw bytes into a readable MB format for the Upload Dropzone
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// 4. Generates a clinical timestamp for the report (e.g., "Oct 24, 2026 - 14:30")
export const generateClinicalTimestamp = () => {
  const now = new Date();
  return now.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};