import React, { useState, useCallback } from 'react';
import { UploadCloud, FileVideo, X, FileSpreadsheet } from 'lucide-react';

const VideoDropzone = ({ onFileSelect, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const validateAndSetFile = (file) => {
    setError('');
    if (!file) return;

    const validTypes = [
      'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi',
      'text/csv', 'application/vnd.ms-excel',
    ];
    const isValidType = validTypes.includes(file.type);

    const fileName = file.name.toLowerCase();
    const isValidExtension =
      fileName.endsWith('.mp4') ||
      fileName.endsWith('.mov') ||
      fileName.endsWith('.avi') ||
      fileName.endsWith('.csv');

    if (!isValidType && !isValidExtension) {
      setError('Please upload a valid MP4, MOV, AVI, or CSV file.');
      return;
    }

    setSelectedFile(file);
    onFileSelect(file);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    onFileSelect(null);
    setError('');
  };

  const isCSV = selectedFile?.name?.toLowerCase().endsWith('.csv');

  return (
    <div className="w-full max-w-2xl mx-auto glow-card rounded-[1.75rem]">
      <div
        className={`group relative flex flex-col items-center justify-center w-full h-72 rounded-3xl
          transition-all duration-300 overflow-hidden
          ${dragActive
            ? 'shadow-glow ring-2 ring-brand-500/60 bg-brand-50/70 scale-[1.01]'
            : 'surface hover:-translate-y-0.5 hover:shadow-lift'}
          ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {/* scanning beam */}
        <div
          className="pointer-events-none absolute inset-x-0 h-24 opacity-70"
          style={{
            background: 'linear-gradient(180deg, transparent, rgba(115,99,232,0.16), rgba(24,198,166,0.10), transparent)',
            animation: 'scan-y 4.5s ease-in-out infinite',
          }}
        />
        {/* dashed inner guide */}
        <div className="pointer-events-none absolute inset-4 rounded-2xl border border-dashed border-ink-900/10" />

        {!selectedFile ? (
          <>
            <input
              type="file"
              className="absolute inset-0 z-10 w-full h-full opacity-0 cursor-pointer"
              onChange={handleChange}
              accept=".mp4,.mov,.avi,.csv"
              disabled={isLoading}
            />
            <div className="relative mb-5 pointer-events-none">
              {/* orbiting accent ring */}
              <div className="absolute inset-0 grid place-items-center">
                <div className="relative w-16 h-16" style={{ animation: 'spin-slow 12s linear infinite' }}>
                  <span className="absolute -top-1 left-1/2 -ml-1 w-2 h-2 rounded-full bg-brand-500" />
                  <span className="absolute top-1/2 -right-1 -mt-1 w-1.5 h-1.5 rounded-full bg-accent" />
                </div>
              </div>
              <div className={`relative grid place-items-center w-16 h-16 rounded-2xl transition-all duration-300
                ${dragActive ? 'bg-brand-600 text-white scale-110' : 'bg-ink-900 text-white group-hover:scale-105'}`}>
                <UploadCloud className={`w-7 h-7 ${dragActive ? '' : 'group-hover:animate-float'}`} />
                <span className="absolute -inset-1 rounded-2xl bg-brand-500/40 blur-lg opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </div>
            <p className="text-base font-semibold text-ink-800">
              Drop a recording, or <span className="text-brand-600 underline decoration-brand-300 underline-offset-4">browse files</span>
            </p>
            <p className="mt-2 text-xs text-ink-400 font-medium tracking-wide">
              MP4 · MOV · AVI · CSV — up to a few minutes of gait
            </p>
          </>
        ) : (
          <div className="relative flex flex-col items-center p-6 text-center animate-scale-in">
            <div className="relative">
              <div className={`grid place-items-center w-20 h-20 rounded-2xl mb-4 ${isCSV ? 'bg-emerald-50 text-emerald-600' : 'bg-brand-50 text-brand-600'}`}>
                {isCSV ? <FileSpreadsheet className="w-9 h-9" /> : <FileVideo className="w-9 h-9" />}
              </div>
              {!isLoading && (
                <button
                  onClick={clearFile}
                  className="absolute -top-2 -right-2 p-1.5 bg-white rounded-full border border-ink-900/10 shadow-soft hover:bg-red-50 hover:text-red-500 hover:border-red-200 transition-colors"
                  aria-label="Remove file"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <p className="text-sm font-semibold text-ink-800 truncate max-w-[240px]">
              {selectedFile.name}
            </p>
            <p className="text-xs text-ink-400 mt-1 font-mono">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · ready
            </p>
          </div>
        )}
      </div>
      {error && (
        <p className="mt-3 text-sm text-red-500 text-center font-medium">{error}</p>
      )}
    </div>
  );
};

export default VideoDropzone;
