import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ScanLine, Gauge, FileBarChart, ArrowUpRight, AlertTriangle, Sparkles } from 'lucide-react';

import VideoDropzone from '../components/upload/VideoDropzone.jsx';
import { analyzePatientData } from '../api/analyzeApi.js';
import Spinner from '../components/common/Spinner.jsx';
import Reveal from '../components/common/Reveal.jsx';
import { useTiltSpotlight } from '../hooks/useInteractions.js';

const FEATURES = [
  {
    icon: ScanLine,
    title: 'Objective by design',
    body: 'Quantifiable 3D kinematic tracking replaces subjective observation with measurable signal.',
  },
  {
    icon: Gauge,
    title: 'Seconds, not sessions',
    body: 'An optimized pose-to-inference pipeline returns a full clinical read in moments.',
  },
  {
    icon: FileBarChart,
    title: 'Explainable output',
    body: 'Per-region SHAP attribution and a temporal risk curve — every score is traceable.',
  },
];

const MARQUEE = [
  '3D pose extraction', 'Spatiotemporal windows', 'SHAP attribution', 'Bilateral symmetry',
  'XGBoost inference', 'Gait kinematics', 'Out-of-distribution check', 'Per-window risk',
];

const FeatureCard = ({ icon: Icon, title, body, delay }) => {
  const { ref, onMouseMove, onMouseLeave } = useTiltSpotlight(6);
  return (
    <Reveal delay={delay}>
      <div
        ref={ref}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        className="group glow-card spotlight tilt surface rounded-3xl p-7 h-full"
      >
        <div className="flex items-center justify-between mb-6">
          <div className="relative grid place-items-center w-12 h-12 rounded-2xl bg-ink-900 text-white overflow-hidden">
            <span className="absolute inset-0 bg-gradient-to-br from-brand-500 to-accent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <Icon className="relative w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
          </div>
          <ArrowUpRight className="w-5 h-5 text-ink-300 group-hover:text-brand-600 group-hover:-translate-y-1 group-hover:translate-x-1 transition-all duration-300" />
        </div>
        <h3 className="font-display text-xl font-medium text-ink-900 mb-2">{title}</h3>
        <p className="text-sm text-ink-500 leading-relaxed">{body}</p>
      </div>
    </Reveal>
  );
};

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [ageGroup, setAgeGroup] = useState('child');
  const navigate = useNavigate();

  const handleFileUpload = async (selectedFile) => {
    if (!selectedFile) {
      setFile(null);
      return;
    }
    setFile(selectedFile);
    setIsProcessing(true);
    setError(null);

    try {
      const results = await analyzePatientData(selectedFile, ageGroup);
      navigate('/report', { state: { patientData: results } });
    } catch (err) {
      setError(err.message || 'An error occurred during analysis.');
      setIsProcessing(false);
    }
  };

  if (isProcessing) {
    const isCSV = file?.name?.toLowerCase().endsWith('.csv');
    const spinnerText = isCSV
      ? 'Running XGBoost clinical inference…'
      : 'Extracting 3D kinematics & scoring windows…';

    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 animate-fade-in">
        <Spinner text={spinnerText} />
        <div className="mt-10 surface rounded-2xl px-6 py-4 max-w-md text-center">
          <p className="text-sm text-ink-700">
            Processing <span className="font-semibold text-ink-900">{file?.name}</span>
          </p>
          <p className="text-xs text-ink-400 mt-1.5">
            {isCSV
              ? 'Applying the model to sensor telemetry.'
              : 'This scales with clip length — usually under a minute.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6">
      {/* HERO */}
      <section className="relative pt-16 sm:pt-24 pb-10 text-center">
        {/* floating decorative orbs */}
        <span className="pointer-events-none absolute left-[8%] top-24 w-3 h-3 rounded-full bg-brand-500/60 blur-[1px] animate-float" />
        <span className="pointer-events-none absolute right-[12%] top-40 w-2 h-2 rounded-full bg-accent/70 animate-float" style={{ animationDelay: '1.2s' }} />
        <span className="pointer-events-none absolute left-[20%] bottom-6 w-2.5 h-2.5 rounded-full bg-brand-400/50 animate-float" style={{ animationDelay: '2.4s' }} />

        <div className="inline-flex items-center gap-2 chip mb-6 animate-fade-up shine">
          <Sparkles className="w-3.5 h-3.5 text-brand-600" />
          Biomarker-driven spatiotemporal analysis
        </div>

        <h1
          className="font-display text-5xl sm:text-6xl lg:text-7xl font-medium leading-[0.98] tracking-tight text-ink-900 text-balance animate-fade-up"
          style={{ animationDelay: '60ms' }}
        >
          Read motor development
          <br className="hidden sm:block" />
          <span className="gradient-text italic"> from movement itself.</span>
        </h1>

        <p
          className="mt-6 text-lg text-ink-500 max-w-2xl mx-auto text-balance animate-fade-up"
          style={{ animationDelay: '120ms' }}
        >
          Upload a clip of natural movement or hardware telemetry. Kinetiq returns an
          objective, region-by-region screening read — no markers, no manual scoring.
        </p>
      </section>

      {/* UPLOAD */}
      <section className="animate-fade-up" style={{ animationDelay: '180ms' }}>
        {error && (
          <div className="mb-6 max-w-2xl mx-auto flex items-center gap-3 rounded-2xl border border-red-200 bg-red-50/80 px-4 py-3 shadow-soft">
            <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
            <p className="text-sm font-medium text-red-700">{error}</p>
          </div>
        )}
        {/* Age-group selector — child uses the full fusion; adult is a provisional HC-only screen */}
        <div className="mb-5 flex flex-col items-center gap-2">
          <div className="inline-flex rounded-full border border-white/60 bg-white/50 p-1 shadow-soft backdrop-blur">
            {[
              { key: 'child', label: 'Child', hint: 'ages ~3–12' },
              { key: 'adult', label: 'Adult', hint: 'provisional' },
            ].map(({ key, label, hint }) => (
              <button
                key={key}
                type="button"
                onClick={() => setAgeGroup(key)}
                aria-pressed={ageGroup === key}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                  ageGroup === key
                    ? 'bg-ink-900 text-white shadow-soft'
                    : 'text-ink-500 hover:text-ink-800'
                }`}
              >
                {label}
                <span className={`ml-2 text-[11px] ${ageGroup === key ? 'text-white/60' : 'text-ink-300'}`}>{hint}</span>
              </button>
            ))}
          </div>
          {ageGroup === 'adult' && (
            <p className="text-[11px] text-amber-600 max-w-md text-center">
              Adult mode is provisional: the movement model is child-trained, so this screens on
              handcrafted biomechanics only. A “typical” adult result is not clinically validated.
            </p>
          )}
        </div>

        <VideoDropzone onFileSelect={handleFileUpload} />
        <p className="mt-4 text-center text-xs text-ink-400">
          Recordings are processed for analysis only. Nothing here constitutes a diagnosis.
        </p>
      </section>

      {/* MARQUEE STRIP */}
      <div className="mt-16 marquee py-3 border-y border-white/50">
        <div className="marquee-track">
          {[...MARQUEE, ...MARQUEE].map((m, i) => (
            <span key={i} className="inline-flex items-center gap-3 px-6 text-sm font-medium text-ink-400">
              <span className="h-1.5 w-1.5 rounded-full bg-brand-500/70" />
              {m}
            </span>
          ))}
        </div>
      </div>

      {/* FEATURES */}
      <section className="mt-20 pb-24 perspective">
        <Reveal className="flex items-end justify-between mb-8">
          <div>
            <p className="eyebrow mb-2">Why it holds up</p>
            <h2 className="font-display text-3xl font-medium tracking-tight text-ink-900">
              A clinical read you can defend
            </h2>
          </div>
        </Reveal>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {FEATURES.map(({ icon, title, body }, i) => (
            <FeatureCard key={title} icon={icon} title={title} body={body} delay={i * 110} />
          ))}
        </div>
      </section>
    </div>
  );
};

export default HomePage;
