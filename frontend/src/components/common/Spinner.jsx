import React from 'react';
import BrandMark from './BrandMark.jsx';

export default function Spinner({ text = 'Analyzing skeletal data…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-7">
      <div className="relative grid place-items-center h-28 w-28">
        {/* expanding rings */}
        <span className="absolute h-20 w-20 rounded-full border border-brand-400/40" style={{ animation: 'pulse-ring 1.8s cubic-bezier(0.22,1,0.36,1) infinite' }} />
        <span className="absolute h-20 w-20 rounded-full border border-accent/40" style={{ animation: 'pulse-ring 1.8s cubic-bezier(0.22,1,0.36,1) infinite', animationDelay: '0.6s' }} />

        {/* rotating conic ring */}
        <span
          className="absolute h-24 w-24 rounded-full"
          style={{
            background: 'conic-gradient(from 0deg, transparent 0deg, #18c6a6 120deg, #7363e8 300deg, transparent 360deg)',
            WebkitMask: 'radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 3px))',
            mask: 'radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 3px))',
            animation: 'spin-slow 1.4s linear infinite',
          }}
        />

        {/* orbiting dots */}
        <span className="absolute h-24 w-24" style={{ animation: 'spin-rev 3s linear infinite' }}>
          <span className="absolute -top-0.5 left-1/2 -ml-1 w-2 h-2 rounded-full bg-brand-500 shadow-[0_0_10px_2px_rgba(115,99,232,0.6)]" />
        </span>

        {/* core */}
        <div className="relative grid place-items-center h-14 w-14 rounded-2xl bg-ink-900 shadow-glow">
          <span className="absolute inset-0 rounded-2xl bg-gradient-to-br from-brand-600/40 to-accent/30" />
          <BrandMark className="relative w-7 h-7" />
        </div>
      </div>
      <p className="text-sm font-medium text-ink-500 tracking-tight">
        <span className="gradient-text font-semibold">{text}</span>
      </p>
    </div>
  );
}
