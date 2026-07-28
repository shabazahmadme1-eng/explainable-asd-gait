import React from 'react';

// Living backdrop: drifting aurora blobs, a slow panning grid, and a soft
// vignette. Pure CSS animation, fixed behind all content (z-0).
const AuroraBackground = () => (
  <div className="aurora-bg fixed inset-0 -z-0 overflow-hidden pointer-events-none" aria-hidden="true">
    {/* base wash */}
    <div className="absolute inset-0 bg-gradient-to-b from-[#f4f5fb] via-[#eef0f7] to-[#e9eaf4]" />

    {/* panning grid */}
    <div
      className="absolute inset-0 opacity-[0.5]"
      style={{
        backgroundImage:
          'linear-gradient(rgba(95,73,217,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(95,73,217,0.06) 1px, transparent 1px)',
        backgroundSize: '44px 44px',
        animation: 'grid-pan 8s linear infinite',
        maskImage: 'radial-gradient(120% 80% at 50% 0%, #000 30%, transparent 75%)',
        WebkitMaskImage: 'radial-gradient(120% 80% at 50% 0%, #000 30%, transparent 75%)',
      }}
    />

    {/* aurora blobs */}
    <div
      className="absolute -top-32 -left-24 w-[42rem] h-[42rem] rounded-full blur-3xl"
      style={{ background: 'radial-gradient(circle, rgba(115,99,232,0.42), transparent 62%)', animation: 'aurora-1 22s ease-in-out infinite' }}
    />
    <div
      className="absolute top-1/3 -right-32 w-[38rem] h-[38rem] rounded-full blur-3xl"
      style={{ background: 'radial-gradient(circle, rgba(24,198,166,0.32), transparent 62%)', animation: 'aurora-2 26s ease-in-out infinite' }}
    />
    <div
      className="absolute bottom-0 left-1/4 w-[34rem] h-[34rem] rounded-full blur-3xl"
      style={{ background: 'radial-gradient(circle, rgba(159,132,240,0.30), transparent 62%)', animation: 'aurora-3 30s ease-in-out infinite' }}
    />

    {/* vignette */}
    <div className="absolute inset-0" style={{ background: 'radial-gradient(120% 100% at 50% -10%, transparent 55%, rgba(13,16,23,0.10) 100%)' }} />
  </div>
);

export default AuroraBackground;
