import React, { useEffect, useRef } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';
import BrandMark from '../common/BrandMark.jsx';
import AuroraBackground from './AuroraBackground.jsx';

const Layout = ({ children }) => {
  const { pathname } = useLocation();
  const onReport = pathname.startsWith('/report');
  const glowRef = useRef(null);

  // Cursor-following ambient glow.
  useEffect(() => {
    const move = (e) => {
      const el = glowRef.current;
      if (el) el.style.setProperty('transform', `translate(${e.clientX}px, ${e.clientY}px)`);
    };
    window.addEventListener('pointermove', move);
    return () => window.removeEventListener('pointermove', move);
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col">
      <AuroraBackground />
      <div ref={glowRef} className="cursor-glow" style={{ transform: 'translate(-100px,-100px)' }} />
      <div className="grain-overlay" />

      {/* Header */}
      <header className="app-header sticky top-0 z-30">
        <div className="absolute inset-0 bg-paper/60 backdrop-blur-xl border-b border-white/40 shadow-[0_1px_0_0_rgba(255,255,255,0.5)_inset]" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <Link to="/" className="group flex items-center gap-2.5">
            <span className="relative transition-transform duration-500 group-hover:rotate-[-8deg] group-hover:scale-110">
              <span className="absolute -inset-2 rounded-2xl bg-brand-500/30 blur-lg opacity-0 group-hover:opacity-100 transition-opacity" />
              <BrandMark className="relative w-8 h-8" />
            </span>
            <span className="flex flex-col leading-none">
              <span className="font-display text-lg font-semibold tracking-tight text-ink-900">
                Kineti<span className="gradient-text">q</span>
              </span>
              <span className="text-[10px] font-medium uppercase tracking-[0.22em] text-ink-400">
                Motor Screening
              </span>
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-1 absolute left-1/2 -translate-x-1/2">
            {[
              { to: '/', label: 'Screen', end: true },
              { to: '/learn', label: 'Learn' },
            ].map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) =>
                  `relative px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive ? 'text-ink-900' : 'text-ink-500 hover:text-ink-900'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {l.label}
                    <span
                      className={`absolute left-3 right-3 -bottom-0.5 h-0.5 rounded-full bg-gradient-to-r from-brand-600 to-accent transition-transform duration-300 origin-left ${
                        isActive ? 'scale-x-100' : 'scale-x-0'
                      }`}
                    />
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex chip">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
              </span>
              Model online
            </span>
            {onReport ? (
              <span className="chip">
                <Activity className="w-3.5 h-3.5 text-brand-600" /> Report
              </span>
            ) : (
              <span className="chip">v1.0 · clinical preview</span>
            )}
          </div>
        </div>
      </header>

      {/* Page content */}
      <main className="relative z-10 flex-1">{children}</main>

      {/* Footer */}
      <footer className="app-footer relative z-10 mt-auto border-t border-white/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-ink-400">
          <div className="flex items-center gap-2">
            <BrandMark className="w-5 h-5" />
            <span>Kinetiq — research preview. Not a diagnostic device.</span>
          </div>
          <div className="flex items-center gap-5">
            <Link to="/learn" className="hover:text-ink-700 transition-colors">Understanding ASD</Link>
            <span className="text-ink-300">·</span>
            <span>XGBoost inference</span>
            <span className="text-ink-300">·</span>
            <span>© {new Date().getFullYear()}</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
