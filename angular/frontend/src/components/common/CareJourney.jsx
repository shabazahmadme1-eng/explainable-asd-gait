import React from 'react';
import { ScanLine, Stethoscope, BadgeCheck, HeartHandshake } from 'lucide-react';
import Reveal from './Reveal.jsx';

const STEPS = [
  {
    icon: ScanLine,
    title: 'Screen',
    tag: 'You are here',
    body: 'An objective motor signal from a short video — at home, in minutes.',
    active: true,
  },
  {
    icon: Stethoscope,
    title: 'Evaluate',
    body: 'A clinician reviews history, observation and validated instruments.',
  },
  {
    icon: BadgeCheck,
    title: 'Diagnose',
    body: 'Formal assessment (e.g. ADOS-2) confirms and characterizes support needs.',
  },
  {
    icon: HeartHandshake,
    title: 'Intervene',
    body: 'Early, targeted support — where developmental outcomes improve most.',
  },
];

const CareJourney = () => (
  <div className="relative">
    {/* connecting line (desktop) */}
    <div className="hidden md:block absolute top-7 left-[12%] right-[12%] h-px bg-gradient-to-r from-brand-500/60 via-brand-400/40 to-accent/50" />

    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 md:gap-3">
      {STEPS.map((s, i) => (
        <Reveal key={s.title} delay={i * 110} className="relative text-center md:px-3">
          <div className="relative inline-grid place-items-center mb-4">
            {s.active && (
              <span className="absolute h-14 w-14 rounded-2xl bg-brand-500/25 blur-md animate-pulse" />
            )}
            <div
              className={`relative grid place-items-center w-14 h-14 rounded-2xl ${
                s.active
                  ? 'bg-ink-900 text-white shadow-glow ring-2 ring-brand-500/50'
                  : 'surface text-ink-500'
              }`}
            >
              <s.icon className="w-6 h-6" />
              <span className="absolute -bottom-2 -right-2 grid place-items-center w-6 h-6 rounded-full bg-white text-[11px] font-bold text-ink-500 ring-1 ring-ink-900/10 shadow-soft">
                {i + 1}
              </span>
            </div>
          </div>
          <div className="flex items-center justify-center gap-2 mb-1">
            <h3 className="font-display text-lg font-medium text-ink-900">{s.title}</h3>
            {s.tag && (
              <span className="chip !py-0.5 !px-2 text-[9px] text-brand-700 border-brand-300 bg-brand-50">
                {s.tag}
              </span>
            )}
          </div>
          <p className="text-xs text-ink-500 leading-relaxed max-w-[16rem] mx-auto">{s.body}</p>
        </Reveal>
      ))}
    </div>

    <p className="mt-8 text-center text-sm text-ink-500 max-w-2xl mx-auto">
      Kinetiq lives at step one. Our job is to{' '}
      <span className="font-semibold text-ink-800">shorten the time to evaluation</span> — not to
      replace the clinician who diagnoses and guides care.
    </p>
  </div>
);

export default CareJourney;
