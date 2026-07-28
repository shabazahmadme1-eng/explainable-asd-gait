import React from 'react';
import { Link } from 'react-router-dom';
import {
  Brain, Activity, Footprints, Scale, Repeat, Baby, Clock, ArrowRight,
  ClipboardCheck, Stethoscope, HeartHandshake, Info, Sparkles,
} from 'lucide-react';
import Reveal from '../components/common/Reveal.jsx';
import CareJourney from '../components/common/CareJourney.jsx';
import { useTiltSpotlight } from '../hooks/useInteractions.js';

const SEVERITY = [
  {
    level: 'Level 1',
    label: 'Requiring support',
    body: 'Without support, social-communication difficulties cause noticeable challenges. Difficulty initiating interactions; inflexibility interferes with functioning in one or more contexts.',
    tone: 'from-emerald-500 to-teal-500',
  },
  {
    level: 'Level 2',
    label: 'Requiring substantial support',
    body: 'Marked deficits in verbal and non-verbal social skills, apparent even with support. Restricted, repetitive behaviors are obvious to a casual observer and interfere across contexts.',
    tone: 'from-amber-500 to-orange-500',
  },
  {
    level: 'Level 3',
    label: 'Requiring very substantial support',
    body: 'Severe deficits in social communication and very limited initiation. Inflexibility and repetitive behaviors markedly interfere with functioning; great distress changing focus or action.',
    tone: 'from-rose-500 to-red-500',
  },
];

const MOTOR_SIGNS = [
  { icon: Footprints, title: 'Gait & posture', body: 'Atypical walking patterns, toe-walking, and reduced postural stability.' },
  { icon: Scale, title: 'Bilateral symmetry', body: 'Breakdowns in left–right coordination and reduced or asymmetric arm swing.' },
  { icon: Repeat, title: 'Repetitive movement', body: 'Motor stereotypies — repeated, patterned movements of the arms or body.' },
  { icon: Clock, title: 'Timing & milestones', body: 'Delays in sitting, crawling and walking relative to typical development.' },
];

const SectionHead = ({ eyebrow, title, children }) => (
  <Reveal className="max-w-3xl mb-8">
    <p className="eyebrow mb-2">{eyebrow}</p>
    <h2 className="font-display text-3xl sm:text-4xl font-medium tracking-tight text-ink-900 mb-3">{title}</h2>
    {children && <p className="text-ink-500 leading-relaxed">{children}</p>}
  </Reveal>
);

const MotorCard = ({ icon: Icon, title, body, delay }) => {
  const { ref, onMouseMove, onMouseLeave } = useTiltSpotlight(6);
  return (
    <Reveal delay={delay}>
      <div ref={ref} onMouseMove={onMouseMove} onMouseLeave={onMouseLeave}
        className="group glow-card spotlight tilt surface rounded-3xl p-6 h-full">
        <div className="relative grid place-items-center w-11 h-11 rounded-2xl bg-ink-900 text-white mb-4 overflow-hidden">
          <span className="absolute inset-0 bg-gradient-to-br from-brand-500 to-accent opacity-0 group-hover:opacity-100 transition-opacity" />
          <Icon className="relative w-5 h-5" />
        </div>
        <h3 className="font-display text-lg font-medium text-ink-900 mb-1.5">{title}</h3>
        <p className="text-sm text-ink-500 leading-relaxed">{body}</p>
      </div>
    </Reveal>
  );
};

const LearnPage = () => (
  <div className="max-w-7xl mx-auto px-4 sm:px-6 pb-24">

    {/* HERO */}
    <section className="pt-16 sm:pt-24 pb-12 text-center">
      <div className="inline-flex items-center gap-2 chip mb-6 animate-fade-up shine">
        <Sparkles className="w-3.5 h-3.5 text-brand-600" /> Understanding autism
      </div>
      <h1 className="font-display text-5xl sm:text-6xl font-medium leading-[0.98] tracking-tight text-ink-900 text-balance animate-fade-up" style={{ animationDelay: '60ms' }}>
        More than a score.
        <br />
        <span className="gradient-text italic">Context for what we measure.</span>
      </h1>
      <p className="mt-6 text-lg text-ink-500 max-w-2xl mx-auto text-balance animate-fade-up" style={{ animationDelay: '120ms' }}>
        Autism is a spectrum, and movement is one of its earliest visible signals. Here's what
        that means — and exactly where an objective motor screen fits in.
      </p>
    </section>

    {/* WHAT ASD IS */}
    <section className="mb-24">
      <SectionHead eyebrow="The basics" title="What autism spectrum disorder is">
        ASD is a neurodevelopmental condition affecting two core areas: social communication and
        interaction, and restricted, repetitive patterns of behavior or interests. It's common —
        roughly <span className="font-semibold text-ink-700">1 in 31 children</span> in the US
        (CDC, 2025) — and presentation varies enormously from one person to the next. That variation
        is exactly why it's called a <span className="italic">spectrum</span>.
      </SectionHead>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {[
          { icon: Brain, t: 'Social communication', b: 'Differences in reciprocal interaction, non-verbal cues, and building relationships.' },
          { icon: Repeat, t: 'Repetitive behaviors', b: 'Patterned movements, routines, intense focused interests, sensory sensitivities.' },
          { icon: Baby, t: 'Appears early', b: 'Signs typically emerge in the first two years — often in movement before speech.' },
        ].map(({ icon: Icon, t, b }, i) => (
          <Reveal key={t} delay={i * 90}>
            <div className="surface rounded-3xl p-6 h-full card-hover">
              <div className="grid place-items-center w-11 h-11 rounded-2xl bg-brand-50 text-brand-600 mb-4">
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="font-display text-lg font-medium text-ink-900 mb-1.5">{t}</h3>
              <p className="text-sm text-ink-500 leading-relaxed">{b}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>

    {/* SEVERITY */}
    <section className="mb-24">
      <SectionHead eyebrow="The spectrum" title="Severity is measured in support needs">
        The DSM-5 describes three severity levels, rated <span className="font-semibold text-ink-700">separately</span> for
        social communication and for repetitive behaviors. Crucially, these levels are assigned by a
        clinician across the whole picture — they are <span className="font-semibold text-ink-700">not</span> a single
        number, and Kinetiq does not grade them.
      </SectionHead>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {SEVERITY.map((s, i) => (
          <Reveal key={s.level} delay={i * 100}>
            <div className="relative surface rounded-3xl p-6 h-full overflow-hidden card-hover">
              <div className={`absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r ${s.tone}`} />
              <div className="flex items-baseline gap-2 mb-2 mt-1">
                <span className={`font-display text-2xl font-semibold bg-gradient-to-r ${s.tone} bg-clip-text text-transparent`}>{s.level}</span>
              </div>
              <p className="font-semibold text-ink-800 mb-2">{s.label}</p>
              <p className="text-sm text-ink-500 leading-relaxed">{s.body}</p>
            </div>
          </Reveal>
        ))}
      </div>

      <Reveal className="mt-6 flex items-start gap-3 surface rounded-2xl p-4">
        <Info className="w-5 h-5 text-brand-600 mt-0.5 shrink-0" />
        <p className="text-sm text-ink-500 leading-relaxed">
          <span className="font-semibold text-ink-700">How Kinetiq relates:</span> our peak-risk score
          is a measure of <span className="italic">motor-kinematic atypicality</span> — it reflects how
          strongly a recording resembles atypical movement patterns, which correlates with the
          likelihood of needing further evaluation. It is not a DSM severity level and not a diagnosis.
        </p>
      </Reveal>
    </section>

    {/* MOTOR SIGNS — where we come in */}
    <section className="mb-24 perspective">
      <SectionHead eyebrow="Our wedge" title="Why movement comes first">
        Motor differences are among the earliest and most consistent markers of atypical development —
        often visible <span className="font-semibold text-ink-700">before</span> social-communication
        signs become obvious. That's the signal Kinetiq is built to read, objectively, from ordinary
        video. These are the four things our pipeline quantifies:
      </SectionHead>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {MOTOR_SIGNS.map(({ icon, title, body }, i) => (
          <MotorCard key={title} icon={icon} title={title} body={body} delay={i * 90} />
        ))}
      </div>

      <Reveal delay={120} className="mt-6 text-center">
        <Link to="/" className="btn-primary">
          <Activity className="w-4 h-4" /> Run a screening
        </Link>
      </Reveal>
    </section>

    {/* CARE JOURNEY */}
    <section className="mb-24">
      <SectionHead eyebrow="Where we fit" title="One step in a longer journey" />
      <div className="surface rounded-3xl p-8 sm:p-10">
        <CareJourney />
      </div>
    </section>

    {/* WHY EARLY + RESOURCES */}
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Reveal>
        <div className="surface rounded-3xl p-8 h-full">
          <div className="grid place-items-center w-11 h-11 rounded-2xl bg-ink-900 text-white mb-4">
            <Clock className="w-5 h-5" />
          </div>
          <h3 className="font-display text-2xl font-medium text-ink-900 mb-2">Why early matters</h3>
          <p className="text-sm text-ink-500 leading-relaxed mb-4">
            The developing brain is most adaptable in the earliest years. Earlier evaluation means
            earlier access to support, and intervention started sooner is consistently linked to
            better developmental and adaptive outcomes. The bottleneck is rarely the therapy — it's
            the <span className="font-semibold text-ink-700">months of waiting</span> to be seen. That
            is the gap an accessible, objective screen can help close.
          </p>
        </div>
      </Reveal>

      <Reveal delay={90}>
        <div className="surface rounded-3xl p-8 h-full">
          <div className="grid place-items-center w-11 h-11 rounded-2xl bg-ink-900 text-white mb-4">
            <ClipboardCheck className="w-5 h-5" />
          </div>
          <h3 className="font-display text-2xl font-medium text-ink-900 mb-4">If you want to go further</h3>
          <ul className="space-y-3 text-sm">
            <li className="flex items-start gap-3">
              <ClipboardCheck className="w-4 h-4 text-brand-600 mt-0.5 shrink-0" />
              <span className="text-ink-600"><span className="font-semibold text-ink-800">M-CHAT-R/F</span> — a validated parent questionnaire for toddlers 16–30 months.</span>
            </li>
            <li className="flex items-start gap-3">
              <Stethoscope className="w-4 h-4 text-brand-600 mt-0.5 shrink-0" />
              <span className="text-ink-600"><span className="font-semibold text-ink-800">Developmental evaluation</span> — a pediatrician or specialist using tools like the ADOS-2.</span>
            </li>
            <li className="flex items-start gap-3">
              <HeartHandshake className="w-4 h-4 text-brand-600 mt-0.5 shrink-0" />
              <span className="text-ink-600"><span className="font-semibold text-ink-800">Early-intervention services</span> — publicly available in many regions for young children.</span>
            </li>
          </ul>
          <p className="mt-4 text-xs text-ink-400 leading-relaxed">
            Kinetiq is a research-preview screening aid, not a diagnostic device. Always consult a
            qualified clinician about any developmental concern.
          </p>
        </div>
      </Reveal>
    </section>
  </div>
);

export default LearnPage;
