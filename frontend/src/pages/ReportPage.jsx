import React, { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ChevronLeft, Download, Info } from 'lucide-react';
import { generateClinicalTimestamp } from '../utils/formatters.js';

// --- OT-facing helpers: turn raw kinematic findings into plain clinical language --- //
const interpretFinding = (f) => {
  const m = (f.marker || '').toLowerCase();
  const below = (f.direction || '').includes('below');
  const aspect = (f.aspect || '').toLowerCase();
  const side = m.includes('right') ? 'right ' : m.includes('left') ? 'left ' : '';
  const variab = aspect.includes('variab');
  if (m.includes('leg angle') || m.includes('knee')) {
    return below
      ? `The ${side}leg under-extends — the knee/hip stays flexed through stance instead of reaching full extension. Consistent with a crouched or short-stance gait.`
      : `The ${side}leg extends more than typical in stance (possible hyperextension).`;
  }
  if (m.includes('ankle-to-ankle') || m.includes('base')) {
    if (variab) return `Step width varies stride-to-stride — an inconsistent base of support, pointing to reduced stepping rhythm and control.`;
    return (f.direction || '').includes('above')
      ? `Wide base of support — the feet are placed far apart, often a compensatory strategy for balance and stability.`
      : `Narrow base of support — the feet track unusually close together.`;
  }
  if (m.includes('arm angle')) {
    return below ? `The ${side}arm is held more flexed than typical — a guarded or reduced arm posture during gait.`
                 : `The ${side}arm is held more extended than typical.`;
  }
  if (m.includes('wrist-to-wrist')) {
    if (variab) return below
      ? `Reduced differential arm swing — the arms move together rather than alternating, so inter-wrist distance barely changes.`
      : `Inter-wrist distance is highly variable across the walk.`;
    return `Inter-wrist distance is ${f.direction}.`;
  }
  if (m.includes('head-to') && m.includes('wrist')) {
    return `Hand carriage relative to the head is ${f.direction} — the ${side}hand is held ${below ? 'closer/higher' : 'further/lower'} than typical during gait.`;
  }
  if (m.includes('elbow')) return below ? `The ${side}elbow is held more flexed than typical.` : `The ${side}elbow extends more than typical.`;
  if (m.includes('hip')) return `${side ? side[0].toUpperCase() + side.slice(1) : ''}hip ${aspect} is ${f.direction}.`;
  if (m.includes('shoulder')) return `${side ? side[0].toUpperCase() + side.slice(1) : ''}shoulder ${aspect} is ${f.direction}.`;
  if (m.includes('spine') || m.includes('trunk')) return `Trunk/spine ${aspect} is ${f.direction} — posture deviates from the typical range.`;
  return `${f.marker} (${f.aspect}) is ${f.direction} — ${f.patient_value} vs ${f.typical_value} typical.`;
};

const otFocusAreas = (regionData, findings) => {
  const areas = [];
  const has = (kw) => findings.some((f) => (f.marker || '').toLowerCase().includes(kw));
  const elevated = (name) => regionData.some((r) => r.name === name && r.status === 'Elevated');
  const anyVar = findings.some((f) => (f.aspect || '').toLowerCase().includes('variab'));
  if (has('leg angle') || has('knee'))
    areas.push({ tag: 'Gait', text: 'Terminal knee/hip extension in stance — one or both legs under-extend. Consider stance-phase strengthening, terminal-extension cueing, and gait training toward fuller weight-bearing extension.' });
  if (has('ankle-to-ankle') || has('base'))
    areas.push({ tag: 'Balance', text: 'Postural stability and narrowing the base — the wide/variable base reads as a balance-compensation strategy. Dynamic-balance and postural-control work, progressing toward a steadier, narrower stance.' });
  if (anyVar)
    areas.push({ tag: 'Rhythm', text: 'Consistent stepping — variable spacing suggests rhythm/timing. Cued or metronome-paced gait may help regularise the base of support.' });
  if (elevated('Upper Body') || has('arm') || has('wrist'))
    areas.push({ tag: 'Upper limb', text: 'Arm-swing amplitude and symmetry — upper-body contribution is elevated. Observe arm swing directly and incorporate if reduced or guarded swing is confirmed clinically.' });
  if (elevated('Symmetry'))
    areas.push({ tag: 'Symmetry', text: 'Bilateral coordination — left–right timing is reduced. Reciprocal, rhythmic bilateral tasks may help restore alternating coordination.' });
  if (!areas.length)
    areas.push({ tag: 'Observe', text: 'No single domain dominates the screen — a general movement-quality observation is suggested to corroborate the result before setting goals.' });
  return areas;
};

const MODALITY = { video: 'video → MediaPipe', 'csv-coordinates': 'coordinate CSV', 'csv-features': 'engineered features' };

const OTR_CSS = `
.otr-report{--paper:#f5f7f7;--surface:#fff;--ink:#181d1e;--muted:#5a6568;--faint:#879497;
  --hair:#d9e0e0;--hair-soft:#e8eded;--tint:#eaf3f2;--accent:#0e7c86;--elev:#b0730f;--elev-bg:#fbf1e0;
  --ok:#2f7d55;--ok-bg:#e7f3ec;--violet:#5f52a0;--violet-bg:#efedf8;
  --mono:ui-monospace,"Cascadia Code","Consolas","SF Mono",Menlo,monospace;
  color:var(--ink);font-size:15px;line-height:1.55}
.otr-report *{box-sizing:border-box}
.otr-sheet{background:var(--surface);border:1px solid var(--hair);border-radius:14px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.otr-report .top{padding:26px 30px;border-bottom:1px solid var(--hair);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;align-items:flex-start}
.otr-brand{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.otr-report h1{font-size:22px;font-weight:680;letter-spacing:-.01em;margin:10px 0 3px}
.otr-sub{color:var(--muted);font-size:13px}
.otr-meta{text-align:right;font-size:12.5px;color:var(--muted);font-family:var(--mono);line-height:1.7}
.otr-meta b{color:var(--ink);font-weight:600}
.otr-report section{padding:22px 30px;border-top:1px solid var(--hair-soft)}
.otr-lbl{font:600 10.5px/1 var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin:0 0 14px}
.otr-result{padding:22px 30px;background:var(--elev-bg)}
.otr-result.ok{background:var(--ok-bg)} .otr-result.bl{background:var(--violet-bg)}
.otr-rflag{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.otr-dot{width:11px;height:11px;border-radius:50%;background:var(--elev);flex:none;box-shadow:0 0 0 4px color-mix(in srgb,var(--elev) 18%,transparent)}
.otr-result.ok .otr-dot{background:var(--ok);box-shadow:0 0 0 4px color-mix(in srgb,var(--ok) 18%,transparent)}
.otr-result.bl .otr-dot{background:var(--violet);box-shadow:0 0 0 4px color-mix(in srgb,var(--violet) 18%,transparent)}
.otr-rtitle{font-size:19px;font-weight:680;letter-spacing:-.01em;color:var(--ink)}
.otr-rconf{margin-left:auto;font:600 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--elev);
  border:1px solid color-mix(in srgb,var(--elev) 40%,var(--hair));border-radius:20px;padding:6px 11px}
.otr-result.ok .otr-rconf{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,var(--hair))}
.otr-result.bl .otr-rconf{color:var(--violet);border-color:color-mix(in srgb,var(--violet) 40%,var(--hair))}
.otr-rmain{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap}
.otr-rmain>div:first-child{flex:1 1 340px}
.otr-rscorebig{text-align:right;flex:none;min-width:96px}
.otr-rscorebig .n{font:680 36px/1 inherit;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.otr-rscorebig .l{font:600 10px/1.3 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.otr-rline{color:var(--ink);font-size:14px;margin:12px 0 0;max-width:60ch}
.otr-rscore{margin-top:16px;padding-top:14px;border-top:1px solid color-mix(in srgb,var(--ink) 8%,transparent);display:flex;gap:22px;flex-wrap:wrap;font-size:12.5px;color:var(--muted)}
.otr-rscore b{color:var(--ink);font-family:var(--mono);font-size:15px}
.otr-warn{margin:14px 0 0;padding:11px 14px;border:1px solid color-mix(in srgb,var(--elev) 30%,var(--hair));border-radius:9px;background:color-mix(in srgb,var(--elev) 7%,transparent);font-size:12px;color:var(--ink);line-height:1.5}
.otr-domains{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.otr-dcard{border:1px solid var(--hair);border-radius:10px;padding:14px 15px;background:var(--surface)}
.otr-dh{display:flex;justify-content:space-between;align-items:baseline}
.otr-dn{font-weight:640;font-size:14px} .otr-dv{font:640 22px/1 inherit;font-variant-numeric:tabular-nums;margin-top:8px}
.otr-pill{font:600 9.5px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;padding:4px 8px;border-radius:20px}
.otr-pill.el{color:var(--elev);background:var(--elev-bg)} .otr-pill.ty{color:var(--ok);background:var(--ok-bg)}
.otr-dd{color:var(--muted);font-size:11.5px;margin-top:9px;line-height:1.5}
.otr-bar{height:5px;border-radius:3px;background:var(--hair-soft);margin-top:11px;overflow:hidden}
.otr-bar span{display:block;height:100%;border-radius:3px}
.otr-find{border:1px solid var(--hair);border-radius:10px;overflow:hidden}
.otr-frow{padding:14px 16px;border-top:1px solid var(--hair-soft);display:grid;grid-template-columns:1fr auto;gap:6px 16px}
.otr-frow:first-child{border-top:none}
.otr-fname{font-weight:640;font-size:14px} .otr-fname small{color:var(--faint);font-weight:600;text-transform:uppercase;font-size:10px;margin-left:8px}
.otr-fnums{font-family:var(--mono);font-size:12px;color:var(--muted);white-space:nowrap;text-align:right}
.otr-z{font-weight:700} .otr-z.hi{color:var(--elev)} .otr-z.lo{color:var(--accent)}
.otr-finterp{grid-column:1/-1;color:var(--muted);font-size:12.5px;line-height:1.5;margin-top:2px}
.otr-focus{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:11px}
.otr-focus li{display:flex;gap:12px;padding:13px 15px;border:1px solid var(--hair);border-radius:10px;font-size:13.5px;line-height:1.5}
.otr-tag{font:600 9.5px/1 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--accent);
  border:1px solid color-mix(in srgb,var(--accent) 35%,var(--hair));border-radius:6px;padding:5px 7px;height:fit-content;white-space:nowrap;flex:none;margin-top:1px}
.otr-focus b{color:var(--ink)}
.otr-scroll{overflow-x:auto}
.otr-report table{width:100%;border-collapse:collapse;font-size:13px;min-width:440px}
.otr-report th,.otr-report td{text-align:right;padding:10px;border-bottom:1px solid var(--hair-soft)}
.otr-report th:first-child,.otr-report td:first-child{text-align:left}
.otr-report thead th{font:600 10px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.otr-report tbody td{font-variant-numeric:tabular-nums} .otr-report tbody td:first-child{font-weight:520}
.otr-base{font-family:var(--mono)} .otr-blank{color:var(--faint)}
.otr-note{background:var(--tint);border:1px solid color-mix(in srgb,var(--accent) 22%,var(--hair));border-radius:10px;padding:13px 16px;font-size:12.5px;color:var(--ink);line-height:1.55}
.otr-note b{color:var(--accent)}
.otr-disc{padding:20px 30px;border-top:1px solid var(--hair);color:var(--muted);font-size:11.5px;line-height:1.6}
.otr-disc b{color:var(--ink)}
@media (max-width:640px){.otr-domains{grid-template-columns:1fr}.otr-meta{text-align:left}}
@media print{.otr-sheet{border:none;box-shadow:none}.otr-no-print{display:none!important}}
`;

const ReportPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const report = location.state?.patientData || location.state?.report;

  const reportId = useMemo(() => {
    const d = new Date();
    const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
    return `CG-${ymd}-${Math.random().toString(16).slice(2, 6).toUpperCase()}`;
  }, []);
  const generatedAt = useMemo(() => generateClinicalTimestamp(), []);

  if (!report) {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center text-center px-4">
        <div className="grid place-items-center w-14 h-14 rounded-2xl bg-ink-900 text-white mb-5 shadow-glow"><Info className="w-6 h-6" /></div>
        <h2 className="font-display text-3xl font-medium text-ink-900 mb-2">No analysis data found</h2>
        <p className="text-ink-500 mb-6 max-w-sm">Run a recording through the model to generate a report.</p>
        <button onClick={() => navigate('/')} className="btn-primary">Return to assessment</button>
      </div>
    );
  }

  const {
    final_risk_score = 0, is_atypical = false, borderline = false, decision_threshold = 0.22,
    age_group = 'child', clinical_summary = {}, regional_breakdown, regions, input_meta = {},
    quality = { warnings: [] }, disclaimer, kinematic_findings = [], result_stability = {},
  } = report;

  const cs = clinical_summary;
  const regionData = (regional_breakdown && regional_breakdown.length
    ? regional_breakdown
    : (regions || []).map((r) => ({ name: r.name, contribution_pct: r.value, status: 'Typical', description: '' })));
  const findings = kinematic_findings;
  const modality = MODALITY[input_meta.kind] || (input_meta.kind || 'analysis').replace(/-/g, ' ');

  // result framing
  const resultClass = !is_atypical ? 'ok' : borderline ? 'bl' : '';
  const title = cs.classification || (is_atypical ? 'Atypical gait kinematics flagged' : 'Typical motor development');
  const conf = !is_atypical ? 'Within typical range'
    : borderline ? 'Borderline · low confidence'
    : (cs.severity ? `${cs.severity} confidence` : 'Flagged');
  const driver = cs.primary_driver || (regionData.find((r) => r.status === 'Elevated')?.name);
  const rline = cs.narrative || (is_atypical
    ? `The movement pattern in this recording differs from typically-developing peers${driver ? `, driven mainly by ${driver.toLowerCase()}` : ''}.${borderline ? ' It sits near the decision threshold — treat as low-confidence and consider a repeat recording.' : ''} This is a screening signal, not a diagnosis — it indicates gait features worth a closer clinical look.`
    : `The gait kinematics in this recording fall within the typical range for this age group. Screening did not flag atypical movement.`);

  const focus = otFocusAreas(regionData, findings);
  const zClass = (z) => (Number(z) < 0 ? 'lo' : 'hi');
  const resultColor = !is_atypical ? 'var(--ok)' : borderline ? 'var(--violet)' : 'var(--elev)';

  return (
    <div className="otr-report max-w-[860px] mx-auto px-4 sm:px-6 pt-6 pb-24">
      <style>{OTR_CSS}</style>

      {/* toolbar (screen only) */}
      <div className="otr-no-print flex justify-between items-center mb-5">
        <button onClick={() => navigate('/')} className="btn-ghost -ml-2"><ChevronLeft className="w-5 h-5" /> Back to screening</button>
        <button onClick={() => window.print()} className="btn-primary"><Download className="w-4 h-4" /> Export / Print PDF</button>
      </div>

      <div className="otr-sheet">
        {/* masthead */}
        <div className="top">
          <div>
            <div className="otr-brand">Movement screening · occupational therapy</div>
            <h1>Gait Kinematics Report</h1>
            <div className="otr-sub">3D skeletal gait analysis · handcrafted + movement-model fusion</div>
          </div>
          <div className="otr-meta">
            Case ID <b>{reportId}</b><br />
            Age band <b>{age_group === 'adult' ? 'Adult · provisional' : 'Child (3–12)'}</b><br />
            Session <b>1 · baseline</b><br />
            Source <b>{modality}</b><br />
            Generated <b>{generatedAt}</b>
          </div>
        </div>

        {/* result banner */}
        <div className={`otr-result ${resultClass}`}>
          <div className="otr-rmain">
            <div>
              <div className="otr-rflag">
                <span className="otr-dot" />
                <span className="otr-rtitle">{title}</span>
                <span className="otr-rconf">{conf}</span>
              </div>
              <p className="otr-rline">{rline}</p>
            </div>
            <div className="otr-rscorebig" style={{ color: resultColor }}>
              <div className="n">{Number(final_risk_score).toFixed(2)}</div>
              <div className="l">score<br />flag ≥ {decision_threshold}</div>
            </div>
          </div>
          <div className="otr-rscore">
            {result_stability.consistency && <span>Consistency <b>{result_stability.consistency}</b></span>}
            <span>Windows analysed <b>{input_meta.windows_analyzed ?? '—'}</b></span>
            <span>Primary driver <b>{driver || '—'}</b></span>
            <span>Input quality <b>{quality.warnings?.length ? 'see note' : 'OK'}</b></span>
          </div>
          {quality.warnings?.length > 0 && (
            <div className="otr-warn">{quality.warnings.join(' ')}</div>
          )}
        </div>

        {/* movement domains */}
        {regionData.length > 0 && (
          <section>
            <p className="otr-lbl">Where it's coming from — movement domains</p>
            <div className="otr-domains">
              {regionData.map((r) => {
                const el = r.status === 'Elevated';
                const color = el ? 'var(--elev)' : 'var(--ok)';
                return (
                  <div className="otr-dcard" key={r.name}>
                    <div className="otr-dh"><span className="otr-dn">{r.name}</span>
                      <span className={`otr-pill ${el ? 'el' : 'ty'}`}>{el ? 'Elevated' : 'Typical'}</span></div>
                    <div className="otr-dv" style={{ color }}>{(r.contribution_pct ?? 0).toFixed(0)}%</div>
                    <div className="otr-bar"><span style={{ width: `${Math.min(r.contribution_pct ?? 0, 100)}%`, background: color }} /></div>
                    {r.description && <div className="otr-dd">{r.description}</div>}
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* findings + interpretation */}
        {findings.length > 0 && (
          <section>
            <p className="otr-lbl">What stood out — measured kinematic findings</p>
            <div className="otr-find">
              {findings.slice(0, 6).map((f, i) => (
                <div className="otr-frow" key={i}>
                  <div className="otr-fname">{f.marker}<small>{f.region}</small></div>
                  <div className="otr-fnums">{f.patient_value} <span style={{ color: 'var(--faint)' }}>vs</span> {f.typical_value} &nbsp; <span className={`otr-z ${zClass(f.z_score)}`}>z {Number(f.z_score) > 0 ? '+' : ''}{f.z_score}</span></div>
                  <div className="otr-finterp">{interpretFinding(f)}</div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* focus areas */}
        {findings.length > 0 && (
          <section>
            <p className="otr-lbl">Where the pattern points — possible OT focus areas</p>
            <ul className="otr-focus">
              {focus.map((a, i) => (
                <li key={i}><span className="otr-tag">{a.tag}</span><span>{a.text}</span></li>
              ))}
            </ul>
            <p className="otr-note" style={{ marginTop: 14 }}><b>Clinical judgment required.</b> These are directions the measured pattern points to — <b>not a prescription</b>. The features are correlational markers of atypical gait, not validated therapeutic targets. Confirm against direct observation before setting goals.</p>
          </section>
        )}

        {/* progress monitoring */}
        {findings.length > 0 && (
          <section>
            <p className="otr-lbl">Progress monitoring — re-measure these over sessions</p>
            <p className="otr-sub" style={{ margin: '-6px 0 14px' }}>Recapture a short natural-walk clip each session; the same metrics recompute so change is objective. Baseline values below.</p>
            <div className="otr-scroll">
              <table>
                <thead><tr><th>Metric</th><th>Baseline</th><th>Typical</th><th>Session 2</th><th>Session 3</th></tr></thead>
                <tbody>
                  {findings.slice(0, 4).map((f, i) => (
                    <tr key={i}>
                      <td>{f.marker} <span style={{ color: 'var(--faint)', textTransform: 'capitalize', fontSize: 12 }}>({f.aspect})</span></td>
                      <td className="otr-base">{f.patient_value}</td>
                      <td className="otr-base" style={{ color: 'var(--muted)' }}>{f.typical_value}</td>
                      <td className="otr-blank">—</td><td className="otr-blank">—</td>
                    </tr>
                  ))}
                  <tr style={{ fontWeight: 700 }}>
                    <td>Overall screening score</td>
                    <td className="otr-base">{Number(final_risk_score).toFixed(2)}</td>
                    <td className="otr-base" style={{ color: 'var(--muted)' }}>&lt;{decision_threshold}</td>
                    <td className="otr-blank">—</td><td className="otr-blank">—</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* disclaimer */}
        <div className="otr-disc">
          <b>Screening aid — not a diagnostic device.</b> {disclaimer || 'This report summarises a 3D gait-kinematics screen and supports, not replaces, clinical judgment; the occupational therapist remains the decision-maker. A flag indicates movement features that differ from typically-developing peers and warrant a closer look — it is not a diagnosis. Kinematic findings are correlational markers, not validated therapy targets.'}{age_group === 'adult' ? ' Adult mode is provisional (movement model validated on children ages ~3–12); a “typical” adult result is not validated.' : ''}
        </div>
      </div>
    </div>
  );
};

export default ReportPage;
