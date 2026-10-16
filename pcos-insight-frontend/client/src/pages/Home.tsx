import { useEffect, useRef, useState } from 'react';
import { ArrowRight, ArrowDownRight, CircleAlert, RefreshCw, ScanLine, ShieldCheck,
  Sparkles, Layers3, Dna, Gauge, Menu, X, Download, Check, LockKeyhole } from 'lucide-react';
import { api, type Schema, type Stage, type Prediction, type Explanation, type Comparison } from '../api';
import { LogoMark, NetworkVisual } from '../visuals';
import { describeField, helpId } from '../fieldDescriptions';
import { clinicalReferences } from '../clinicalReferences';

const percent = (v: number) => `${(v * 100).toFixed(1)}%`;
const metricNames: Record<string, string> = { recall: 'Sensitivity', specificity: 'Specificity',
  roc_auc: 'ROC-AUC', f1: 'F1', accuracy: 'Accuracy', balanced_accuracy: 'Balanced accuracy' };
function Label({ n, children }: { n: string; children: React.ReactNode }) {
  return <div className="section-label"><span className="mono label-index">{n}</span><span>{children}</span></div>;
}
function choiceLabel(name: string, v: number) {
  if (name.trim() === 'Cycle(R/I)') return v === 2 ? 'Regular' : v === 4 ? 'Irregular' : String(v);
  return v === 0 ? 'No' : v === 1 ? 'Yes' : String(v);
}

function Screening({ schema }: { schema: Schema }) {
  const [stageId, setStageId] = useState(schema.stages[0].id);
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [error, setError] = useState('');
  const [explainError, setExplainError] = useState('');
  const [busy, setBusy] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const stage = schema.stages.find(s => s.id === stageId)!;
  const count = stage.fields.filter(f => values[f.name] !== undefined && values[f.name] !== '').length;
  const completion = Math.round(count / stage.n_features * 100);
  const tiers = [...new Set(stage.fields.map(f => f.tier))];
  const clearResult = () => { setResult(null); setExplanation(null); setError(''); setExplainError(''); };
  const run = async (event: React.FormEvent) => {
    event.preventDefault(); clearResult();
    if (count !== stage.n_features) { setError('Complete every required input in this feature group.'); return; }
    const numeric = Object.fromEntries(stage.fields.map(f => [f.name, Number(values[f.name])]));
    if (Object.values(numeric).some(v => !Number.isFinite(v))) { setError('All inputs must be finite numbers.'); return; }
    const payload = { stage: stage.id, values: numeric, fingerprint: schema.fingerprint };
    const controller = new AbortController(); request.current = controller; setBusy(true);
    try {
      const prediction = await api<Prediction>('predict', payload, controller.signal);
      setResult(prediction); setExplaining(true);
      try {
        const exp = await api<Explanation>('explain', payload, controller.signal);
        if (Math.abs(exp.score - prediction.score) > 1e-6) throw new Error('Explanation does not match this prediction. Restart the model service.');
        setExplanation(exp);
      } catch (e) { if (!controller.signal.aborted) setExplainError((e as Error).message); }
    } catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    finally { setBusy(false); setExplaining(false); }
  };
  const download = () => {
    const data = { result, explanation, evaluation: schema.evaluation_label,
      note: 'Screening support only. Not a diagnosis or a calibrated individual disease risk.' };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'pcos-screening-summary.json'; link.click();
    URL.revokeObjectURL(url);
  };
  return <section id="screen" className="section-space screening-section"><div className="container">
    <div className="screening-banner"><div><Label n="02">Patient screening</Label>
      <h2 className="display section-title">Start with what<br /><em>is available.</em></h2>
      <p>Choose the information you have. Each group uses its own fitted model and the exact inputs from your notebook.</p></div>
      <div className="screening-side-note"><LockKeyhole size={18} /><span>Processed locally.<br /><b>No prediction history is saved.</b></span></div></div>
    <div className="screening-layout">
      <form className="screening-form soft-panel glow-panel" onSubmit={run}>
        <div className="form-top"><div><span className="eyebrow">{stage.n_features} required inputs</span>
          <h3 className="display">Build a screening profile</h3></div>
          <div className="completion"><span className="mono">{completion}%</span><small>complete</small>
            <div className="completion-track"><i style={{ width: `${completion}%` }} /></div></div></div>
        <div className="stage-pills">{schema.stages.map(s => <button type="button" key={s.id} disabled={busy}
          aria-pressed={stageId === s.id} className={stageId === s.id ? 'stage-pill-active' : ''}
          onClick={() => { setStageId(s.id); clearResult(); }}><span className="mono">{s.id.split('_')[0]}</span>{s.title}</button>)}</div>
        <p className="form-model">{stage.model} <span>·</span> {stage.n_features} inputs <span>·</span> exported threshold {percent(stage.threshold)}</p>
        <fieldset disabled={busy} className="profile-fields">
          {tiers.map(tier => <div key={tier} className="field-group"><h4 className="eyebrow">{tier}</h4>
            <div className="form-grid">{stage.fields.filter(f => f.tier === tier).map(f => <label className="field" key={f.name}>
              <span>{f.label}</span>{f.choices ?
                <select required aria-label={f.label} aria-describedby={helpId(f.name)} value={values[f.name] ?? ''}
                  onChange={e => { setValues(v => ({ ...v, [f.name]: e.target.value })); clearResult(); }}>
                  <option value="" disabled>Select an option</option>{f.choices.map(v => <option key={v} value={v}>{choiceLabel(f.name, v)}</option>)}
                </select> :
                <input aria-label={f.label} aria-describedby={helpId(f.name)} required type="number" min="0" step="any" placeholder="Enter value"
                  value={values[f.name] ?? ''} onChange={e => { setValues(v => ({ ...v, [f.name]: e.target.value })); clearResult(); }} />}
              <small id={helpId(f.name)} className="field-description">{describeField(f.name)}</small>
            </label>)}</div></div>)}
        </fieldset>
        <p className="helper">Bleeding duration means days of menstrual bleeding.</p>
        {error && <div role="alert" className="form-error"><CircleAlert size={16} />{error}</div>}
        <div className="form-actions"><button className="primary-button btn-lift" type="submit" disabled={busy}>
          {busy ? <RefreshCw size={16} className="spin" /> : <ScanLine size={16} />}{busy ? 'Computing locally…' : 'Screen and explain'}<ArrowRight size={16} /></button>
          <button type="button" className="text-button" disabled={busy} onClick={() => { setValues({}); clearResult(); }}>Clear profile</button></div>
      </form>
      <div className="screening-result" aria-live="polite">
        {!result ? <div className="awaiting-card soft-panel"><div className="awaiting-orbit"><div className="awaiting-core"><ScanLine size={25} /></div></div>
          <span className="eyebrow">{busy ? 'Computing screening score' : 'Ready when you are'}</span>
          <h3 className="display">Your profile.<br /><em>A clearer signal.</em></h3>
          <p>Complete the {stage.n_features} inputs to run {stage.model}. Your model score and feature explanation will appear here.</p>
          <div className="awaiting-meta"><span className="mono">GROUP</span><b>{stage.id.split('_')[0]}</b><span className="mono">MODEL</span><b>{stage.model}</b></div></div> :
          <div className="soft-panel result-card"><span className="eyebrow">PCOS-class model score</span>
            <div className="score-number display">{percent(result.score)}</div>
            <div className={`screen-class ${result.screening_class.startsWith('Positive') ? 'screen-positive' : ''}`}>{result.screening_class}</div>
            <div className="score-track"><span style={{ width: percent(result.score) }} /><i style={{ left: percent(result.threshold) }} /></div>
            <p>Classification threshold: {percent(result.threshold)}. This is an uncalibrated model score, not your probability of having PCOS.</p>
            {result.warnings.map(w => <p className="range-warning" key={w}>{w}</p>)}
            <div className="result-details"><span>{result.model}</span><span>{result.stage.split('_')[0]} · {stage.n_features} inputs</span></div>
            <button className="secondary-button" onClick={download}><Download size={15} />Download summary</button>
          </div>}
        {explaining && <p className="helper explaining"><RefreshCw size={14} className="spin" />Score ready. Computing feature contributions…</p>}
        {explainError && <p className="form-error" role="alert">Score available; explanation unavailable: {explainError}</p>}
        {explanation && <div className="soft-panel contribution-panel"><span className="eyebrow">Local SHAP explanation</span>
          <h3 className="display">What moved the score?</h3><p>Reference {percent(explanation.base_score)} → model score {percent(explanation.score)}</p>
          <div className="contribution-legend"><span>← Lowers score</span><span>Raises score →</span></div>
          {explanation.contributions.map(c => {
            const max = Math.max(...explanation.contributions.map(v => Math.abs(v.contribution_pp)), 1);
            const width = Math.abs(c.contribution_pp) / max * 48;
            return <div className="contribution-row" key={c.feature}><div><b>{c.feature}</b><small>Entered value: {c.value}</small></div>
              <div className="signed-track"><span className={c.contribution_pp >= 0 ? 'positive-bar' : 'negative-bar'}
                style={{ width: `${width}%`, left: `${c.contribution_pp >= 0 ? 50 : 50-width}%` }} /></div>
              <strong>{c.contribution_pp > 0 ? '+' : ''}{c.contribution_pp.toFixed(2)}<small>pp</small></strong></div>;
          })}
          <p className="helper">pp = percentage points. Reference + all contributions equals the score. These explain model behavior, not medical causes.</p></div>}
      </div>
    </div><div className="disclaimer"><CircleAlert size={17} /><p><b>Screening support only.</b> A positive screen is not a diagnosis. A negative screen does not rule out PCOS. Review symptoms and results with a clinician.</p></div>
  </div></section>;
}

function ModelLab({ schema, rows }: { schema: Schema; rows: Comparison[] }) {
  const [analysis, setAnalysis] = useState('Primary');
  const [stage, setStage] = useState(schema.stages[0].id);
  const [metric, setMetric] = useState('recall');
  const [search, setSearch] = useState('');
  const available = rows.filter(r => r.analysis === analysis && r.feature_set === stage && r.model.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => Number(b[metric + '_mean'] ?? -1) - Number(a[metric + '_mean'] ?? -1));
  const analyses = [...new Set(rows.map(r => r.analysis))];
  const value = (r: Comparison, key: string) => typeof r[key + '_mean'] === 'number' ? percent(r[key + '_mean'] as number) : '—';
  return <section id="lab" className="section-space lab-section"><div className="container">
    <div className="lab-header"><div><Label n="03">Model console</Label><h2 className="display section-title">MODEL <em>LAB</em></h2>
      <p className="section-intro">Real comparisons from your notebook export. Filter by analysis and group, then choose the metric to rank.</p></div>
      <div className="lab-badge"><Dna size={20} /><span><b>EXPLORATORY RESULTS</b><small>Provenance stays visible</small></span></div></div>
    <div className="lab-toolbar"><select aria-label="Analysis" value={analysis} onChange={e => setAnalysis(e.target.value)}>
      {analyses.map(a => <option key={a}>{a}</option>)}</select>
      <select aria-label="Comparison feature group" value={stage} onChange={e => setStage(e.target.value)}>{schema.stages.map(s => <option key={s.id} value={s.id}>{s.title}</option>)}</select>
      <input className="model-search" aria-label="Search models" placeholder="Search models…" value={search} onChange={e => setSearch(e.target.value)} />
      <span className="toolbar-note">{available.length} model configurations</span></div>
    <div className="matrix-panel soft-panel"><div className="panel-heading"><div><span className="eyebrow">Selected metric / {metricNames[metric]}</span>
      <h3 className="display">Performance, in context.</h3></div><div className="metric-select">{Object.entries(metricNames).map(([key, label]) =>
        <button key={key} className={metric === key ? 'metric-selected' : ''} onClick={() => setMetric(key)}>{label}</button>)}</div></div>
      {available.length ? <><div className="comparison-bars">{available.map(r => <div className="comparison-bar" key={r.model}><span>{r.model}</span>
        <div><i style={{ width: `${Number(r[metric + '_mean'] ?? 0) * 100}%` }} /></div><strong>{value(r, metric)}</strong></div>)}</div>
        <div className="table-scroll"><table><thead><tr><th>Model</th>{['recall', 'specificity', 'f1', 'roc_auc', 'accuracy'].map(m => <th key={m}>{metricNames[m]}</th>)}</tr></thead>
          <tbody>{available.map(r => <tr key={r.model}><td>{r.model}</td>{['recall', 'specificity', 'f1', 'roc_auc', 'accuracy'].map(m => <td key={m}>{value(r, m)}</td>)}</tr>)}</tbody></table></div></> :
        <div className="empty-state">No exported comparisons match these filters. No metrics are substituted from another group.</div>}
      <div className="matrix-note"><CircleAlert size={15} /><span>{schema.evaluation_label}. Fold SD, where available in the export, is not a confidence interval. Selection reuses exploratory CV.</span></div>
      <details className="raw-metrics"><summary>View exported means and fold spread</summary><pre>{JSON.stringify(available, null, 2)}</pre></details>
    </div>
  </div></section>;
}

function CostAnalysis({ stages }: { stages: Stage[] }) {
  const [active, setActive] = useState(stages[0].id);
  const stage = stages.find(s => s.id === active)!;
  return <section id="cost" className="section-space cost-section"><div className="container"><Label n="04">Available information</Label>
    <div className="section-heading-row"><h2 className="display section-title">More information.<br /><em>A different tradeoff.</em></h2>
      <p className="section-intro">Explore exactly which inputs each model needs. These are qualitative acquisition groups; no monetary savings have been established.</p></div>
    <div className="cost-layout"><div className="ladder-panel soft-panel"><div className="ladder-top"><span className="eyebrow">Feature groups</span><span className="mono">{stages.length} distinct regimes</span></div>
      <div className="ladder-steps">{stages.map((s, i) => <button key={s.id} aria-pressed={active === s.id} onClick={() => setActive(s.id)}
        className={`ladder-step ${active === s.id ? 'ladder-step-active' : ''}`} style={{ '--step-accent': ['#77f2d0','#91d7ff','#ad91ff','#f3bd69'][i] } as React.CSSProperties}>
        <span className="ladder-code mono">{s.id.split('_')[0]}</span><span className="ladder-line"><i style={{ width: `${s.n_features / stages[stages.length-1].n_features*100}%` }} /></span>
        <span className="ladder-copy"><b>{s.title}</b><small>{s.model}</small></span><span className="ladder-count mono">{s.n_features} inputs</span><ArrowRight size={15} /></button>)}</div></div>
      <div className="feature-panel"><div className="feature-panel-header"><div><span className="eyebrow">{stage.id}</span><h3 className="display">{stage.title}</h3></div>
        <div className="feature-panel-mark">{stage.n_features}<small>required inputs</small></div></div>
        <div className="feature-grid">{stage.fields.map((f, i) => <div key={f.name} className="feature-chip"><span className="chip-index mono">{String(i+1).padStart(2,'0')}</span><span>{f.label}</span><Check size={13} /></div>)}</div></div></div>
  </div></section>;
}

function InputReference({ schema }: { schema: Schema }) {
  const fields = [...new Map(schema.stages.flatMap(stage => stage.fields).map(field => [field.name, field])).values()];
  const [query, setQuery] = useState('');
  const shown = fields.filter(field => `${field.label} ${field.name} ${field.tier}`.toLowerCase().includes(query.toLowerCase()));
  const number = (value: number) => value.toLocaleString('en-US', { maximumFractionDigits: 3 });
  const rows = [['Connection', 'This app reads fitted models from the notebook artifacts folder. A live notebook kernel is not required.'],
    ['Preprocessing', 'Inputs follow the exported feature order and the original cleaning rules. Learned preprocessing stays in the fitted model.'],
    ['Explanation', 'Permutation SHAP explains the complete predictor in probability units. Contributions are checked for additivity.'],
    ['Evaluation', schema.evaluation_label], ['Selection', schema.selection_rule],
    ['Limits', schema.scope], ['Variable definition', 'Cycle length(days) is duration of menstrual bleeding, as clarified by the project owner.'],
    ['Privacy', 'Forms and predictions stay in process/browser memory. No analytics, session replay, patient database, or prediction history is included. Downloaded summaries are saved only when you request them.']];
  return <section id="about" className="section-space about-section"><div className="container"><Label n="05">Input Reference</Label>
    <h2 className="display section-title">Know the input.<br /><em>Understand the context.</em></h2>
    <p className="section-intro">Training observations describe the exported dataset, including unusual values; they are not medical normal ranges or input limits. Clinical examples depend on age, cycle phase, pregnancy, medicines and the laboratory assay.</p>
    <div className="reference-toolbar"><label htmlFor="reference-search">Find an input</label><input id="reference-search" type="search" placeholder="Search name or feature group…" value={query} onChange={event => setQuery(event.target.value)} /><span>{shown.length} of {fields.length} inputs</span></div>
    <div className="soft-panel table-scroll reference-table"><table><caption>Training observations and clinical context · sources checked 17 October 2026</caption><thead><tr><th scope="col">Input and one-line explanation</th><th scope="col">Observed in training</th><th scope="col">Clinical reference / context</th></tr></thead><tbody>
      {shown.map(field => { const reference = clinicalReferences[field.name.trim()]; return <tr key={field.name}>
        <th scope="row"><b>{field.label}</b><small>{describeField(field.name)}</small><span className="reference-tier">{field.tier}</span></th>
        <td className="reference-observed">{number(field.min_observed)} – {number(field.max_observed)}{field.choices && <small>{field.choices.map(value => `${value} = ${choiceLabel(field.name, value)}`).join(' · ')}</small>}</td>
        <td>{reference?.text ?? 'Use a clinician-interpreted reference for this measurement.'}{reference?.url && <a href={reference.url} target="_blank" rel="noreferrer">{reference.source} ↗</a>}</td>
      </tr>; })}
    </tbody></table>{!shown.length && <p className="empty-state">No inputs match this search.</p>}</div>
    <p className="helper">PCOM means polycystic ovarian morphology. Ultrasound criteria here apply to adults and depend on scan quality and counting method; they are not recommended for diagnosing PCOS in adolescents. These references do not change model predictions.</p>
    <details className="reference-methods"><summary>Model details, evaluation and limitations</summary>
    <div className="about-grid"><div className="method-list">{rows.map(([title, copy], i) => <div className="method-row" key={title}><span className="mono method-number">0{i+1}</span><div><h3>{title}</h3><p>{copy}</p></div></div>)}</div>
      <div className="soft-panel version-panel"><span className="eyebrow">Connected notebook export</span><h3 className="display">Reproducible by design.</h3>
        <dl>{Object.entries(schema.versions).map(([k,v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
        <p className="helper">Export fingerprint<br /><code>{schema.fingerprint}</code></p>
        {schema.version_warnings.map(w => <p className="range-warning" key={w}>{w}</p>)}</div></div>
    </details>
  </div></section>;
}

export default function Home() {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [rows, setRows] = useState<Comparison[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [mobile, setMobile] = useState(false);
  const load = async () => {
    setLoading(true); setError('');
    try { const [s, r] = await Promise.all([api<Schema>('schema'), api<Comparison[]>('comparisons')]); setSchema(s); setRows(r); }
    catch (e) { setError((e as Error).message); } finally { setLoading(false); }
  };
  useEffect(() => { void load(); }, []);
  const go = (id: string) => { setMobile(false); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }); };
  const status = loading ? 'CONNECTING' : error ? 'OFFLINE' : 'CONNECTED';
  return <div className="site-shell"><div className="ambient ambient-one" /><div className="ambient ambient-two" />
    <header className="top-nav top-nav-scrolled"><div className="container nav-inner"><button className="brand" onClick={() => go('overview')} aria-label="PCOS Insight home"><LogoMark /><span>PCOS<br /><b>INSIGHT</b></span></button>
      <nav className={`main-nav ${mobile ? 'main-nav-open' : ''}`} aria-label="Primary navigation">{[['overview','Overview'],['screen','Screen patient'],['lab','Model lab'],['cost','Feature groups'],['about','Input Reference']].map(([id,label]) => <button key={id} onClick={() => go(id)}>{label}</button>)}</nav>
      <div className="nav-meta"><div className={`status-pill ${error ? 'status-offline' : ''}`}><span className="status-dot" /><strong>{status}</strong></div></div>
      <button className="mobile-menu" aria-label="Toggle navigation" aria-expanded={mobile} onClick={() => setMobile(!mobile)}>{mobile ? <X size={20} /> : <Menu size={20} />}</button></div></header>
    <main><section id="overview" className="hero-section"><div className="container hero-grid"><div className="hero-copy"><div className="hero-kicker"><span className="pulse-line" /><span className="mono">EXPLAINABLE AI / COST-AWARE SCREENING</span></div>
      <h1 className="display hero-title">See the signal<br /><em>before the diagnosis.</em></h1>
      <p className="hero-subtitle">Your models. A clearer interface. Explore PCOS screening models, available-test tradeoffs, and individual feature explanations.</p>
      <div className="hero-actions"><button className="primary-button btn-lift" onClick={() => go('screen')}>Screen a patient<ArrowRight size={16} /></button><button className="text-button" onClick={() => go('lab')}>Explore the model lab<ArrowDownRight size={16} /></button></div>
      <div className="hero-caption"><ShieldCheck size={20} /><span>Evaluation support, not a diagnosis.<br />Powered by your notebook’s fitted models.</span></div></div>
      <div className="hero-visual"><div className="visual-topline"><span className="eyebrow">Patient → model → explanation</span><span className="mono">LOCAL INFERENCE</span></div><NetworkVisual />
        <div className="visual-footer"><div><span className="mono">SOURCE</span><b>Notebook export</b></div><div><span className="mono">STATE</span><b className="mint-text">{status.toLowerCase()}</b></div><div><span className="mono">MODE</span><b>evaluation</b></div></div></div></div>
      <div className="container"><div className="metric-strip">{[{ label:'FEATURE GROUPS', value:schema?.stages.length ?? '—', icon:Layers3 },
        {label:'COMPARISON MODELS',value: schema ? new Set(rows.map(r => r.model)).size : '—',icon:Dna},
        {label:'EXPLANATIONS',value:'Individual SHAP',icon:Sparkles},{label:'COST TIERS',value:'Qualitative',icon:Gauge},
        {label:'VALIDATION',value:'Exploratory',icon:ShieldCheck}].map(({label,value,icon:Icon}) => <div className="metric-strip-item" key={label}><Icon size={18} /><div><div className="mono metric-label">{label}</div><div className="metric-value">{value}</div></div></div>)}</div></div>
    </section>
    {error ? <section id="screen" className="container connection-error" role="alert"><CircleAlert size={24} /><h2 className="display">Connect your notebook export.</h2><p>{error}</p><p>Start the Python model service with the exact artifacts folder, then retry. No scores are generated while disconnected.</p><button className="primary-button" onClick={load}><RefreshCw size={16} />Retry connection</button></section> :
      schema ? <><Screening key={schema.fingerprint} schema={schema} /><ModelLab schema={schema} rows={rows} /><CostAnalysis stages={schema.stages} /><InputReference schema={schema} /></> :
        <section className="container connection-error" role="status">Connecting to your local notebook export…</section>}
    </main><footer className="site-footer"><div className="container footer-inner"><div className="footer-brand"><LogoMark /><span>PCOS INSIGHT<br /><small>Explainable screening evaluation</small></span></div>
      <div className="footer-links"><button onClick={() => go('screen')}>Screen patient</button><button onClick={() => go('about')}>Input Reference</button></div><div className="mono footer-meta">Evaluation use only<br />Local model connection</div></div></footer>
  </div>;
}
