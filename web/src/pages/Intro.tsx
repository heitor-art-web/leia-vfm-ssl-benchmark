import type { PageKey } from '../types';
import './Intro.css';

export interface IntroData {
  official: {
    name: string;
    expanded: string;
    summary: string;
    goals: string[];
    sourceLabel: string;
    sourceUrl: string;
  };
  benchmark: {
    title: string;
    summary: string;
    mapping: Array<{ leia: string; benchmark: string }>;
  };
  contribution: {
    title: string;
    items: Array<{ title: string; text: string }>;
  };
  status: {
    complete: string[];
    next: string;
  };
  boundary: string;
}

interface IntroProps {
  data: IntroData;
  onNavigate: (page: PageKey) => void;
}

export function Intro({ data, onNavigate }: IntroProps) {
  return (
    <div className="page-stack intro-page">
      <section className="panel intro-hero">
        <div className="intro-hero-copy">
          <span className="hero-kicker">Research context</span>
          <p className="intro-acronym">{data.official.name}</p>
          <h2>{data.official.expanded}</h2>
          <p className="intro-lead">{data.official.summary}</p>
          <div className="hero-actions">
            <button className="primary" onClick={() => onNavigate('overview')}>Open benchmark showcase</button>
            <a className="secondary intro-link-button" href={data.official.sourceUrl} target="_blank" rel="noreferrer">
              {data.official.sourceLabel} ↗
            </a>
          </div>
        </div>

        <aside className="intro-goals" aria-label="Public LEIA goals">
          <p className="eyebrow">Public LEIA goals</p>
          <div className="intro-goal-list">
            {data.official.goals.map((goal, index) => (
              <div className="intro-goal" key={goal}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <p>{goal}</p>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className="panel intro-fit">
        <div className="intro-fit-heading">
          <div>
            <p className="eyebrow">How this work fits</p>
            <h2>{data.benchmark.title}</h2>
          </div>
          <p>{data.benchmark.summary}</p>
        </div>

        <div className="intro-map-grid">
          {data.benchmark.mapping.map((row) => (
            <article className="intro-map-card" key={row.leia}>
              <p className="intro-map-label">LEIA question</p>
              <h3>{row.leia}</h3>
              <div className="intro-map-arrow" aria-hidden="true">↓</div>
              <p className="intro-map-label">Benchmark instantiation</p>
              <p>{row.benchmark}</p>
            </article>
          ))}
        </div>
      </section>

      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Why this is useful</p>
            <h2>{data.contribution.title}</h2>
          </div>
        </div>
        <div className="intro-contribution-grid">
          {data.contribution.items.map((item, index) => (
            <article className="panel intro-contribution-card" key={item.title}>
              <span className="intro-card-index">{String(index + 1).padStart(2, '0')}</span>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <div className="two-column intro-status-grid">
        <section className="panel prose-panel">
          <p className="eyebrow">Evidence already established</p>
          <h2>Before spending compute</h2>
          <div className="intro-check-list">
            {data.status.complete.map((item) => (
              <div key={item}><span>✓</span><p>{item}</p></div>
            ))}
          </div>
        </section>

        <section className="panel intro-next">
          <p className="eyebrow">Next scientific milestone</p>
          <h2>{data.status.next}</h2>
          <p>
            That run establishes the first frozen supervised result. Mean Teacher and the MedSAM-assisted arm are
            interpreted only after the supervised reference is stable.
          </p>
          <button className="secondary" onClick={() => onNavigate('benchmark')}>Inspect frozen protocol</button>
        </section>
      </div>

      <section className="panel intro-boundary">
        <div className="intro-boundary-mark">i</div>
        <div>
          <p className="eyebrow">Project boundary</p>
          <h2>Independent benchmark, explicit scope</h2>
          <p>{data.boundary}</p>
        </div>
      </section>
    </div>
  );
}
