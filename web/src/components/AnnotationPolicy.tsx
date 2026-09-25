import type { AnnotationPolicyRow } from '../types';

export function AnnotationPolicy({ rows }: { rows: AnnotationPolicyRow[] }) {
  return (
    <section className="panel annotation-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Target encoding</p>
          <h2>Annotation policy</h2>
        </div>
      </div>
      <div className="policy-list">
        {rows.map((row) => (
          <div className="policy-row" key={row.votes}>
            <span className={`legend-dot ${row.tone}`} />
            <strong>{row.votes} votes</strong>
            <span className="policy-arrow">→</span>
            <span>{row.label}</span>
            <code>{row.value}</code>
          </div>
        ))}
      </div>
    </section>
  );
}
