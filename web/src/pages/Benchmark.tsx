import type { BenchmarkData, MethodRecord } from '../types';
import { MethodCards } from '../components/MethodCards';

export function Benchmark({ data, methods }: { data: BenchmarkData; methods: MethodRecord[] }) {
  return (
    <div className="page-stack">
      <section className="stats-grid">
        <article className="stat-card"><small>Train</small><strong>{data.split.train}</strong><span>patients</span></article>
        <article className="stat-card"><small>Validation</small><strong>{data.split.validation}</strong><span>patients</span></article>
        <article className="stat-card"><small>Test</small><strong>{data.split.test}</strong><span>patients</span></article>
        <article className="stat-card"><small>Seeds</small><strong>{data.seeds.length}</strong><span>{data.seeds.join(' · ')}</span></article>
      </section>

      <section className="panel">
        <div className="section-heading"><div><p className="eyebrow">Frozen train pool</p><h2>Label budgets</h2></div></div>
        <div className="budget-grid">
          {data.budgets.map((budget) => (
            <article key={budget.id}>
              <strong>{budget.label}</strong>
              <span>{budget.patients} labelled patients</span>
              <small>remaining train patients become the unlabelled pool for SSL</small>
            </article>
          ))}
        </div>
      </section>

      <section>
        <div className="section-heading"><div><p className="eyebrow">Controlled comparison</p><h2>Arms</h2></div></div>
        <MethodCards methods={methods} />
      </section>

      <section className="panel protocol-grid">
        <div><p className="eyebrow">Primary deltas</p><h2>What we measure</h2></div>
        <code>ΔSSL = metric(MT) − metric(SUP)</code>
        <code>ΔVFM = metric(MT + MedSAM) − metric(MT)</code>
        <p>Same patient split, labelled subset, image representation and supervised schedule wherever technically possible.</p>
      </section>
    </div>
  );
}
