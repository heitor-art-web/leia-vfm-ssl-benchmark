import type { BenchmarkData } from '../types';
import { ResultsTable } from '../components/ResultsTable';

export function Results({ data }: { data: BenchmarkData }) {
  return (
    <div className="page-stack">
      <section className="panel pending-banner">
        <div>
          <p className="eyebrow">Scientific status</p>
          <h2>No benchmark scores are published yet</h2>
          <p>
            The real-data YOLO26 smoke run proves integration only. Dice and IoU stay blank until the frozen full
            validation runs complete.
          </p>
        </div>
        <span className="pending-orbit">…</span>
      </section>

      <section className="panel">
        <div className="section-heading"><div><p className="eyebrow">Run matrix</p><h2>Results registry</h2></div></div>
        <ResultsTable rows={data.matrix} />
      </section>
    </div>
  );
}
