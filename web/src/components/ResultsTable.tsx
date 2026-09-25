import type { BenchmarkData } from '../types';

export function ResultsTable({ rows }: { rows: BenchmarkData['matrix'] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Budget</th>
            <th>Status</th>
            <th>Dice</th>
            <th>IoU</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.method}-${row.budget}`}>
              <td>{row.method}</td>
              <td>{row.budget}</td>
              <td><span className={`pill ${row.status}`}>{row.status}</span></td>
              <td>{row.dice ?? '—'}</td>
              <td>{row.iou ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
