import type { CaseRecord } from '../types';

interface CasesProps {
  cases: CaseRecord[];
  onOpen: (record: CaseRecord) => void;
}

export function Cases({ cases, onOpen }: CasesProps) {
  return (
    <div className="case-grid">
      {cases.map((record) => (
        <button className="case-card" key={record.caseId} onClick={() => onOpen(record)}>
          <div className="case-card-top">
            <span className={`case-role ${record.role}`}>{record.role}</span>
            <span>{record.annotationVotes} votes</span>
          </div>
          <h2>{record.patientId}</h2>
          <strong>{record.title}</strong>
          <p>{record.description}</p>
          <div className="case-card-metrics">
            <span><small>Diameter</small>{record.diameterMm === null ? '—' : `${record.diameterMm.toFixed(2)} mm`}</span>
            <span><small>Malignancy</small>{record.malignancyMedian ?? '—'}</span>
          </div>
          <span className="open-case">Open in viewer →</span>
        </button>
      ))}
    </div>
  );
}
