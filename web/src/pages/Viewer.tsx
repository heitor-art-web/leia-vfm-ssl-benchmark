import type { CaseRecord } from '../types';
import { ViewerPanel } from '../components/ViewerPanel';

interface ViewerProps {
  cases: CaseRecord[];
  selected: CaseRecord;
  onSelect: (record: CaseRecord) => void;
}

export function Viewer({ cases, selected, onSelect }: ViewerProps) {
  return (
    <div className="viewer-layout">
      <div className="viewer-case-list panel">
        <p className="eyebrow">QC cohort</p>
        <h2>Cases</h2>
        {cases.map((record) => (
          <button
            className={selected.caseId === record.caseId ? 'selected' : ''}
            key={record.caseId}
            onClick={() => onSelect(record)}
          >
            <span className={`legend-dot ${record.role === 'unknown' ? 'unknown' : record.role === 'trusted' ? 'trusted' : 'background'}`} />
            <span><strong>{record.patientId}</strong><small>{record.title}</small></span>
          </button>
        ))}
      </div>
      <ViewerPanel record={selected} />
    </div>
  );
}
