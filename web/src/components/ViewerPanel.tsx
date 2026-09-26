import { useState } from 'react';
import type { CaseRecord } from '../types';

interface ViewerPanelProps {
  record: CaseRecord;
}

export function ViewerPanel({ record }: ViewerPanelProps) {
  const [failed, setFailed] = useState(false);

  return (
    <section className="viewer-shell">
      <div className="viewer-toolbar">
        <div>
          <span className={`case-role ${record.role}`}>{record.role}</span>
          <strong>{record.patientId}</strong>
          <small>{record.caseId}</small>
        </div>
        <div className="viewer-meta">
          <span>{record.annotationVotes} annotation vote{record.annotationVotes === 1 ? '' : 's'}</span>
          {record.diameterMm !== null && <span>{record.diameterMm.toFixed(2)} mm</span>}
        </div>
      </div>

      <div className="viewer-canvas">
        {!failed ? (
          <img
            key={record.caseId}
            src={record.assets.contactSheet}
            alt={`QC contact sheet for ${record.patientId}`}
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="asset-placeholder">
            <div className="lung-symbol">◖ ◗</div>
            <strong>QC preview asset could not be loaded</strong>
            <p>
              The repository normally bundles this validated contact sheet. Re-run the QC asset sync workflow if the
              static file is missing or was removed from a deployment.
            </p>
            <code>{record.assets.contactSheet}</code>
          </div>
        )}
      </div>

      <div className="viewer-footer">
        <span><i className="legend-dot trusted" /> green = trusted foreground</span>
        <span><i className="legend-dot unknown" /> magenta = UNKNOWN / ignore</span>
        <span>QC images are evidence, not model predictions.</span>
      </div>
    </section>
  );
}
