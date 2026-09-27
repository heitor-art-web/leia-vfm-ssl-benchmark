import { useEffect, useState } from 'react';
import type { CaseRecord } from '../types';

interface ViewerPanelProps {
  record: CaseRecord;
}

type ViewerMode = 'ct' | 'mask' | 'overlay' | 'crop';

const VIEWER_MODES: Array<{ id: ViewerMode; label: string; description: string }> = [
  { id: 'ct', label: 'Original', description: 'CT lung-window evidence slice' },
  { id: 'mask', label: 'Target', description: 'Trusted / UNKNOWN semantic target preview' },
  { id: 'overlay', label: 'Overlay', description: 'Target overlaid on the CT evidence slice' },
  { id: 'crop', label: 'Crop', description: 'Focused evidence region for visual QC' },
];

export function ViewerPanel({ record }: ViewerPanelProps) {
  const [failed, setFailed] = useState(false);
  const [mode, setMode] = useState<ViewerMode>('overlay');

  useEffect(() => {
    setFailed(false);
    setMode('overlay');
  }, [record.caseId]);

  const activeMode = VIEWER_MODES.find((item) => item.id === mode) ?? VIEWER_MODES[2];

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
          {record.sliceIndex !== null && <span>slice {record.sliceIndex}</span>}
          {record.diameterMm !== null && <span>{record.diameterMm.toFixed(2)} mm</span>}
        </div>
      </div>

      <div className="viewer-mode-bar" role="tablist" aria-label="QC evidence layer">
        {VIEWER_MODES.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={mode === item.id}
            className={mode === item.id ? 'active' : ''}
            onClick={() => setMode(item.id)}
          >
            {item.label}
          </button>
        ))}
        <span>{activeMode.description}</span>
      </div>

      <div className="viewer-canvas viewer-canvas-interactive">
        {!failed ? (
          <div className={`contact-quadrant contact-quadrant-${mode}`}>
            <img
              key={`${record.caseId}-${mode}`}
              src={record.assets.contactSheet}
              alt={`${activeMode.label} QC evidence for ${record.patientId}`}
              onError={() => setFailed(true)}
            />
          </div>
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

      <div className="viewer-evidence-note">
        <strong>Phase 0 evidence viewer.</strong>
        <span>
          This switches between validated layers from one selected QC evidence slice; it is not yet a full-volume CT
          browser and it does not display model predictions.
        </span>
      </div>

      <div className="viewer-footer">
        <span><i className="legend-dot trusted" /> green = trusted foreground</span>
        <span><i className="legend-dot unknown" /> magenta = UNKNOWN / ignore</span>
        <span>QC images are benchmark-construction evidence.</span>
      </div>
    </section>
  );
}
