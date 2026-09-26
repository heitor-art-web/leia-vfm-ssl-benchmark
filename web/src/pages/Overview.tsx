import type { AnnotationPolicyRow, MethodRecord, PipelineNode, PageKey } from '../types';
import { AnnotationPolicy } from '../components/AnnotationPolicy';
import { MethodCards } from '../components/MethodCards';
import { Pipeline } from '../components/Pipeline';

interface OverviewProps {
  methods: MethodRecord[];
  policy: AnnotationPolicyRow[];
  pipeline: PipelineNode[];
  onNavigate: (page: PageKey) => void;
}

export function Overview({ methods, policy, pipeline, onNavigate }: OverviewProps) {
  return (
    <div className="page-stack">
      <section className="hero panel">
        <div className="hero-copy">
          <span className="hero-kicker">Research question</span>
          <h2>Can semi-supervised learning still add value when a medical vision foundation model is available?</h2>
          <p>
            We compare a supervised YOLO26 semantic specialist, Mean Teacher, and a MedSAM-assisted co-teacher
            under frozen 1%, 5%, 10% and 25% labelled-patient budgets.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={() => onNavigate('viewer')}>Explore real QC cases</button>
            <button className="secondary" onClick={() => onNavigate('benchmark')}>View protocol</button>
          </div>
        </div>
        <div className="hero-status">
          <p className="eyebrow">Current milestone</p>
          <strong>Phase 0 data pipeline</strong>
          <span>Frozen splits · annotation-vote targets · real-data smoke test</span>
          <div className="progress-track"><div className="progress-fill" /></div>
          <small>Next: full supervised 1% baseline</small>
        </div>
      </section>

      <section>
        <div className="section-heading">
          <div><p className="eyebrow">Comparison arms</p><h2>Methods</h2></div>
        </div>
        <MethodCards methods={methods} />
      </section>

      <div className="two-column">
        <AnnotationPolicy rows={policy} />
        <section className="panel research-note">
          <p className="eyebrow">Key design choice</p>
          <h2>UNKNOWN is not background</h2>
          <p>
            Voxels covered by only one or two volumetric contour annotations are excluded from the trusted loss
            target instead of being silently treated as negative tissue.
          </p>
          <div className="color-key"><span className="trusted-swatch" /> trusted</div>
          <div className="color-key"><span className="unknown-swatch" /> unknown / ignore</div>
        </section>
      </div>

      <Pipeline nodes={pipeline} />
    </div>
  );
}
