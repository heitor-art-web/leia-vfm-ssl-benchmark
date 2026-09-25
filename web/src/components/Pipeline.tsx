import type { PipelineNode } from '../types';

export function Pipeline({ nodes }: { nodes: PipelineNode[] }) {
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Research path</p>
          <h2>Pipeline</h2>
        </div>
      </div>
      <div className="pipeline">
        {nodes.map((node, index) => (
          <div className="pipeline-fragment" key={node.id}>
            <article className="pipeline-node">
              <span className="pipeline-icon">{index + 1}</span>
              <strong>{node.title}</strong>
              <small>{node.subtitle}</small>
            </article>
            {index < nodes.length - 1 && <span className="pipeline-arrow">→</span>}
          </div>
        ))}
      </div>
    </section>
  );
}
