import type { MethodRecord } from '../types';

export function MethodCards({ methods }: { methods: MethodRecord[] }) {
  return (
    <div className="method-grid">
      {methods.map((method, index) => (
        <article className={`method-card method-${index + 1}`} key={method.id}>
          <div className="method-topline">
            <span className="method-index">0{index + 1}</span>
            <span className={`pill ${method.status}`}>{method.status}</span>
          </div>
          <h3>{method.name}</h3>
          <p className="method-model">{method.model}</p>
          <p>{method.description}</p>
        </article>
      ))}
    </div>
  );
}
