export function About() {
  return (
    <div className="page-stack">
      <section className="panel prose-panel">
        <p className="eyebrow">Scope</p>
        <h2>Independent research benchmark</h2>
        <p>
          This showcase presents an independent benchmark for pulmonary nodule segmentation under limited and
          incomplete annotations. It is not a clinical diagnostic product and is not intended to determine whether
          a patient has cancer.
        </p>
      </section>

      <div className="two-column">
        <section className="panel prose-panel">
          <p className="eyebrow">Dataset</p>
          <h2>LIDC-IDRI</h2>
          <p>
            The v1 task targets volumetrically annotated pulmonary nodules. Radiologist malignancy scores are used
            only as descriptive QC metadata and are not treated as pathology-confirmed cancer labels.
          </p>
        </section>
        <section className="panel prose-panel">
          <p className="eyebrow">Reproducibility</p>
          <h2>Frozen before comparison</h2>
          <p>
            Patient splits, label budgets, seeds, annotation policy and the primary comparison structure are
            versioned before the benchmark results are inspected.
          </p>
        </section>
      </div>
    </div>
  );
}
