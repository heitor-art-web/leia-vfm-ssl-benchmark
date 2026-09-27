interface HeaderProps {
  title: string;
  subtitle: string;
}

const REPO_URL = 'https://github.com/heitor-art-web/leia-vfm-ssl-benchmark';
const PROTOCOL_URL = `${REPO_URL}/blob/main/docs/07_lidc_experiment_protocol.md`;

export function Header({ title, subtitle }: HeaderProps) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">LIDC-IDRI · limited annotations · semantic segmentation</p>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="header-side">
        <div className="header-links" aria-label="Project resources">
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub ↗</a>
          <a href={PROTOCOL_URL} target="_blank" rel="noreferrer">Protocol ↗</a>
        </div>
        <div className="header-badge">
          <strong>v0.1</strong>
          <span>Phase 0 complete</span>
        </div>
      </div>
    </header>
  );
}
