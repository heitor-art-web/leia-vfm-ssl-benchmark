interface HeaderProps {
  title: string;
  subtitle: string;
}

export function Header({ title, subtitle }: HeaderProps) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">LIDC-IDRI · limited annotations · semantic segmentation</p>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="header-badge">
        <strong>v0.1</strong>
        <span>Phase 0 / showcase</span>
      </div>
    </header>
  );
}
