import type { NavigationItem, PageKey } from '../types';

interface SidebarProps {
  items: NavigationItem[];
  active: PageKey;
  onNavigate: (page: PageKey) => void;
}

export function Sidebar({ items, active, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <button className="brand" onClick={() => onNavigate('overview')} aria-label="Open overview">
        <span className="brand-mark">L</span>
        <span>
          <strong>LEIA</strong>
          <small>research showcase</small>
        </span>
      </button>

      <nav className="nav-list" aria-label="Showcase navigation">
        {items.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${active === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="nav-glyph">{item.glyph}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-note">
        <span className="status-dot" />
        <span>Research preview</span>
        <small>Not a medical device</small>
      </div>
    </aside>
  );
}
