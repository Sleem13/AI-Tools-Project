import { NavLink } from "react-router-dom";
import { ScanBox } from "lucide-react";
import { navigationLinks } from "./navigation";

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 bottom-0 hidden w-60 bg-bg-secondary border-r border-border lg:flex flex-col z-50">
      <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
        <ScanBox className="w-7 h-7 text-accent" />
        <div>
          <h1 className="text-base font-semibold text-text-primary leading-tight">
            ALPR Pipeline
          </h1>
          <p className="text-xs text-text-muted">Egyptian License Plates</p>
        </div>
      </div>

      <nav className="flex-1 py-4 px-3 space-y-1">
        {navigationLinks.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-accent/15 text-accent"
                  : "text-text-secondary hover:text-text-primary hover:bg-bg-hover"
              }`
            }
          >
            <Icon className="w-4.5 h-4.5" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-5 py-4 border-t border-border">
        <div className="flex items-center gap-2 mb-2"><span className="w-1.5 h-1.5 rounded-full bg-success" /><p className="text-[11px] text-text-secondary">3-stage detection cascade</p></div>
        <p className="text-xs text-text-muted">v1.0.0 &middot; MIT</p>
      </div>
    </aside>
  );
}
