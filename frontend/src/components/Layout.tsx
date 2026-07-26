import { ScanBox } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import { navigationLinks } from "./navigation";

export default function Layout() {
  return (
    <div className="min-h-screen bg-bg-primary">
      <Sidebar />
      <header className="sticky top-0 z-50 border-b border-border bg-bg-secondary/95 backdrop-blur lg:hidden">
        <div className="flex items-center gap-2 px-4 py-3">
          <ScanBox className="h-6 w-6 shrink-0 text-accent" />
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-text-primary">
              ALPR Pipeline
            </p>
            <p className="text-[11px] text-text-muted">
              Three-stage Egyptian plate recognition
            </p>
          </div>
        </div>
        <nav
          aria-label="Mobile navigation"
          className="flex gap-1 overflow-x-auto px-2 pb-2"
        >
          {navigationLinks.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-accent/15 text-accent"
                    : "text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="p-4 sm:p-6 lg:ml-60 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}
