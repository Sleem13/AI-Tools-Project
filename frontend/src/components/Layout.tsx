import { useEffect, useState } from "react";
import { ScanBox } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import ThemeToggle, { type ThemeMode } from "./ThemeToggle";
import { navigationLinks } from "./navigation";

const THEME_STORAGE_KEY = "alpr-theme";

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "dark";
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export default function Layout() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  }

  return (
    <div className="min-h-screen bg-bg-primary">
      <Sidebar theme={theme} onToggleTheme={toggleTheme} />
      <header className="sticky top-0 z-50 border-b border-border bg-bg-secondary/95 backdrop-blur lg:hidden">
        <div className="flex items-center gap-2 px-4 py-3">
          <ScanBox className="h-6 w-6 shrink-0 text-accent" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-text-primary">
              ALPR Pipeline
            </p>
            <p className="text-[11px] text-text-muted">
              Three-stage Egyptian plate recognition
            </p>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
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
