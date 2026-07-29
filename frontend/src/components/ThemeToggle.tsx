import { Moon, Sun } from "lucide-react";

export type ThemeMode = "dark" | "light";

interface Props {
  theme: ThemeMode;
  onToggle: () => void;
}

export default function ThemeToggle({ theme, onToggle }: Props) {
  const isLight = theme === "light";

  return (
    <button
      type="button"
      aria-label={`Switch to ${isLight ? "dark" : "light"} theme`}
      title={`Switch to ${isLight ? "dark" : "light"} theme`}
      onClick={onToggle}
      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border bg-bg-primary text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
    >
      {isLight ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </button>
  );
}
