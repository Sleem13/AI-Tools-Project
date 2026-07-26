import {
  BrainCircuit,
  Database,
  GitBranch,
  LayoutDashboard,
  Settings,
} from "lucide-react";

export const navigationLinks = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/training", label: "Training", icon: BrainCircuit },
  { to: "/workflow", label: "Workflow", icon: GitBranch },
  { to: "/settings", label: "Settings", icon: Settings },
];
