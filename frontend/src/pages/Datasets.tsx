import { useEffect, useState } from "react";
import {
  Database,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Play,
  FileImage,
  FileText,
} from "lucide-react";
import StatsGrid from "../components/StatsGrid";
import LoadingSpinner from "../components/LoadingSpinner";
import { getDatasets, getPipelineStatus } from "../api/client";

const STAGE_COLORS: Record<string, string> = {
  completed: "text-success",
  pending: "text-text-muted",
  running: "text-info",
  error: "text-danger",
};

export default function Datasets() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [pipeline, setPipeline] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getDatasets().catch(() => ({ datasets: [] })), getPipelineStatus().catch(() => ({ stages: [], total_images: 0, datasets: [] }))])
      .then(([d, p]) => {
        setDatasets(d.datasets);
        setPipeline(p);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading datasets..." />;

  const totalImages = datasets.reduce((s: number, d: any) => s + (d.image_count || 0), 0);
  const totalAnnotations = datasets.reduce((s: number, d: any) => s + (d.annotation_count || 0), 0);
  const totalIssues = datasets.reduce((s: number, d: any) => s + (d.issues || 0), 0);

  return (
    <div className="max-w-6xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3">
          <Database className="w-8 h-8 text-accent" />
          Datasets
        </h1>
        <p className="text-text-secondary mt-1">
          Browse datasets and monitor the 7-stage pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatsGrid
          icon={<FileImage className="w-4 h-4" />}
          label="Total Images"
          value={totalImages}
          sub={`${datasets.length} dataset(s)`}
          color="text-accent"
        />
        <StatsGrid
          icon={<FileText className="w-4 h-4" />}
          label="Annotations"
          value={totalAnnotations}
          color="text-info"
        />
        <StatsGrid
          icon={<AlertTriangle className="w-4 h-4" />}
          label="Issues Found"
          value={totalIssues}
          sub={totalIssues === 0 ? "All clean" : "Review recommended"}
          color={totalIssues > 0 ? "text-warning" : "text-success"}
        />
      </div>

      {datasets.length > 0 && (
        <div className="bg-bg-card border border-border rounded-xl overflow-hidden">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border bg-bg-secondary/50">
                <th className="px-5 py-3 text-sm font-medium text-text-secondary">Name</th>
                <th className="px-5 py-3 text-sm font-medium text-text-secondary">Format</th>
                <th className="px-5 py-3 text-sm font-medium text-text-secondary text-right">Images</th>
                <th className="px-5 py-3 text-sm font-medium text-text-secondary text-right">Annotations</th>
                <th className="px-5 py-3 text-sm font-medium text-text-secondary text-right">Issues</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d: any, i: number) => (
                <tr key={i} className="border-b border-border last:border-0 hover:bg-bg-hover/50 transition-colors">
                  <td className="px-5 py-3 text-sm font-medium text-text-primary">{d.name}</td>
                  <td className="px-5 py-3">
                    <span className="text-xs px-2 py-0.5 rounded-md bg-accent/10 text-accent font-mono">
                      {d.format}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-sm text-text-secondary text-right">{d.image_count}</td>
                  <td className="px-5 py-3 text-sm text-text-secondary text-right">{d.annotation_count}</td>
                  <td className="px-5 py-3 text-sm text-right">
                    <span className={d.issues > 0 ? "text-warning" : "text-success"}>
                      {d.issues}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {pipeline?.stages?.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-text-primary mb-4">Pipeline Stages</h2>
          <div className="space-y-2">
            {pipeline.stages.map((s: any, i: number) => (
              <div
                key={i}
                className={`flex items-center gap-4 px-5 py-3 rounded-xl border border-border bg-bg-card`}
              >
                <span className="text-xs font-mono text-text-muted w-6">{i + 1}.</span>
                <div className="flex-1">
                  <p className="text-sm font-medium text-text-primary">{s.name}</p>
                  <p className="text-xs text-text-muted">{s.script}</p>
                </div>
                <div className="flex items-center gap-2">
                  {s.status === "completed" ? (
                    <CheckCircle2 className="w-4 h-4 text-success" />
                  ) : s.status === "running" ? (
                    <Play className="w-4 h-4 text-info animate-pulse" />
                  ) : s.status === "error" ? (
                    <AlertTriangle className="w-4 h-4 text-danger" />
                  ) : (
                    <Clock className="w-4 h-4 text-text-muted" />
                  )}
                  <span className={`text-xs font-medium ${STAGE_COLORS[s.status]}`}>
                    {s.status}
                  </span>
                </div>
                {s.last_run && (
                  <span className="text-xs text-text-muted hidden sm:block">
                    {new Date(s.last_run).toLocaleDateString()}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {datasets.length === 0 && (
        <div className="text-center py-16 text-text-muted">
          <Database className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>No datasets found. Run the inspection pipeline first.</p>
        </div>
      )}
    </div>
  );
}
