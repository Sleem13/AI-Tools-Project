import { useEffect, useState } from "react";
import {
  BrainCircuit,
  Target,
  Type,
  CheckCircle2,
  Clock,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import StatsGrid from "../components/StatsGrid";
import LoadingSpinner from "../components/LoadingSpinner";
import { getTrainingStatus } from "../api/client";

const STATUS_ICON: Record<string, any> = {
  idle: Clock,
  training: Loader2,
  completed: CheckCircle2,
  error: AlertTriangle,
};

const STATUS_COLOR: Record<string, string> = {
  idle: "text-text-muted",
  training: "text-info",
  completed: "text-success",
  error: "text-danger",
};

export default function Training() {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTrainingStatus()
      .then((res) => setModels(res.training || []))
      .catch(() =>
        setModels([
          {
            model: "YOLOv8 Detection",
            status: "idle",
            epoch: 0,
            total_epochs: 100,
            loss: 0,
            metrics: {},
          },
          {
            model: "CRNN OCR",
            status: "idle",
            epoch: 0,
            total_epochs: 50,
            loss: 0,
            metrics: {},
          },
        ])
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading training status..." />;

  return (
    <div className="max-w-6xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3">
          <BrainCircuit className="w-8 h-8 text-accent" />
          Model Training
        </h1>
        <p className="text-text-secondary mt-1">
          Monitor and launch model training jobs.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatsGrid
          icon={<Target className="w-4 h-4" />}
          label="Detection Model"
          value="YOLOv8"
          sub="License plate detection"
          color="text-accent"
        />
        <StatsGrid
          icon={<Type className="w-4 h-4" />}
          label="OCR Model"
          value="CRNN + CTC"
          sub="Text recognition"
          color="text-info"
        />
        <StatsGrid
          icon={<BrainCircuit className="w-4 h-4" />}
          label="Models Trained"
          value={
            models.filter((m: any) => m.status === "completed").length +
            " / " +
            models.length
          }
          color="text-success"
        />
      </div>

      <div className="space-y-4">
        {models.map((m: any, i: number) => {
          const Icon = STATUS_ICON[m.status] || Clock;
          const progress =
            m.total_epochs > 0 ? (m.epoch / m.total_epochs) * 100 : 0;

          return (
            <div
              key={i}
              className="bg-bg-card border border-border rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-bg-primary ${STATUS_COLOR[m.status]}`}>
                    <Icon
                      className={`w-5 h-5 ${m.status === "training" ? "animate-spin" : ""}`}
                    />
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-text-primary">
                      {m.model}
                    </h3>
                    <p className="text-xs text-text-muted capitalize">
                      Status: {m.status}
                    </p>
                  </div>
                </div>
                <span className="text-sm font-mono text-text-secondary">
                  Epoch {m.epoch}/{m.total_epochs}
                </span>
              </div>

              <div className="w-full h-2 rounded-full bg-bg-primary overflow-hidden mb-4">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    m.status === "completed"
                      ? "bg-success"
                      : m.status === "training"
                        ? "bg-info"
                        : m.status === "error"
                          ? "bg-danger"
                          : "bg-text-muted"
                  }`}
                  style={{ width: `${progress}%` }}
                />
              </div>

              {Object.keys(m.metrics || {}).length > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {Object.entries(m.metrics).map(([k, v]) => (
                    <div key={k} className="bg-bg-primary rounded-lg px-3 py-2">
                      <p className="text-xs text-text-muted">{k}</p>
                      <p className="text-sm font-medium text-text-primary font-mono">
                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {m.loss > 0 && (
                <div className="mt-3 flex items-center gap-2 text-sm">
                  <span className="text-text-muted">Loss:</span>
                  <span className="font-mono text-text-primary">
                    {m.loss.toFixed(6)}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {models.length === 0 && (
        <div className="text-center py-16 text-text-muted">
          <BrainCircuit className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>No training jobs found.</p>
        </div>
      )}
    </div>
  );
}
