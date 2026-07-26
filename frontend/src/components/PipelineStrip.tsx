import { ArrowRight, CarFront, CheckCircle2, CircleDashed, CreditCard, Type } from "lucide-react";

export type PipelineStageState = "ready" | "missing" | "checking" | "offline";

interface Props {
  vehicle: PipelineStageState;
  plate: PipelineStageState;
  character: PipelineStageState;
  compact?: boolean;
}

const STAGES = [
  { key: "vehicle", number: "01", label: "Vehicles", model: "YOLO11", icon: CarFront },
  { key: "plate", number: "02", label: "License plates", model: "YOLO11", icon: CreditCard },
  { key: "character", number: "03", label: "Characters", model: "YOLO26", icon: Type },
] as const;

const STATE_COPY: Record<PipelineStageState, { label: string; classes: string }> = {
  ready: { label: "Ready", classes: "text-success bg-success/10 border-success/20" },
  missing: { label: "Training required", classes: "text-warning bg-warning/10 border-warning/20" },
  checking: { label: "Checking", classes: "text-text-muted bg-bg-primary border-border" },
  offline: { label: "Backend offline", classes: "text-danger bg-danger/10 border-danger/20" },
};

export default function PipelineStrip({ vehicle, plate, character, compact = false }: Props) {
  const states = { vehicle, plate, character };

  return (
    <section className="bg-bg-card border border-border rounded-xl p-4 sm:p-5">
      <div className="flex items-center justify-between gap-4 mb-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-accent">Production inference</p>
          {!compact && <p className="text-sm text-text-secondary mt-1">Every accepted plate passes through three specialized detection stages.</p>}
        </div>
        <span className="hidden sm:inline-flex px-2.5 py-1 rounded-full bg-accent/10 text-accent text-xs font-medium">3-stage cascade</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto_1fr] gap-3 items-stretch">
        {STAGES.map((stage, index) => {
          const state = states[stage.key];
          const copy = STATE_COPY[state];
          const Icon = stage.icon;
          return (
            <div key={stage.key} className="contents">
              <article className={`rounded-lg border p-4 ${state === "ready" ? "border-success/25 bg-success/[0.03]" : "border-border bg-bg-primary/50"}`}>
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-bg-primary grid place-items-center text-accent"><Icon className="w-4.5 h-4.5" /></div>
                    <div><p className="text-[10px] tracking-[0.16em] text-text-muted">STAGE {stage.number}</p><h2 className="text-sm font-semibold text-text-primary mt-0.5">{stage.label}</h2></div>
                  </div>
                  {state === "ready" ? <CheckCircle2 className="w-4 h-4 text-success mt-1" /> : <CircleDashed className="w-4 h-4 text-text-muted mt-1" />}
                </div>
                <div className="flex items-center justify-between gap-3 mt-4">
                  <span className="text-xs font-mono text-text-secondary">{stage.model}</span>
                  <span className={`text-[10px] px-2 py-1 rounded-full border ${copy.classes}`}>{copy.label}</span>
                </div>
              </article>
              {index < STAGES.length - 1 && <div className="hidden md:grid place-items-center text-text-muted"><ArrowRight className="w-4 h-4" /></div>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
