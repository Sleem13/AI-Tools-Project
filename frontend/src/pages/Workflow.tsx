import { useEffect, useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Database,
  FileVideo,
  Flag,
  GitBranch,
  Image as ImageIcon,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Target,
  Type,
} from "lucide-react";
import { getHealth, getPipelineStatus, getTrainingWorkbench } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import type { TrainingWorkbench } from "../types";

type NodeState = "complete" | "ready" | "blocked" | "future";

const STATE_STYLE: Record<NodeState, { border: string; badge: string; label: string }> = {
  complete: { border: "border-success/50", badge: "bg-success/10 text-success", label: "Complete" },
  ready: { border: "border-info/50", badge: "bg-info/10 text-info", label: "Ready" },
  blocked: { border: "border-warning/50", badge: "bg-warning/10 text-warning", label: "Waiting" },
  future: { border: "border-accent/40", badge: "bg-accent/10 text-accent", label: "Planned" },
};

function WorkflowNode({ icon: Icon, title, description, state, detail }: { icon: typeof Database; title: string; description: string; state: NodeState; detail: string }) {
  const style = STATE_STYLE[state];
  return (
    <article className={`relative bg-bg-card border ${style.border} rounded-xl p-5 min-h-48 flex flex-col`}>
      <div className="flex items-start justify-between gap-3"><div className="p-2.5 rounded-lg bg-bg-primary text-accent"><Icon className="w-5 h-5" /></div><span className={`text-[11px] font-medium px-2.5 py-1 rounded-full ${style.badge}`}>{style.label}</span></div>
      <h3 className="text-base font-semibold text-text-primary mt-5">{title}</h3>
      <p className="text-sm text-text-secondary mt-1 flex-1">{description}</p>
      <p className="text-xs font-mono text-text-muted mt-4">{detail}</p>
    </article>
  );
}

function Connector() {
  return <div className="hidden 2xl:grid place-items-center text-text-muted"><ArrowRight className="w-5 h-5" /></div>;
}

export default function Workflow() {
  const [workbench, setWorkbench] = useState<TrainingWorkbench | null>(null);
  const [pipeline, setPipeline] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getTrainingWorkbench(), getPipelineStatus(), getHealth()])
      .then(([training, pipelineStatus, healthStatus]) => {
        setWorkbench(training);
        setPipeline(pipelineStatus);
        setHealth(healthStatus);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading || !workbench) return <LoadingSpinner text="Building workflow state..." />;

  const completedPipelineStages = pipeline?.stages?.filter((stage: any) => stage.status === "completed").length || 0;
  const totalPipelineStages = pipeline?.stages?.length || 7;
  const plateModelReady = workbench.run.best_model_exists;
  const detectorReady = Boolean(health?.models_loaded?.detection);
  const trainingProgress = workbench.job.status === "training"
    ? `${workbench.job.metrics.latest?.epoch || 0}/${workbench.job.epochs} epochs`
    : plateModelReady ? "best.pt available" : "training required";

  return (
    <div className="max-w-7xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3"><GitBranch className="w-8 h-8 text-accent" /> Complete ALPR workflow</h1>
        <p className="text-text-secondary mt-1">A live system map from raw data and model training to real media inference and human feedback.</p>
      </div>

      <section className="bg-gradient-to-br from-accent/15 via-bg-card to-info/10 border border-accent/30 rounded-2xl p-6 lg:p-8 overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5">
          <div><p className="text-xs uppercase tracking-[0.2em] text-accent">System artwork</p><h2 className="text-2xl font-semibold text-text-primary mt-2">Learn from data. See vehicles. Find plates. Verify with people.</h2><p className="text-sm text-text-secondary mt-2 max-w-3xl">Training and inference are joined by one feedback loop: every human decision can become a reviewed example for the next dataset version.</p></div>
          <div className="grid grid-cols-3 gap-3 shrink-0"><div className="bg-bg-primary/70 rounded-lg p-3 text-center"><p className="text-xl font-mono text-text-primary">{completedPipelineStages}/{totalPipelineStages}</p><p className="text-[11px] text-text-muted">Data stages</p></div><div className="bg-bg-primary/70 rounded-lg p-3 text-center"><p className="text-xl font-mono text-text-primary">{workbench.dataset.total_images}</p><p className="text-[11px] text-text-muted">Master images</p></div><div className="bg-bg-primary/70 rounded-lg p-3 text-center"><p className="text-xl font-mono text-text-primary">{plateModelReady ? "1" : "0"}</p><p className="text-[11px] text-text-muted">Plate model</p></div></div>
        </div>
      </section>

      <div>
        <div className="flex items-center gap-3 mb-4"><Database className="w-5 h-5 text-accent" /><div><h2 className="text-lg font-semibold text-text-primary">Learning lane</h2><p className="text-xs text-text-muted">Build trustworthy model assets from inspected data.</p></div></div>
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] gap-3 items-stretch">
          <WorkflowNode icon={Database} title="Raw datasets" description="Ingest Egyptian vehicle and plate imagery with annotations from multiple sources." state={pipeline?.total_images > 0 ? "complete" : "blocked"} detail={`${pipeline?.total_images || 0} raw images discovered`} />
          <Connector />
          <WorkflowNode icon={ShieldCheck} title="Inspect & harmonize" description="Validate images, labels, duplicates, geometry, quality, and normalized YOLO structure." state={completedPipelineStages >= 4 ? "complete" : "blocked"} detail={`${completedPipelineStages}/${totalPipelineStages} engineering stages complete`} />
          <Connector />
          <WorkflowNode icon={Target} title="Master Plate dataset" description="Preview train, validation, and test splits before committing GPU time." state={workbench.dataset.ready ? "complete" : "blocked"} detail={`${workbench.dataset.splits.train.images} train · ${workbench.dataset.splits.val.images} val`} />
          <Connector />
          <WorkflowNode icon={BrainCircuit} title="YOLO11 plate training" description="Fine-tune, monitor epochs, inspect curves and confusion matrices, and retain best.pt." state={workbench.job.status === "training" ? "ready" : plateModelReady ? "complete" : "blocked"} detail={trainingProgress} />
        </div>
      </div>

      <div className="flex justify-center text-text-muted"><ArrowDown className="w-6 h-6" /></div>

      <div>
        <div className="flex items-center gap-3 mb-4"><ScanLine className="w-5 h-5 text-info" /><div><h2 className="text-lg font-semibold text-text-primary">Inference & interaction lane</h2><p className="text-xs text-text-muted">Turn real pictures and videos into reviewable evidence.</p></div></div>
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] gap-3 items-stretch">
          <WorkflowNode icon={FileVideo} title="Image or video" description="A human supplies real traffic media and chooses confidence and frame stride." state="ready" detail="JPG · PNG · MP4 · MOV · AVI · MKV · WebM" />
          <Connector />
          <WorkflowNode icon={ImageIcon} title="YOLO11 vehicles" description="Stage one finds cars, motorcycles, buses, and trucks before plate analysis." state={detectorReady ? "complete" : "blocked"} detail="COCO classes 2 · 3 · 5 · 7" />
          <Connector />
          <WorkflowNode icon={Target} title="YOLO11 plates" description="Stage two detects plates inside vehicle crops and maps boxes back to the full frame." state={plateModelReady ? "complete" : "blocked"} detail={plateModelReady ? "latest best.pt selected" : "awaiting trained checkpoint"} />
          <Connector />
          <WorkflowNode icon={Type} title="YOLO26 characters" description="Future stage detects ordered plate characters from enhanced high-resolution crops." state="future" detail="crop contract already available" />
          <Connector />
          <WorkflowNode icon={Flag} title="Human verification" description="Accept, reject, or flag each result and export an auditable review record." state="ready" detail="review JSON closes the loop" />
        </div>
      </div>

      <section className="bg-bg-card border border-border rounded-xl p-5 flex flex-col md:flex-row items-start md:items-center gap-4">
        <div className="p-3 rounded-full bg-success/10 text-success"><Sparkles className="w-5 h-5" /></div>
        <div className="flex-1"><h2 className="text-base font-semibold text-text-primary">Feedback becomes the next training signal</h2><p className="text-sm text-text-secondary mt-1">Exported human decisions identify false positives, missed plates, and OCR errors for annotation and the next Master Plate release.</p></div>
        <div className="inline-flex items-center gap-2 text-sm text-success"><CheckCircle2 className="w-4 h-4" /> Human in the loop</div>
      </section>
    </div>
  );
}
