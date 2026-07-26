import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  FileImage,
  Gauge,
  ImageIcon,
  Loader2,
  Play,
  RefreshCw,
  ScanSearch,
  Sparkles,
  SquareTerminal,
  Type,
  XCircle,
} from "lucide-react";
import {
  evaluateValidationSamples,
  getDatasetSamples,
  getTrainingWorkbench,
  startPlateTraining,
} from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import PipelineStrip from "../components/PipelineStrip";
import StatsGrid from "../components/StatsGrid";
import type {
  DatasetSample,
  EvaluationSample,
  TrainingMetrics,
  TrainingWorkbench,
} from "../types";

type View = "overview" | "dataset" | "train" | "characters" | "evaluate";
type Split = "train" | "val" | "test";

const VIEWS: Array<{ id: View; label: string; icon: typeof BrainCircuit }> = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "train", label: "Train & results", icon: BrainCircuit },
  { id: "characters", label: "Characters", icon: Type },
  { id: "evaluate", label: "Evaluate", icon: ScanSearch },
];

function percent(value?: number | null) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function ReadinessRow({ ready, label, value }: { ready: boolean; label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 py-3 border-b border-border last:border-0">
      <div className="flex items-center gap-3 min-w-0">
        {ready ? <CheckCircle2 className="w-4 h-4 text-success shrink-0" /> : <XCircle className="w-4 h-4 text-danger shrink-0" />}
        <span className="text-sm text-text-secondary">{label}</span>
      </div>
      <span className="text-sm font-mono text-text-primary truncate text-right">{value}</span>
    </div>
  );
}

function MetricTiles({ metrics }: { metrics: TrainingMetrics | null }) {
  const items = [
    ["Precision", percent(metrics?.precision)],
    ["Recall", percent(metrics?.recall)],
    ["mAP50", percent(metrics?.map50)],
    ["mAP50–95", percent(metrics?.map50_95)],
  ];
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {items.map(([label, value]) => (
        <div key={label} className="bg-bg-primary border border-border rounded-lg px-4 py-3">
          <p className="text-xs text-text-muted">{label}</p>
          <p className="text-lg font-semibold font-mono text-text-primary mt-1">{value}</p>
        </div>
      ))}
    </div>
  );
}

export default function Training() {
  const [view, setView] = useState<View>("overview");
  const [workbench, setWorkbench] = useState<TrainingWorkbench | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [sampleSplit, setSampleSplit] = useState<Split>("train");
  const [sampleSeed, setSampleSeed] = useState(42);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({ epochs: 50, imgsz: 640, batch: 16, device: "0" });
  const [characterConfig, setCharacterConfig] = useState({ epochs: 100, imgsz: 640, batch: 32, device: "0" });
  const [starting, setStarting] = useState(false);
  const [evaluation, setEvaluation] = useState<EvaluationSample[]>([]);
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationConfig, setEvaluationConfig] = useState({ count: 3, confidence: 0.4, scale_factor: 4, seed: 42 });

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const data = await getTrainingWorkbench();
      setWorkbench(data);
      setTrainingConfig((current) => (workbench ? current : data.defaults));
      setCharacterConfig((current) => (workbench ? current : data.character_defaults));
      setError(null);
    } catch (caught: any) {
      setError(caught.message || "Could not load the training workbench");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [workbench]);

  useEffect(() => {
    void refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (workbench?.job.status !== "training") return;
    const timer = window.setInterval(() => void refresh(true), 3000);
    return () => window.clearInterval(timer);
  }, [workbench?.job.status, refresh]);

  const loadSamples = useCallback(async (split: Split, seed: number) => {
    setSamplesLoading(true);
    try {
      const response = await getDatasetSamples(split, 4, seed);
      setSamples(response.images);
      setError(null);
    } catch (caught: any) {
      setSamples([]);
      setError(caught.message || "Could not load dataset samples");
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (view === "dataset" && workbench?.dataset.ready) void loadSamples(sampleSplit, sampleSeed);
  }, [view, sampleSplit, sampleSeed, workbench?.dataset.ready, loadSamples]);

  const latestMetrics = workbench?.job.metrics.latest || workbench?.run.latest || null;
  const epoch = latestMetrics?.epoch || 0;
  const totalEpochs = workbench?.job.epochs || trainingConfig.epochs;
  const progress = totalEpochs > 0 ? Math.min(100, (epoch / totalEpochs) * 100) : 0;
  const referenceMetrics = useMemo<TrainingMetrics | null>(() => {
    const metrics = workbench?.reference_run?.metrics;
    return metrics ? { precision: metrics.precision, recall: metrics.recall, map50: metrics.map50, map50_95: metrics.map50_95 } : null;
  }, [workbench?.reference_run]);

  async function launchTraining(stage: "plate" | "character" = "plate") {
    setStarting(true);
    try {
      const config = stage === "character" ? characterConfig : trainingConfig;
      await startPlateTraining({ ...config, stage });
      setError(null);
      await refresh(true);
    } catch (caught: any) {
      setError(caught.message || "Could not start training");
    } finally {
      setStarting(false);
    }
  }

  async function runEvaluation() {
    setEvaluating(true);
    setEvaluation([]);
    try {
      const result = await evaluateValidationSamples(evaluationConfig);
      setEvaluation(result.samples);
      setError(null);
    } catch (caught: any) {
      setError(caught.message || "Could not evaluate validation samples");
    } finally {
      setEvaluating(false);
    }
  }

  if (loading && !workbench) return <LoadingSpinner text="Loading Master Plate workbench..." />;

  return (
    <div className="max-w-7xl space-y-7">
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3">
            <BrainCircuit className="w-8 h-8 text-accent" />
            Three-stage ALPR training
          </h1>
          <p className="text-text-secondary mt-1">Prepare YOLO11 plate detection and YOLO26 character recognition for the production cascade.</p>
        </div>
        <button onClick={() => void refresh()} className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-border bg-bg-card text-sm text-text-secondary hover:text-text-primary hover:bg-bg-hover transition-colors">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatsGrid icon={<Cpu className="w-4 h-4" />} label="GPU runtime" value={workbench?.runtime.cuda_available ? "CUDA ready" : "CPU only"} sub={workbench?.runtime.gpu_name || `PyTorch ${workbench?.runtime.torch || "unknown"}`} color={workbench?.runtime.ready ? "text-success" : "text-warning"} />
        <StatsGrid icon={<Database className="w-4 h-4" />} label="Plate images" value={workbench?.dataset.splits.train.images || 0} sub={`${workbench?.dataset.splits.val.images || 0} validation images`} color="text-accent" />
        <StatsGrid icon={<BrainCircuit className="w-4 h-4" />} label="Plate model" value={workbench?.run.best_model_exists ? "Trained" : "Not trained"} sub={workbench?.job.status === "training" ? `Epoch ${epoch}/${totalEpochs}` : workbench?.job.status || "idle"} color={workbench?.run.best_model_exists ? "text-success" : "text-text-muted"} />
        <StatsGrid icon={<Type className="w-4 h-4" />} label="Character model" value={workbench?.character_run.best_model_exists ? "Trained" : "Not trained"} sub={`${workbench?.character_dataset.total_images || 0} segmented plates · 38 classes`} color={workbench?.character_run.best_model_exists ? "text-success" : "text-warning"} />
      </div>

      {workbench && (
        <PipelineStrip
          compact
          vehicle="ready"
          plate={workbench.run.best_model_exists ? "ready" : "missing"}
          character={workbench.character_run.best_model_exists ? "ready" : "missing"}
        />
      )}

      <div className="flex flex-wrap gap-2 p-1.5 bg-bg-secondary border border-border rounded-xl w-fit">
        {VIEWS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setView(id)} className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${view === id ? "bg-accent text-white" : "text-text-secondary hover:text-text-primary hover:bg-bg-hover"}`}>
            <Icon className="w-4 h-4" /> {label}
          </button>
        ))}
      </div>

      {view === "overview" && workbench && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <section className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4"><Gauge className="w-5 h-5 text-accent" /><h2 className="text-lg font-semibold text-text-primary">Live readiness</h2></div>
            <ReadinessRow ready={workbench.runtime.python_supported} label="Python runtime" value={workbench.runtime.python} />
            <ReadinessRow ready={workbench.runtime.cuda_available} label="CUDA device" value={workbench.runtime.gpu_name || "Unavailable"} />
            <ReadinessRow ready={workbench.dataset.data_yaml_exists} label="Dataset configuration" value={workbench.dataset.data_yaml_exists ? "data.yaml found" : "Missing"} />
            <ReadinessRow ready={workbench.dataset.ready} label="Train and validation splits" value={`${workbench.dataset.splits.train.images} / ${workbench.dataset.splits.val.images}`} />
            <ReadinessRow ready={workbench.character_dataset.ready} label="Segmented characters" value={`${workbench.character_dataset.total_images} plates / ${workbench.character_dataset.class_count} classes`} />
            <p className="text-xs text-text-muted mt-4 font-mono break-all">{workbench.dataset.root}</p>
          </section>

          <section className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-3"><BarChart3 className="w-5 h-5 text-info" /><h2 className="text-lg font-semibold text-text-primary">Notebook reference run</h2></div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-info/10 text-info">Imported evidence</span>
            </div>
            <MetricTiles metrics={referenceMetrics} />
            <div className="grid grid-cols-2 gap-3 mt-4 text-sm">
              <div className="bg-bg-primary rounded-lg p-3"><p className="text-text-muted text-xs">Training</p><p className="text-text-primary mt-1">50 epochs · 0.336 hours</p></div>
              <div className="bg-bg-primary rounded-lg p-3"><p className="text-text-muted text-xs">Validation</p><p className="text-text-primary mt-1">527 images · 545 plates</p></div>
            </div>
            <div className="mt-4 space-y-2">
              {workbench.reference_run?.notes.map((note) => <p key={note} className="text-xs text-text-muted flex gap-2"><span className="text-warning">•</span>{note}</p>)}
            </div>
          </section>
        </div>
      )}

      {view === "dataset" && workbench && (
        <section className="space-y-5">
          <div className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div><h2 className="text-lg font-semibold text-text-primary">Dataset preview</h2><p className="text-sm text-text-secondary mt-1">Randomly inspect actual images before starting a GPU run.</p></div>
              <div className="flex flex-wrap items-center gap-2">
                {(["train", "val", "test"] as Split[]).map((split) => <button key={split} onClick={() => setSampleSplit(split)} className={`px-3 py-1.5 rounded-lg text-xs font-medium uppercase ${sampleSplit === split ? "bg-accent text-white" : "bg-bg-primary text-text-secondary hover:text-text-primary"}`}>{split}</button>)}
                <button onClick={() => setSampleSeed((seed) => seed + 1)} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-primary text-xs text-text-secondary hover:text-text-primary"><RefreshCw className="w-3.5 h-3.5" /> New sample</button>
              </div>
            </div>
          </div>
          {!workbench.dataset.ready ? (
            <div className="p-8 text-center bg-warning/10 border border-warning/30 rounded-xl"><AlertTriangle className="w-9 h-9 text-warning mx-auto mb-3" /><p className="text-text-primary font-medium">Master Plate dataset is not mounted</p><p className="text-sm text-text-secondary mt-1">Place it at the path shown in Overview or set MASTER_PLATE_DATASET_ROOT.</p></div>
          ) : samplesLoading ? <LoadingSpinner text="Sampling dataset images..." /> : (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              {samples.map((sample) => (
                <article key={sample.path} className="bg-bg-card border border-border rounded-xl overflow-hidden">
                  <img src={sample.url} alt={sample.name} className="w-full aspect-[4/3] object-cover bg-bg-secondary" />
                  <div className="p-3"><p className="text-sm text-text-primary truncate" title={sample.name}>{sample.name}</p><p className={`text-xs mt-1 ${sample.label_exists ? "text-success" : "text-warning"}`}>{sample.label_exists ? "Label found" : "Label missing"}</p></div>
                </article>
              ))}
            </div>
          )}
        </section>
      )}

      {view === "train" && workbench && (
        <div className="space-y-5">
          <section className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
              <div className="flex-1"><h2 className="text-lg font-semibold text-text-primary">YOLO11 training configuration</h2><p className="text-sm text-text-secondary mt-1">Notebook defaults are preserved and can be adjusted before launch.</p></div>
              <button disabled={starting || workbench.job.status === "training" || !workbench.dataset.ready} onClick={() => void launchTraining("plate")} className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                {starting || workbench.job.status === "training" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}{workbench.job.status === "training" ? "Training in progress" : "Start training"}
              </button>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
              <label className="text-xs text-text-muted">Epochs<input type="number" min={1} max={1000} value={trainingConfig.epochs} onChange={(event) => setTrainingConfig({ ...trainingConfig, epochs: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary outline-none focus:border-accent" /></label>
              <label className="text-xs text-text-muted">Image size<select value={trainingConfig.imgsz} onChange={(event) => setTrainingConfig({ ...trainingConfig, imgsz: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary outline-none focus:border-accent"><option value={640}>640 px</option><option value={960}>960 px</option><option value={1280}>1280 px</option></select></label>
              <label className="text-xs text-text-muted">Batch size<input type="number" min={1} max={256} value={trainingConfig.batch} onChange={(event) => setTrainingConfig({ ...trainingConfig, batch: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary outline-none focus:border-accent" /></label>
              <label className="text-xs text-text-muted">Device<select value={trainingConfig.device} onChange={(event) => setTrainingConfig({ ...trainingConfig, device: event.target.value })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary outline-none focus:border-accent"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></label>
            </div>
          </section>

          {(workbench.job.status !== "idle" || workbench.run.latest) && (
            <section className="bg-bg-card border border-border rounded-xl p-6 space-y-5">
              <div className="flex items-center justify-between"><div><h2 className="text-lg font-semibold text-text-primary">{workbench.job.run_name || "Latest training run"}</h2><p className="text-xs text-text-muted mt-1 capitalize">Status: {workbench.job.status}</p></div><span className="font-mono text-sm text-text-secondary">Epoch {epoch}/{totalEpochs}</span></div>
              <div className="w-full h-2 rounded-full bg-bg-primary overflow-hidden"><div className={`h-full rounded-full transition-all duration-500 ${workbench.job.status === "error" ? "bg-danger" : workbench.job.status === "completed" ? "bg-success" : "bg-info"}`} style={{ width: `${progress}%` }} /></div>
              <MetricTiles metrics={latestMetrics} />
              {workbench.job.log_tail && <details className="bg-bg-primary border border-border rounded-lg"><summary className="cursor-pointer px-4 py-3 text-sm text-text-secondary flex items-center gap-2"><SquareTerminal className="w-4 h-4" /> Training log</summary><pre className="px-4 pb-4 text-xs text-text-muted overflow-x-auto max-h-72 whitespace-pre-wrap">{workbench.job.log_tail}</pre></details>}
            </section>
          )}

          {workbench.run.artifacts.some((artifact) => artifact.exists) && (
            <section><div className="flex items-center gap-3 mb-4"><FileImage className="w-5 h-5 text-accent" /><h2 className="text-lg font-semibold text-text-primary">Training artifacts</h2></div><div className="grid grid-cols-1 xl:grid-cols-2 gap-4">{workbench.run.artifacts.filter((artifact) => artifact.exists).map((artifact) => <article key={artifact.name} className="bg-bg-card border border-border rounded-xl overflow-hidden"><div className="px-4 py-3 border-b border-border"><p className="text-sm font-medium text-text-primary">{artifact.label}</p></div>{artifact.url && <img src={artifact.url} alt={artifact.label} className="w-full bg-white object-contain max-h-[520px]" />}</article>)}</div></section>
          )}
        </div>
      )}

      {view === "characters" && workbench && (
        <div className="space-y-5">
          <section className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
              <div className="flex-1">
                <div className="flex items-center gap-3"><Type className="w-5 h-5 text-accent" /><h2 className="text-lg font-semibold text-text-primary">YOLO26 character detection</h2></div>
                <p className="text-sm text-text-secondary mt-2">Train stage three on cropped Egyptian plates with per-character boxes, then decode stage-two plate crops in reading order.</p>
                <div className="inline-flex items-center gap-2 mt-3 px-3 py-1.5 rounded-full bg-info/10 text-info text-xs"><CheckCircle2 className="w-3.5 h-3.5" /> Input comes directly from stage-two plate crops</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
                  <div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Classes</p><p className="text-lg font-mono text-text-primary mt-1">{workbench.character_dataset.class_count}</p></div>
                  <div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Train</p><p className="text-lg font-mono text-text-primary mt-1">{workbench.character_dataset.splits.train.images}</p></div>
                  <div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Validation</p><p className="text-lg font-mono text-text-primary mt-1">{workbench.character_dataset.splits.val.images}</p></div>
                  <div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Checkpoint</p><p className={`text-sm font-medium mt-2 ${workbench.character_run.best_model_exists ? "text-success" : "text-warning"}`}>{workbench.character_run.best_model_exists ? "Ready" : "Required"}</p></div>
                </div>
              </div>
              <button disabled={starting || workbench.job.status === "training" || !workbench.character_dataset.ready} onClick={() => void launchTraining("character")} className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                {starting || workbench.job.status === "training" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} Train characters
              </button>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
              <label className="text-xs text-text-muted">Epochs<input type="number" min={1} max={1000} value={characterConfig.epochs} onChange={(event) => setCharacterConfig({ ...characterConfig, epochs: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary" /></label>
              <label className="text-xs text-text-muted">Image size<select value={characterConfig.imgsz} onChange={(event) => setCharacterConfig({ ...characterConfig, imgsz: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary"><option value={640}>640 px</option><option value={960}>960 px</option></select></label>
              <label className="text-xs text-text-muted">Batch size<input type="number" min={1} max={256} value={characterConfig.batch} onChange={(event) => setCharacterConfig({ ...characterConfig, batch: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary" /></label>
              <label className="text-xs text-text-muted">Device<select value={characterConfig.device} onChange={(event) => setCharacterConfig({ ...characterConfig, device: event.target.value })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></label>
            </div>
            {!workbench.character_dataset.ready && <p className="text-sm text-warning mt-4">Character dataset is not ready at <span className="font-mono">{workbench.character_dataset.root}</span>.</p>}
          </section>
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-bg-card border border-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-text-primary">Decoder contract</h3>
              <div className="mt-4 space-y-3 text-sm text-text-secondary">
                <p className="flex gap-3"><span className="text-accent font-mono">01</span> Detect each digit or Arabic character box.</p>
                <p className="flex gap-3"><span className="text-accent font-mono">02</span> Cluster characters into plate rows.</p>
                <p className="flex gap-3"><span className="text-accent font-mono">03</span> Read each row right-to-left and separate letter/digit groups.</p>
              </div>
            </div>
            <div className="bg-warning/5 border border-warning/25 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-warning">Benchmark note</h3>
              <p className="text-sm text-text-secondary mt-3">The supplied split contains nine original base identities shared across train and validation/test. Regroup those identities before publishing final accuracy.</p>
            </div>
          </section>
          {workbench.job.stage === "character" && workbench.job.status !== "idle" && (
            <section className="bg-bg-card border border-border rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between gap-4"><div><h3 className="font-semibold text-text-primary">{workbench.job.run_name}</h3><p className="text-xs text-text-muted mt-1 capitalize">{workbench.job.status} · YOLO26 characters</p></div><span className="text-sm font-mono text-text-secondary">{workbench.job.metrics.latest?.epoch || 0}/{workbench.job.epochs}</span></div>
              <MetricTiles metrics={workbench.job.metrics.latest} />
              {workbench.job.log_tail && <details className="bg-bg-primary border border-border rounded-lg"><summary className="cursor-pointer px-4 py-3 text-sm text-text-secondary flex items-center gap-2"><SquareTerminal className="w-4 h-4" /> Character training log</summary><pre className="px-4 pb-4 text-xs text-text-muted overflow-x-auto max-h-72 whitespace-pre-wrap">{workbench.job.log_tail}</pre></details>}
            </section>
          )}
          {workbench.character_run.latest && (
            <section className="bg-bg-card border border-border rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4"><BarChart3 className="w-5 h-5 text-info" /><h2 className="text-lg font-semibold text-text-primary">Latest character run</h2></div>
              <MetricTiles metrics={workbench.character_run.latest} />
            </section>
          )}
        </div>
      )}

      {view === "evaluate" && workbench && (
        <section className="space-y-5">
          <div className="bg-bg-card border border-border rounded-xl p-6">
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5">
              <div><div className="flex items-center gap-3"><Sparkles className="w-5 h-5 text-accent" /><h2 className="text-lg font-semibold text-text-primary">Random validation evaluation</h2></div><p className="text-sm text-text-secondary mt-2">Detect plates, compare the full image, and generate natural Lanczos-enhanced crops.</p></div>
              <button disabled={evaluating || !workbench.run.best_model_exists || workbench.dataset.splits.val.images === 0} onClick={() => void runEvaluation()} className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed">{evaluating ? <Loader2 className="w-4 h-4 animate-spin" /> : <ScanSearch className="w-4 h-4" />} Run evaluation</button>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
              <label className="text-xs text-text-muted">Samples<input type="number" min={1} max={5} value={evaluationConfig.count} onChange={(event) => setEvaluationConfig({ ...evaluationConfig, count: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary" /></label>
              <label className="text-xs text-text-muted">Confidence<input type="number" min={0.01} max={1} step={0.05} value={evaluationConfig.confidence} onChange={(event) => setEvaluationConfig({ ...evaluationConfig, confidence: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary" /></label>
              <label className="text-xs text-text-muted">Crop scale<select value={evaluationConfig.scale_factor} onChange={(event) => setEvaluationConfig({ ...evaluationConfig, scale_factor: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary"><option value={2}>2×</option><option value={4}>4×</option><option value={6}>6×</option><option value={8}>8×</option></select></label>
              <label className="text-xs text-text-muted">Random seed<input type="number" value={evaluationConfig.seed} onChange={(event) => setEvaluationConfig({ ...evaluationConfig, seed: Number(event.target.value) })} className="mt-1.5 w-full bg-bg-primary border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary" /></label>
            </div>
          </div>
          {evaluating && <LoadingSpinner text="Running YOLO11 on validation samples..." />}
          {evaluation.map((sample) => (
            <article key={sample.name} className="bg-bg-card border border-border rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-border flex items-center justify-between"><p className="text-sm font-medium text-text-primary truncate">{sample.name}</p><span className="text-xs text-text-muted">{sample.detections.length} plate(s)</span></div>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-0">
                <div className="p-4 border-b xl:border-b-0 xl:border-r border-border"><p className="text-xs uppercase tracking-wide text-text-muted mb-3">Full image detections</p><img src={sample.annotated_image} alt={`Detection result for ${sample.name}`} className="w-full rounded-lg object-contain max-h-[520px] bg-bg-primary" /></div>
                <div className="p-4"><p className="text-xs uppercase tracking-wide text-text-muted mb-3">Enhanced plate crops</p>{sample.detections.length ? <div className="space-y-4">{sample.detections.map((detection, index) => <div key={`${sample.name}-${index}`} className="bg-bg-primary rounded-lg p-3"><img src={detection.image} alt={`Enhanced plate ${index + 1}`} className="w-full max-h-48 object-contain rounded" /><div className="flex items-center justify-between mt-2 text-xs"><span className="text-text-muted">Plate {index + 1}</span><span className="font-mono text-success">{percent(detection.confidence)}</span></div></div>)}</div> : <div className="h-48 grid place-items-center text-text-muted"><div className="text-center"><ImageIcon className="w-9 h-9 mx-auto mb-2 opacity-30" /><p className="text-sm">No plate above threshold</p></div></div>}</div>
              </div>
            </article>
          ))}
        </section>
      )}
    </div>
  );
}
