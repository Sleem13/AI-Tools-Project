export interface Detection {
  id: string;
  bbox: [number, number, number, number];
  confidence: number;
  plate_text: string;
  formatted_text: string;
  plate_crop?: string;
  combined_confidence?: number;
  vehicle?: {
    bbox: [number, number, number, number];
    confidence: number;
    class_id: number;
    class_name: string;
  };
}

export interface DetectionResponse {
  detections: Detection[];
  annotated_image: string;
  processing_time_ms: number;
}

export type ReviewDecision = "accepted" | "rejected" | "needs_review";

export interface VideoDetectionEvent {
  frame: number;
  time_seconds: number;
  detections: number;
  plates: string[];
}

export interface VideoJob {
  id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "error";
  progress: number;
  processed_frames: number;
  total_frames: number;
  frames_with_detections: number;
  total_detections: number;
  events: VideoDetectionEvent[];
  error: string | null;
  result_url: string | null;
  created_at: string;
}

export interface PipelineStage {
  name: string;
  status: "completed" | "pending" | "running" | "error";
  last_run: string | null;
  script: string;
}

export interface PipelineStatus {
  stages: PipelineStage[];
  total_images: number;
  datasets: string[];
}

export interface DatasetInfo {
  name: string;
  format: string;
  image_count: number;
  annotation_count: number;
  issues: number;
}

export interface TrainingStatus {
  model: string;
  status: "idle" | "training" | "completed" | "error";
  epoch: number;
  total_epochs: number;
  loss: number;
  metrics: Record<string, number>;
}

export interface RuntimeInfo {
  python: string;
  python_supported: boolean;
  torch: string;
  torch_cuda: string | null;
  ultralytics: string;
  cuda_available: boolean;
  device_count: number;
  gpu_name: string | null;
  gpu_memory_gib: number | null;
  compute_capability: string | null;
  ready: boolean;
}

export interface DatasetSplitSummary {
  images: number;
  labels: number;
}

export interface MasterDatasetSummary {
  root: string;
  exists: boolean;
  data_yaml: string;
  data_yaml_exists: boolean;
  splits: Record<"train" | "val" | "test", DatasetSplitSummary>;
  total_images: number;
  total_labels: number;
  ready: boolean;
}

export interface TrainingMetrics {
  epoch?: number | null;
  train_box_loss?: number | null;
  train_cls_loss?: number | null;
  train_dfl_loss?: number | null;
  precision?: number | null;
  recall?: number | null;
  map50?: number | null;
  map50_95?: number | null;
}

export interface TrainingJob {
  status: "idle" | "training" | "completed" | "error";
  run_name: string | null;
  run_dir: string | null;
  log_path: string | null;
  epochs: number;
  return_code: number | null;
  log_tail: string;
  metrics: { history: TrainingMetrics[]; latest: TrainingMetrics | null };
}

export interface TrainingArtifact {
  name: string;
  label: string;
  exists: boolean;
  url: string | null;
}

export interface NotebookReferenceRun {
  label: string;
  environment: Record<string, string | number>;
  dataset: Record<string, number>;
  training: Record<string, string | number | boolean>;
  metrics: Record<string, number>;
  notes: string[];
}

export interface TrainingWorkbench {
  runtime: RuntimeInfo;
  dataset: MasterDatasetSummary;
  job: TrainingJob;
  run: {
    directory: string | null;
    best_model: string | null;
    best_model_exists: boolean;
    latest: TrainingMetrics | null;
    history: TrainingMetrics[];
    artifacts: TrainingArtifact[];
  };
  reference_run: NotebookReferenceRun | null;
  defaults: { epochs: number; imgsz: number; batch: number; device: string };
}

export interface DatasetSample {
  name: string;
  path: string;
  label_exists: boolean;
  url: string;
}

export interface EvaluationSample {
  name: string;
  annotated_image: string;
  detections: Array<{
    image: string;
    confidence: number;
    bbox: [number, number, number, number];
  }>;
}

export interface HealthResponse {
  status: string;
  models_loaded: {
    detection: boolean;
    ocr: boolean;
  };
}
