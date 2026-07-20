export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  plate_text: string;
  formatted_text: string;
}

export interface DetectionResponse {
  detections: Detection[];
  processing_time_ms: number;
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

export interface HealthResponse {
  status: string;
  models_loaded: {
    detection: boolean;
    ocr: boolean;
  };
}
