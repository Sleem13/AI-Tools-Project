const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail = payload?.detail || payload?.error || res.statusText;
    throw new Error(detail);
  }
  return res.json();
}

export function detectPlate(file: File, conf = 0.25) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", String(conf));
  return request<import("../types").DetectionResponse>(
    "/api/detect",
    { method: "POST", body: form }
  );
}

export function startVideoDetection(file: File, conf = 0.25, frameStride = 3) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", String(conf));
  form.append("frame_stride", String(frameStride));
  return request<{ job: import("../types").VideoJob }>("/api/detect/video", {
    method: "POST",
    body: form,
  });
}

export function getVideoDetectionStatus(jobId: string) {
  return request<{ job: import("../types").VideoJob }>(`/api/detect/video/${jobId}`);
}

export function getHealth() {
  return request<{ status: string; models_loaded: { detection: boolean; ocr: boolean } }>(
    "/api/health"
  );
}

export function getPipelineStatus() {
  return request<{ stages: any[]; total_images: number; datasets: string[] }>(
    "/api/pipeline/status"
  );
}

export function getDatasets() {
  return request<{ datasets: any[] }>("/api/datasets");
}

export function getTrainingStatus() {
  return request<{ training: any[] }>("/api/training/status");
}

export function getTrainingWorkbench() {
  return request<import("../types").TrainingWorkbench>("/api/training/workbench");
}

export function getDatasetSamples(split: "train" | "val" | "test", count = 4, seed = 42) {
  const params = new URLSearchParams({ split, count: String(count), seed: String(seed) });
  return request<{ split: string; count: number; images: import("../types").DatasetSample[] }>(
    `/api/training/dataset/samples?${params}`,
  );
}

export function startPlateTraining(config: { epochs: number; imgsz: number; batch: number; device: string }) {
  return request<{ job: import("../types").TrainingJob }>("/api/training/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export function evaluateValidationSamples(config: {
  count: number;
  confidence: number;
  scale_factor: number;
  seed: number;
}) {
  return request<{ model: string; samples: import("../types").EvaluationSample[] }>(
    "/api/training/evaluate",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    },
  );
}
