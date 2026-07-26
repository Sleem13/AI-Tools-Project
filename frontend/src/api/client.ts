import type {
  DatasetInfo,
  DatasetSample,
  DetectionResponse,
  EvaluationSample,
  HealthResponse,
  PipelineStatus,
  TrainingJob,
  TrainingStatus,
  TrainingWorkbench,
  VideoJob,
} from "../types";

const BASE = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/+$/, "");

export function apiUrl(path: string): string {
  if (!path || /^(?:data:|blob:|https?:\/\/)/i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${BASE}${normalizedPath}`;
}

function errorDetail(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const detail = "detail" in payload ? payload.detail : "error" in payload ? payload.error : null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const message = "msg" in item ? String(item.msg) : JSON.stringify(item);
        const location = "loc" in item && Array.isArray(item.loc) ? item.loc.join(".") : "";
        return location ? `${location}: ${message}` : message;
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new Error(
      `Could not connect to the ALPR API${BASE ? ` at ${BASE}` : ""}. Check that the backend is running.`,
    );
  }
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    throw new Error(errorDetail(payload, res.statusText || `Request failed (${res.status})`));
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function normalizeVideoJob(job: VideoJob): VideoJob {
  return {
    ...job,
    result_url: job.result_url ? apiUrl(job.result_url) : null,
  };
}

export function detectPlate(file: File, conf = 0.25) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", String(conf));
  return request<DetectionResponse>("/api/detect", { method: "POST", body: form });
}

export function startVideoDetection(file: File, conf = 0.25, frameStride = 3) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", String(conf));
  form.append("frame_stride", String(frameStride));
  return request<{ job: VideoJob }>("/api/detect/video", {
    method: "POST",
    body: form,
  }).then(({ job }) => ({ job: normalizeVideoJob(job) }));
}

export function getVideoDetectionStatus(jobId: string) {
  return request<{ job: VideoJob }>(`/api/detect/video/${encodeURIComponent(jobId)}`)
    .then(({ job }) => ({ job: normalizeVideoJob(job) }));
}

export function getHealth() {
  return request<HealthResponse>("/api/health");
}

export function getPipelineStatus() {
  return request<PipelineStatus>("/api/pipeline/status");
}

export function getDatasets() {
  return request<{ datasets: DatasetInfo[] }>("/api/datasets");
}

export function getTrainingStatus() {
  return request<{ training: TrainingStatus[] }>("/api/training/status");
}

export function getTrainingWorkbench() {
  return request<TrainingWorkbench>("/api/training/workbench").then((workbench) => ({
    ...workbench,
    run: {
      ...workbench.run,
      artifacts: workbench.run.artifacts.map((artifact) => ({
        ...artifact,
        url: artifact.url ? apiUrl(artifact.url) : null,
      })),
    },
    character_run: {
      ...workbench.character_run,
      artifacts: workbench.character_run.artifacts.map((artifact) => ({
        ...artifact,
        url: artifact.url ? apiUrl(artifact.url) : null,
      })),
    },
  }));
}

export function getDatasetSamples(split: "train" | "val" | "test", count = 4, seed = 42) {
  const params = new URLSearchParams({ split, count: String(count), seed: String(seed) });
  return request<{ split: string; count: number; images: DatasetSample[] }>(
    `/api/training/dataset/samples?${params}`,
  ).then((response) => ({
    ...response,
    images: response.images.map((image) => ({ ...image, url: apiUrl(image.url) })),
  }));
}

export function startPlateTraining(config: {
  epochs: number;
  imgsz: number;
  batch: number;
  device: string;
  stage?: "plate" | "character";
}) {
  return request<{ job: TrainingJob }>("/api/training/start", {
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
  return request<{ model: string; samples: EvaluationSample[] }>(
    "/api/training/evaluate",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    },
  );
}
