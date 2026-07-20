const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export function detectPlate(file: File, conf = 0.25) {
  const form = new FormData();
  form.append("file", file);
  form.append("conf", String(conf));
  return request<{ detections: any[]; processing_time_ms: number }>(
    "/api/detect",
    { method: "POST", body: form }
  );
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
