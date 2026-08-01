import { useEffect, useState } from "react";
import {
  Settings as SettingsIcon,
  Shield,
  Sliders,
  Server,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import LoadingSpinner from "../components/LoadingSpinner";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../types";

export default function Settings() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const ocrSourceLabel =
    health?.model_sources?.ocr === "keras_crnn"
      ? "Keras CRNN"
      : health?.model_sources?.ocr === "crnn"
        ? "PyTorch CRNN"
        : null;

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({
          status: "offline",
          models_loaded: { detection: false, character: false, ocr: false },
          model_sources: { ocr: null, ocr_path: null },
        })
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading settings..." />;

  return (
    <div className="max-w-4xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3">
          <SettingsIcon className="w-8 h-8 text-accent" />
          Settings
        </h1>
        <p className="text-text-secondary mt-1">
          System status and configuration.
        </p>
      </div>

      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <Server className="w-5 h-5 text-accent" />
          <h2 className="text-lg font-semibold text-text-primary">API Status</h2>
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2">
            <span className="text-sm text-text-secondary">Backend</span>
            <span
              className={`flex items-center gap-2 text-sm font-medium ${
                health?.status === "ok" ? "text-success" : "text-danger"
              }`}
            >
              {health?.status === "ok" ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {health?.status === "ok" ? "Connected" : "Offline"}
            </span>
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <span className="text-sm text-text-secondary">Detection Model</span>
            <span
              className={`flex items-center gap-2 text-sm ${
                health?.models_loaded?.detection ? "text-success" : "text-text-muted"
              }`}
            >
              {health?.models_loaded?.detection ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {health?.models_loaded?.detection ? "Loaded" : "Not loaded"}
            </span>
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <span className="text-sm text-text-secondary">Character Model</span>
            <span
              className={`flex items-center gap-2 text-sm ${
                health?.models_loaded?.character ? "text-success" : "text-text-muted"
              }`}
            >
              {health?.models_loaded?.character ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {health?.models_loaded?.character ? "Loaded" : "Not loaded"}
            </span>
          </div>
          <div className="flex items-center justify-between py-2 border-t border-border">
            <span className="text-sm text-text-secondary">OCR Model</span>
            <span
              className={`flex items-center gap-2 text-sm ${
                health?.models_loaded?.ocr ? "text-success" : "text-text-muted"
              }`}
            >
              {health?.models_loaded?.ocr ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <XCircle className="w-4 h-4" />
              )}
              {health?.models_loaded?.ocr ? `Loaded${ocrSourceLabel ? ` (${ocrSourceLabel})` : ""}` : "Not loaded"}
            </span>
          </div>
          {health?.models_loaded?.ocr && health?.model_sources?.ocr_path && (
            <div className="flex items-center justify-between py-2 border-t border-border">
              <span className="text-sm text-text-secondary">OCR Weights</span>
              <span className="max-w-md truncate text-right text-xs font-mono text-text-muted" title={health.model_sources.ocr_path}>
                {health.model_sources.ocr_path}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <Sliders className="w-5 h-5 text-accent" />
          <h2 className="text-lg font-semibold text-text-primary">
            Default Parameters
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            { label: "Detection Confidence", value: "0.25" },
            { label: "IOU Threshold", value: "0.45" },
            { label: "Character Confidence", value: "0.25" },
            { label: "Character Reading Order", value: "Right to left" },
            { label: "OCR Fallback", value: ocrSourceLabel ?? "Automatic" },
            { label: "Input Height (CRNN)", value: "32" },
            { label: "Duplicate Hash Threshold", value: "5" },
            { label: "Blur Threshold", value: "100.0" },
            { label: "Random Seed", value: "42" },
            { label: "Target Image Size", value: "640 x 640" },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center justify-between bg-bg-primary rounded-lg px-4 py-3"
            >
              <span className="text-sm text-text-secondary">{item.label}</span>
              <span className="text-sm font-mono text-text-primary">{item.value}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-bg-card border border-border rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <Shield className="w-5 h-5 text-accent" />
          <h2 className="text-lg font-semibold text-text-primary">About</h2>
        </div>
        <div className="space-y-2 text-sm text-text-secondary">
          <p>
            <strong className="text-text-primary">ALPR Dataset Pipeline</strong> v1.0.0
          </p>
          <p>Production-grade dataset engineering and ALPR for Egyptian license plates.</p>
          <p>
            License: MIT &middot; Python 3.11+ &middot; React + Vite
          </p>
          <p className="text-text-muted text-xs mt-3">
            Built as a three-stage cascade: YOLO11 vehicles, YOLO11 plates,
            and YOLO26 character detection with automatic CRNN+CTC fallback.
          </p>
        </div>
      </div>
    </div>
  );
}
