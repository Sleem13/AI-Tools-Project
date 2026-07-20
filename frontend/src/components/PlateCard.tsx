import { CreditCard, CheckCircle, AlertCircle } from "lucide-react";
import type { Detection } from "../types";

interface Props {
  detection: Detection;
  index: number;
}

export default function PlateCard({ detection, index }: Props) {
  const isValid = detection.formatted_text.length >= 3;

  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 hover:border-accent/40 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <CreditCard className="w-4 h-4 text-accent" />
          <span className="text-sm font-medium text-text-secondary">
            Plate #{index + 1}
          </span>
        </div>
        {isValid ? (
          <CheckCircle className="w-4 h-4 text-success" />
        ) : (
          <AlertCircle className="w-4 h-4 text-warning" />
        )}
      </div>

      <div className="space-y-2">
        <div>
          <p className="text-xs text-text-muted mb-0.5">Formatted</p>
          <p className="text-xl font-bold font-mono tracking-wider text-text-primary">
            {detection.formatted_text || "—"}
          </p>
        </div>
        <div>
          <p className="text-xs text-text-muted mb-0.5">Raw OCR</p>
          <p className="text-sm font-mono text-text-secondary">
            {detection.plate_text || "—"}
          </p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-border flex items-center justify-between">
        <span className="text-xs text-text-muted">Confidence</span>
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 rounded-full bg-bg-primary overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                detection.confidence > 0.7
                  ? "bg-success"
                  : detection.confidence > 0.4
                    ? "bg-warning"
                    : "bg-danger"
              }`}
              style={{ width: `${detection.confidence * 100}%` }}
            />
          </div>
          <span className="text-sm font-medium text-text-primary">
            {(detection.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
