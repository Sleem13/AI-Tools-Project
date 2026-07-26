import { AlertCircle, Check, CheckCircle, CreditCard, Flag, X } from "lucide-react";
import type { Detection, ReviewDecision } from "../types";

interface Props {
  detection: Detection;
  index: number;
  review?: ReviewDecision;
  onReview?: (decision: ReviewDecision) => void;
}

const REVIEW_STYLES: Record<ReviewDecision, string> = {
  accepted: "border-success/60 bg-success/5",
  rejected: "border-danger/60 bg-danger/5",
  needs_review: "border-warning/60 bg-warning/5",
};

export default function PlateCard({ detection, index, review, onReview }: Props) {
  const isValid = detection.formatted_text.length >= 3;

  return (
    <div className={`bg-bg-card border rounded-xl overflow-hidden transition-colors ${review ? REVIEW_STYLES[review] : "border-border hover:border-accent/40"}`}>
      {detection.plate_crop && <div className="h-28 bg-bg-primary border-b border-border grid place-items-center p-2"><img src={detection.plate_crop} alt={`Enhanced plate ${index + 1}`} className="max-w-full max-h-full object-contain" /></div>}
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2"><CreditCard className="w-4 h-4 text-accent" /><span className="text-sm font-medium text-text-secondary">Plate #{index + 1}</span></div>
          <div className="flex items-center gap-2">
            {detection.text_source && <span className={`text-[10px] px-2 py-1 rounded-full ${detection.text_source === "character_detector" ? "bg-accent/10 text-accent" : "bg-info/10 text-info"}`}>{detection.text_source === "character_detector" ? "YOLO26" : "CRNN fallback"}</span>}
            {isValid ? <CheckCircle className="w-4 h-4 text-success" /> : <AlertCircle className="w-4 h-4 text-warning" />}
          </div>
        </div>
        <div className="space-y-2">
          <div><p className="text-xs text-text-muted mb-0.5">Decoded plate</p><p className="text-xl font-bold font-mono tracking-wider text-text-primary" dir="auto">{detection.formatted_text || "—"}</p></div>
          <div><p className="text-xs text-text-muted mb-0.5">{detection.text_source === "character_detector" ? "Character detector" : "Raw OCR"}</p><p className="text-sm font-mono text-text-secondary" dir="auto">{detection.plate_text || "—"}</p></div>
          {detection.characters && detection.characters.length > 0 && (
            <div className="pt-1">
              <p className="text-[10px] uppercase tracking-wide text-text-muted mb-2">{detection.characters.length} ordered characters</p>
              <div className="flex flex-wrap gap-1.5">
              {detection.characters.map((character) => (
                <span key={`${character.row}-${character.order}`} title={`${character.class_name} · ${(character.confidence * 100).toFixed(1)}%`} className="px-2 py-1 rounded bg-bg-primary text-sm text-text-primary">
                  {character.glyph}
                </span>
              ))}
              </div>
            </div>
          )}
          {detection.vehicle && <p className="text-xs text-text-muted">Inside {detection.vehicle.class_name} · {(detection.vehicle.confidence * 100).toFixed(1)}% vehicle confidence</p>}
        </div>
        <div className="mt-3 pt-3 border-t border-border flex items-center justify-between">
          <span className="text-xs text-text-muted">Plate confidence</span>
          <span className="text-sm font-medium text-text-primary">{(detection.confidence * 100).toFixed(1)}%</span>
        </div>
        {onReview && (
          <div className="grid grid-cols-3 gap-2 mt-4">
            <button onClick={() => onReview("accepted")} className={`inline-flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs ${review === "accepted" ? "bg-success text-white" : "bg-bg-primary text-text-secondary hover:text-success"}`}><Check className="w-3.5 h-3.5" /> Accept</button>
            <button onClick={() => onReview("needs_review")} className={`inline-flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs ${review === "needs_review" ? "bg-warning text-white" : "bg-bg-primary text-text-secondary hover:text-warning"}`}><Flag className="w-3.5 h-3.5" /> Review</button>
            <button onClick={() => onReview("rejected")} className={`inline-flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs ${review === "rejected" ? "bg-danger text-white" : "bg-bg-primary text-text-secondary hover:text-danger"}`}><X className="w-3.5 h-3.5" /> Reject</button>
          </div>
        )}
      </div>
    </div>
  );
}
