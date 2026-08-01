import { AlertCircle, Check, CheckCircle, CreditCard, Flag, X } from "lucide-react";
import type { Detection, ReviewDecision } from "../types";

interface Props {
  detection: Detection;
  index: number;
  review?: ReviewDecision;
  correction?: string;
  onReview?: (decision: ReviewDecision) => void;
  onCorrectionChange?: (value: string) => void;
}

const REVIEW_STYLES: Record<ReviewDecision, string> = {
  accepted: "border-success/60 bg-success/5",
  rejected: "border-danger/60 bg-danger/5",
  needs_review: "border-warning/60 bg-warning/5",
};

function PlateText({ value, className }: { value: string; className: string }) {
  if (!value) {
    return <span className={className}>—</span>;
  }

  const tokens = value.match(/[A-Za-z\u0600-\u06FF]+|\d+|[^\s]/g) ?? [value];
  return (
    <span className={`${className} inline-flex flex-wrap items-baseline gap-x-2`} dir="ltr">
      {tokens.map((token, tokenIndex) => (
        <bdi key={`${token}-${tokenIndex}`} dir={/[\u0600-\u06FF]/.test(token) ? "rtl" : "ltr"}>
          {token}
        </bdi>
      ))}
    </span>
  );
}

function PlatePreview({ src, index }: { src: string; index: number }) {
  return (
    <div className="h-32 bg-bg-primary border-b border-border grid place-items-center px-4 py-3 overflow-hidden">
      <div className="h-24 w-full max-w-72 overflow-hidden rounded-md bg-bg-secondary/45">
        <img
          src={src}
          alt={`Plate crop ${index + 1}`}
          loading="lazy"
          decoding="async"
          className="block h-full w-full object-contain object-center"
        />
      </div>
    </div>
  );
}

export default function PlateCard({ detection, index, review, correction = "", onReview, onCorrectionChange }: Props) {
  const isValid = detection.formatted_text.length >= 3;
  const selectedVariant = detection.character_preprocess?.selected_variant;
  const selectedVariantLabel = selectedVariant?.replace(/_/g, " ");
  const triedVariants = detection.character_preprocess?.tried_variants ?? 0;
  const characterCount = detection.characters?.length ?? 0;
  const hasCharacterCandidate = Boolean(detection.character_text && characterCount > 0);
  const characterComplete = detection.text_source === "character_detector";
  const ocrLabel = characterComplete
    ? detection.ocr_source === "keras_crnn" ? "Keras OCR comparison" : "OCR comparison"
    : "Raw OCR fallback";

  return (
    <div className={`bg-bg-card border rounded-xl overflow-hidden transition-colors ${review ? REVIEW_STYLES[review] : "border-border hover:border-accent/40"}`}>
      {detection.plate_crop && <PlatePreview src={detection.plate_crop} index={index} />}
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2"><CreditCard className="w-4 h-4 text-accent" /><span className="text-sm font-medium text-text-secondary">Plate #{index + 1}</span></div>
          <div className="flex items-center gap-2">
            {hasCharacterCandidate && !characterComplete && <span className="text-[10px] px-2 py-1 rounded-full bg-warning/10 text-warning">YOLO26 partial</span>}
            {detection.text_source && <span className={`text-[10px] px-2 py-1 rounded-full ${characterComplete ? "bg-accent/10 text-accent" : "bg-info/10 text-info"}`}>{characterComplete ? "YOLO26" : detection.text_source === "keras_crnn" ? "Keras CRNN" : "CRNN fallback"}</span>}
            {isValid ? <CheckCircle className="w-4 h-4 text-success" /> : <AlertCircle className="w-4 h-4 text-warning" />}
          </div>
        </div>
        <div className="space-y-2">
          <div><p className="text-xs text-text-muted mb-0.5">Decoded plate</p><p><PlateText value={detection.formatted_text} className="text-xl font-bold font-mono text-text-primary" /></p></div>
          {hasCharacterCandidate && (
            <div>
              <p className="text-xs text-text-muted mb-0.5">{characterComplete ? "YOLO26 character detector" : `YOLO26 partial candidate (${characterCount}/6 minimum)`}</p>
              <p><PlateText value={detection.character_text || ""} className={`text-sm font-mono ${characterComplete ? "text-accent" : "text-warning"}`} /></p>
            </div>
          )}
          {detection.ocr_text && (
            <div><p className="text-xs text-text-muted mb-0.5">{ocrLabel}</p><p><PlateText value={detection.ocr_formatted || detection.ocr_text} className="text-sm font-mono text-text-secondary" /></p></div>
          )}
          {!hasCharacterCandidate && !detection.ocr_text && detection.plate_text && (
            <div><p className="text-xs text-text-muted mb-0.5">Model output</p><p><PlateText value={detection.plate_text} className="text-sm font-mono text-text-secondary" /></p></div>
          )}
          {selectedVariantLabel && (
            <p className="text-[11px] text-text-muted">
              Stage-three crop: {selectedVariantLabel}{triedVariants ? ` · ${triedVariants} variant${triedVariants === 1 ? "" : "s"}` : ""}
            </p>
          )}
          {detection.characters && detection.characters.length > 0 && (
            <div className="pt-1">
              <p className="text-[10px] uppercase tracking-wide text-text-muted mb-2">{detection.characters.length} ordered characters · right to left</p>
              <div className="flex flex-wrap justify-start gap-1.5" dir="rtl" aria-label="Ordered plate characters, read from right to left">
              {detection.characters.map((character) => (
                <span dir="auto" key={`${character.row}-${character.order}`} title={`${character.class_name} · ${(character.confidence * 100).toFixed(1)}%`} className="px-2 py-1 rounded bg-bg-primary text-sm text-text-primary">
                  {character.glyph}
                </span>
              ))}
              </div>
            </div>
          )}
          {detection.vehicle && <p className="text-xs text-text-muted">Inside {detection.vehicle.class_name} · {(detection.vehicle.confidence * 100).toFixed(1)}% vehicle confidence</p>}
          {onCorrectionChange && (
            <label className="block pt-1">
              <span className="text-xs text-text-muted">Corrected plate after review</span>
              <input
                value={correction}
                onChange={(event) => onCorrectionChange(event.target.value)}
                placeholder={detection.formatted_text || "Enter reviewed plate"}
                dir="auto"
                className="mt-1 w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-sm font-mono text-text-primary outline-none focus:border-accent"
              />
            </label>
          )}
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
