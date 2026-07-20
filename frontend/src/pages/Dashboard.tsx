import { useCallback, useState } from "react";
import {
  ScanBox,
  Clock,
  Zap,
  AlertTriangle,
} from "lucide-react";
import FileUploader from "../components/FileUploader";
import PlateCard from "../components/PlateCard";
import StatsGrid from "../components/StatsGrid";
import LoadingSpinner from "../components/LoadingSpinner";
import { detectPlate } from "../api/client";
import type { Detection } from "../types";

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState<number | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setDetections([]);
    setTimeMs(null);
    setImageName(file.name);

    try {
      const res = await detectPlate(file);
      setDetections(res.detections);
      setTimeMs(res.processing_time_ms);
    } catch (e: any) {
      setError(e.message || "Detection failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="max-w-6xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3">
          <ScanBox className="w-8 h-8 text-accent" />
          License Plate Detection
        </h1>
        <p className="text-text-secondary mt-1">
          Upload a vehicle image to detect and read Egyptian license plates.
        </p>
      </div>

      <FileUploader onFile={handleFile} />

      {loading && <LoadingSpinner text="Running detection + OCR pipeline..." />}

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {!loading && detections.length > 0 && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatsGrid
              icon={<ScanBox className="w-4 h-4" />}
              label="Plates Detected"
              value={detections.length}
              color="text-accent"
            />
            <StatsGrid
              icon={<Clock className="w-4 h-4" />}
              label="Processing Time"
              value={timeMs !== null ? `${timeMs.toFixed(0)}ms` : "—"}
              color="text-info"
            />
            <StatsGrid
              icon={<Zap className="w-4 h-4" />}
              label="Avg Confidence"
              value={
                detections.length > 0
                  ? `${(
                      (detections.reduce((s, d) => s + d.confidence, 0) /
                        detections.length) *
                      100
                    ).toFixed(1)}%`
                  : "—"
              }
              color="text-success"
            />
          </div>

          <div>
            <h2 className="text-lg font-semibold text-text-primary mb-4">
              Results {imageName && <span className="text-text-muted font-normal text-sm">— {imageName}</span>}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {detections.map((d, i) => (
                <PlateCard key={i} detection={d} index={i} />
              ))}
            </div>
          </div>
        </>
      )}

      {!loading && detections.length === 0 && !error && (
        <div className="text-center py-16 text-text-muted">
          <ScanBox className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>No results yet. Upload an image to get started.</p>
        </div>
      )}
    </div>
  );
}
