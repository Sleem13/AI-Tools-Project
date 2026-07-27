import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Clock, Download, FileVideo, ScanBox, ShieldCheck, X, Zap } from "lucide-react";
import { detectPlate, getHealth, getVideoDetectionStatus, startVideoDetection } from "../api/client";
import FileUploader from "../components/FileUploader";
import LoadingSpinner from "../components/LoadingSpinner";
import PipelineStrip from "../components/PipelineStrip";
import PlateCard from "../components/PlateCard";
import StatsGrid from "../components/StatsGrid";
import type { Detection, HealthResponse, ReviewDecision, VideoDetectionEvent, VideoJob } from "../types";

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [videoJob, setVideoJob] = useState<VideoJob | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewDecision>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState<number | null>(null);
  const [mediaName, setMediaName] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.25);
  const [frameStride, setFrameStride] = useState(3);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [reviewFrame, setReviewFrame] = useState<VideoDetectionEvent | null>(null);
  const videoJobId = videoJob?.id;
  const videoJobStatus = videoJob?.status;

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setHealthChecked(true));
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setDetections([]);
    setAnnotatedImage(null);
    setVideoJob(null);
    setReviews({});
    setTimeMs(null);
    setMediaName(file.name);
    try {
      if (file.type.startsWith("video/")) {
        const response = await startVideoDetection(file, confidence, frameStride);
        setVideoJob(response.job);
      } else {
        const response = await detectPlate(file, confidence);
        setDetections(response.detections);
        setAnnotatedImage(response.annotated_image);
        setTimeMs(response.processing_time_ms);
      }
    } catch (caught: any) {
      setError(caught.message || "Detection failed");
    } finally {
      setLoading(false);
    }
  }, [confidence, frameStride]);

  useEffect(() => {
    if (!videoJobId || !videoJobStatus || !["queued", "processing"].includes(videoJobStatus)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await getVideoDetectionStatus(videoJobId);
        setVideoJob(response.job);
        if (response.job.status === "error") setError(response.job.error || "Video processing failed");
      } catch (caught: any) {
        setError(caught.message || "Could not read video progress");
        window.clearInterval(timer);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [videoJobId, videoJobStatus]);

  function exportReview() {
    const payload = { media: mediaName, reviewed_at: new Date().toISOString(), detections, reviews, video_job: videoJob };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${(mediaName || "media").replace(/\.[^.]+$/, "")}-review.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const reviewedCount = Object.keys(reviews).length;
  const averageConfidence = detections.length ? detections.reduce((sum, detection) => sum + detection.confidence, 0) / detections.length : 0;

  return (
    <div className="max-w-7xl space-y-7">
      <div>
        <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3"><ScanBox className="w-8 h-8 text-accent" /> Human review media lab</h1>
        <p className="text-text-secondary mt-1">Run vehicle, plate, and character detection on real images or videos, then validate every decoded result.</p>
      </div>

      <PipelineStrip
        vehicle={!healthChecked ? "checking" : !health ? "offline" : "ready"}
        plate={!healthChecked ? "checking" : !health ? "offline" : health.models_loaded.detection ? "ready" : "missing"}
        character={!healthChecked ? "checking" : !health ? "offline" : health.models_loaded.character ? "ready" : "missing"}
      />

      <div className="bg-bg-card border border-border rounded-xl p-5 grid grid-cols-1 md:grid-cols-2 gap-5">
        <label className="text-xs text-text-muted">Minimum plate confidence <span className="float-right font-mono text-text-primary">{Math.round(confidence * 100)}%</span><input type="range" min={0.05} max={0.95} step={0.05} value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} className="w-full mt-3 accent-indigo-500" /></label>
        <label className="text-xs text-text-muted">Video inference stride <span className="float-right font-mono text-text-primary">Every {frameStride} frame{frameStride > 1 ? "s" : ""}</span><input type="range" min={1} max={15} step={1} value={frameStride} onChange={(event) => setFrameStride(Number(event.target.value))} className="w-full mt-3 accent-indigo-500" /></label>
      </div>

      <FileUploader onFile={handleFile} />
      {loading && <LoadingSpinner text="Preparing media for inference..." />}
      {error && <div className="flex items-center gap-3 p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger"><AlertTriangle className="w-5 h-5 shrink-0" /><p className="text-sm">{error}</p></div>}

      {videoJob && (
        <section className="bg-bg-card border border-border rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><FileVideo className="w-5 h-5 text-info" /><div><h2 className="text-base font-semibold text-text-primary">Video inference</h2><p className="text-xs text-text-muted capitalize">{videoJob.status} · {videoJob.processed_frames}/{videoJob.total_frames || "?"} frames</p></div></div><span className="text-sm font-mono text-text-primary">{Math.round(videoJob.progress * 100)}%</span></div>
          <div className="w-full h-2.5 rounded-full bg-bg-primary overflow-hidden"><div className={`h-full rounded-full transition-all duration-500 ${videoJob.status === "error" ? "bg-danger" : videoJob.status === "completed" ? "bg-success" : "bg-info"}`} style={{ width: `${videoJob.progress * 100}%` }} /></div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3"><div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Processed frames</p><p className="text-lg font-mono text-text-primary mt-1">{videoJob.processed_frames}</p></div><div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Detection frames</p><p className="text-lg font-mono text-text-primary mt-1">{videoJob.frames_with_detections}</p></div><div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Plate observations</p><p className="text-lg font-mono text-text-primary mt-1">{videoJob.total_detections}</p></div><div className="bg-bg-primary rounded-lg p-3"><p className="text-xs text-text-muted">Review events</p><p className="text-lg font-mono text-text-primary mt-1">{videoJob.events.length}</p></div></div>
          {videoJob.status === "completed" && videoJob.result_url && <div className="space-y-3"><video src={videoJob.result_url} controls className="w-full max-h-[620px] rounded-xl bg-black" /><a href={videoJob.result_url} download className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-white text-sm"><Download className="w-4 h-4" /> Download annotated video</a></div>}
          {videoJob.events.length > 0 && <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b border-border text-text-muted"><th className="py-2 pr-4">Time</th><th className="py-2 pr-4">Frame</th><th className="py-2 pr-4">Detections</th><th className="py-2 pr-4">Character / OCR text</th><th className="py-2">Human decision</th></tr></thead><tbody>{videoJob.events.map((event) => { const reviewKey = `video-frame-${event.frame}`; return <tr key={event.frame} className="border-b border-border/60"><td className="py-2 pr-4 font-mono text-text-primary">{event.time_seconds.toFixed(2)}s</td><td className="py-2 pr-4 text-text-secondary">{event.frame}</td><td className="py-2 pr-4 text-text-secondary">{event.detections}</td><td className="py-2 pr-4 font-mono text-text-secondary" dir="auto">{event.plates.join(", ") || "—"}</td><td className="py-2"><div className="flex gap-1"><button aria-label={`Accept event at ${event.time_seconds} seconds`} onClick={() => setReviews((current) => ({ ...current, [reviewKey]: "accepted" }))} className={`px-2 py-1 rounded text-xs ${reviews[reviewKey] === "accepted" ? "bg-success text-white" : "bg-bg-primary text-text-muted hover:text-success"}`}>Accept</button><button aria-label={`Flag event at ${event.time_seconds} seconds`} onClick={() => { setReviews((current) => ({ ...current, [reviewKey]: "needs_review" })); setReviewFrame(event); }} className={`px-2 py-1 rounded text-xs ${reviews[reviewKey] === "needs_review" ? "bg-warning text-white" : "bg-bg-primary text-text-muted hover:text-warning"}`}>Review</button><button aria-label={`Reject event at ${event.time_seconds} seconds`} onClick={() => setReviews((current) => ({ ...current, [reviewKey]: "rejected" }))} className={`px-2 py-1 rounded text-xs ${reviews[reviewKey] === "rejected" ? "bg-danger text-white" : "bg-bg-primary text-text-muted hover:text-danger"}`}>Reject</button></div></td></tr>; })}</tbody></table></div>}
          {videoJob.events.length > 0 && <button onClick={exportReview} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-bg-primary text-sm text-text-secondary hover:text-text-primary"><Download className="w-4 h-4" /> Export video review JSON</button>}
        </section>
      )}

      {!loading && detections.length > 0 && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4"><StatsGrid icon={<ScanBox className="w-4 h-4" />} label="Plates detected" value={detections.length} color="text-accent" /><StatsGrid icon={<Clock className="w-4 h-4" />} label="Processing time" value={timeMs !== null ? `${timeMs.toFixed(0)}ms` : "—"} color="text-info" /><StatsGrid icon={<Zap className="w-4 h-4" />} label="Average confidence" value={`${(averageConfidence * 100).toFixed(1)}%`} color="text-success" /><StatsGrid icon={<ShieldCheck className="w-4 h-4" />} label="Human reviewed" value={`${reviewedCount}/${detections.length}`} color={reviewedCount === detections.length ? "text-success" : "text-warning"} /></div>
          {annotatedImage && <section className="bg-bg-card border border-border rounded-xl overflow-hidden"><div className="px-5 py-3 border-b border-border"><h2 className="text-sm font-semibold text-text-primary">Annotated pipeline output</h2></div><img src={annotatedImage} alt="Annotated vehicle and plate detections" className="w-full max-h-[680px] object-contain bg-bg-primary" /></section>}
          <section><div className="flex items-center justify-between gap-4 mb-4"><h2 className="text-lg font-semibold text-text-primary">Human verification {mediaName && <span className="text-text-muted font-normal text-sm">— {mediaName}</span>}</h2><button onClick={exportReview} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-bg-card text-sm text-text-secondary hover:text-text-primary"><Download className="w-4 h-4" /> Export review JSON</button></div><div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{detections.map((detection, index) => <PlateCard key={detection.id || index} detection={detection} index={index} review={reviews[detection.id]} onReview={(decision) => setReviews((current) => ({ ...current, [detection.id]: decision }))} />)}</div></section>
        </>
      )}

      {!loading && !videoJob && detections.length === 0 && !error && (
        <div className="text-center py-12 text-text-muted">
          <CheckCircle2 className="w-11 h-11 mx-auto mb-3 opacity-25" />
          <p>{mediaName ? "No license plate was found above the selected confidence threshold." : "Upload media to begin an auditable three-stage inference session."}</p>
          {mediaName && <p className="text-xs mt-2">Try lowering the plate confidence or verify that the plate checkpoint is loaded.</p>}
        </div>
      )}

      {reviewFrame && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={() => setReviewFrame(null)}>
          <div className="bg-bg-card border border-border rounded-2xl max-w-3xl w-full mx-4 overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
              <div>
                <h3 className="text-sm font-semibold text-text-primary">Frame {reviewFrame.frame}</h3>
                <p className="text-xs text-text-muted">{reviewFrame.time_seconds.toFixed(2)}s · {reviewFrame.detections} detection{reviewFrame.detections !== 1 ? "s" : ""} · {reviewFrame.plates.join(", ") || "No plates"}</p>
              </div>
              <button onClick={() => setReviewFrame(null)} className="p-1 rounded hover:bg-bg-primary text-text-muted hover:text-text-primary"><X className="w-5 h-5" /></button>
            </div>
            {reviewFrame.frame_url && <img src={reviewFrame.frame_url} alt={`Annotated frame ${reviewFrame.frame}`} className="w-full max-h-[520px] object-contain bg-black" />}
          </div>
        </div>
      )}
    </div>
  );
}
