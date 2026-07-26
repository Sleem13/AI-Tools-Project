import { useCallback, useEffect, useRef, useState } from "react";
import { FileVideo, Image as ImageIcon, Upload, X } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
}

export default function FileUploader({ onFile }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const handle = useCallback(
    (file: File) => {
      const type = file.type.startsWith("image/") ? "image" : file.type.startsWith("video/") ? "video" : null;
      if (!type) return;
      setPreview((current) => {
        if (current) URL.revokeObjectURL(current);
        return URL.createObjectURL(file);
      });
      setMediaType(type);
      onFile(file);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files[0];
      if (file) handle(file);
    },
    [handle],
  );

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setMediaType(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border border-border bg-bg-card">
        {mediaType === "video" ? (
          <video src={preview} controls className="w-full max-h-[560px] bg-black" />
        ) : (
          <img src={preview} alt="Uploaded media" className="w-full max-h-[560px] object-contain" />
        )}
        <button onClick={clear} aria-label="Clear uploaded media" className="absolute top-3 right-3 p-2 rounded-lg bg-bg-primary/90 hover:bg-danger/80 text-text-primary transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-4 p-14 rounded-xl border-2 border-dashed cursor-pointer transition-all ${dragOver ? "border-accent bg-accent/10" : "border-border bg-bg-card hover:border-accent/50 hover:bg-bg-hover"}`}
    >
      <div className="flex items-center gap-2">
        <div className="p-3 rounded-full bg-accent/10"><ImageIcon className="w-7 h-7 text-accent" /></div>
        <div className="p-3 rounded-full bg-info/10"><FileVideo className="w-7 h-7 text-info" /></div>
      </div>
      <div className="text-center">
        <p className="text-text-primary font-medium">{dragOver ? "Drop media here" : "Upload an image or video"}</p>
        <p className="text-sm text-text-muted mt-1">JPG, PNG, BMP, MP4, MOV, AVI, MKV or WebM · videos up to 512 MB</p>
      </div>
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium"><Upload className="w-4 h-4" /> Browse media</div>
      <input ref={inputRef} type="file" accept="image/*,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) handle(file); }} />
    </div>
  );
}
