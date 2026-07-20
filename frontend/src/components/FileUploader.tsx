import { useCallback, useRef, useState } from "react";
import { Upload, X, Image as ImageIcon } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
}

export default function FileUploader({ onFile }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      setPreview(url);
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handle(f);
    },
    [handle]
  );

  const clear = () => {
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border border-border bg-bg-card">
        <img
          src={preview}
          alt="Uploaded"
          className="w-full max-h-[500px] object-contain"
        />
        <button
          onClick={clear}
          className="absolute top-3 right-3 p-1.5 rounded-lg bg-bg-primary/80 hover:bg-danger/80 text-text-primary transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-4 p-16 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
        dragOver
          ? "border-accent bg-accent/10"
          : "border-border bg-bg-card hover:border-accent/50 hover:bg-bg-hover"
      }`}
    >
      <div className="p-4 rounded-full bg-accent/10">
        {dragOver ? (
          <ImageIcon className="w-8 h-8 text-accent" />
        ) : (
          <Upload className="w-8 h-8 text-accent" />
        )}
      </div>
      <div className="text-center">
        <p className="text-text-primary font-medium">
          {dragOver ? "Drop image here" : "Upload vehicle image"}
        </p>
        <p className="text-sm text-text-muted mt-1">
          Drag & drop or click to browse &middot; JPG, PNG, BMP
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handle(f);
        }}
      />
    </div>
  );
}
