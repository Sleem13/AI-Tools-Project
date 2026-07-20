import { Loader2 } from "lucide-react";

export default function LoadingSpinner({ text = "Processing..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center gap-3 py-12">
      <Loader2 className="w-8 h-8 text-accent animate-spin" />
      <p className="text-sm text-text-secondary">{text}</p>
    </div>
  );
}
