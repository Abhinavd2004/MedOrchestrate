// Confidence tiers: green >= THRESHOLD (matches backend/services/confidence.py's
// THRESHOLD=0.65), amber for a "borderline, worth a second look" middle
// band, red below that. Mirrors the backend's threshold rather than
// inventing a separate one client-side.
const THRESHOLD = 0.65;
const AMBER_FLOOR = 0.4;

function tierFor(confidence) {
  if (confidence >= THRESHOLD) {
    return { label: "high", bar: "bg-emerald-500", text: "text-emerald-700", chip: "bg-emerald-50 text-emerald-700 border-emerald-200" };
  }
  if (confidence >= AMBER_FLOOR) {
    return { label: "moderate", bar: "bg-amber-500", text: "text-amber-700", chip: "bg-amber-50 text-amber-700 border-amber-200" };
  }
  return { label: "low", bar: "bg-red-500", text: "text-red-700", chip: "bg-red-50 text-red-700 border-red-200" };
}

/**
 * confidence: a 0-1 float.
 * size: "sm" (thin bar, for per-diagnosis rows) or "md" (default, for
 * the overall-confidence summary).
 */
export default function ConfidenceBar({ confidence, size = "md", label }) {
  const clamped = Math.max(0, Math.min(1, confidence ?? 0));
  const percent = Math.round(clamped * 100);
  const tier = tierFor(clamped);
  const height = size === "sm" ? "h-1.5" : "h-2.5";

  return (
    <div className="w-full">
      <div className="flex items-center justify-between gap-2">
        {label && <span className="text-sm font-medium text-slate-700">{label}</span>}
        <span className={`text-sm font-semibold ${tier.text}`}>{percent}%</span>
      </div>
      <div className={`mt-1 w-full overflow-hidden rounded-full bg-slate-100 ${height}`}>
        <div
          className={`h-full rounded-full ${tier.bar} transition-all`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export { tierFor };
