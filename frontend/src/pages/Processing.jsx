import { useEffect, useState } from "react";

const STAGES = ["Clinical processing", "Medical imaging", "Biomedical RAG", "Current evidence", "Clinical fusion"];

const STAGE_DURATION_MS = 1400;
const MAX_OPTIMIZER_ITERATIONS = 3;

/**
 * Phase 9's /diagnose is fully synchronous -- there is no real
 * progress signal from the server while the request is in flight, so
 * this is a purely client-side illustrative animation: it steps
 * through the named stages on a timer, and if the wait continues past
 * all of them, cycles an "adaptive optimizer" indicator so the screen
 * doesn't look stalled. It does not reflect what the backend is
 * actually doing at any given moment.
 */
export default function Processing() {
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const [optimizerIteration, setOptimizerIteration] = useState(null);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStageIndex((current) => {
        const next = current + 1;
        if (next < STAGES.length) {
          return next;
        }
        setOptimizerIteration((iter) => (iter === null || iter >= MAX_OPTIMIZER_ITERATIONS ? 1 : iter + 1));
        return current;
      });
    }, STAGE_DURATION_MS);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="mx-auto max-w-md">
      <h2 className="text-lg font-semibold text-slate-800">Analyzing case</h2>
      <p className="mt-1 text-sm text-slate-500">
        This can take a little while — the pipeline is working through each stage below.
      </p>

      <ul className="mt-6 space-y-3">
        {STAGES.map((stage, index) => {
          const isDone = index < activeStageIndex;
          const isActive = index === activeStageIndex;
          return (
            <li key={stage} className="flex items-center gap-3">
              <StageIcon done={isDone} active={isActive} />
              <span
                className={
                  isDone
                    ? "text-sm text-slate-400 line-through"
                    : isActive
                      ? "text-sm font-medium text-slate-800"
                      : "text-sm text-slate-400"
                }
              >
                {stage}
              </span>
            </li>
          );
        })}
      </ul>

      {optimizerIteration !== null && (
        <div className="mt-6 flex items-center gap-3 rounded-md border border-blue-200 bg-blue-50 px-4 py-3">
          <Spinner />
          <span className="text-sm font-medium text-blue-800">
            Adaptive optimizer running… Iteration {optimizerIteration}/{MAX_OPTIMIZER_ITERATIONS}
          </span>
        </div>
      )}
    </div>
  );
}

function StageIcon({ done, active }) {
  if (done) {
    return (
      <span className="flex h-5 w-5 flex-none items-center justify-center rounded-full bg-emerald-500 text-white">
        <svg viewBox="0 0 20 20" fill="currentColor" className="h-3 w-3">
          <path
            fillRule="evenodd"
            d="M16.7 5.3a1 1 0 010 1.4l-7 7a1 1 0 01-1.4 0l-3-3a1 1 0 111.4-1.4L9 11.6l6.3-6.3a1 1 0 011.4 0z"
            clipRule="evenodd"
          />
        </svg>
      </span>
    );
  }
  if (active) {
    return <Spinner small />;
  }
  return <span className="h-5 w-5 flex-none rounded-full border-2 border-slate-200" />;
}

function Spinner({ small }) {
  const size = small ? "h-5 w-5" : "h-4 w-4";
  return <span className={`${size} flex-none animate-spin rounded-full border-2 border-blue-200 border-t-blue-600`} />;
}
