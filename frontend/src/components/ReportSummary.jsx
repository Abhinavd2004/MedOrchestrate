import ConfidenceBar from "./ConfidenceBar.jsx";

/**
 * The data-display half of a FinalReport -- confidence, conflicts,
 * differential diagnoses, and supporting evidence. Shared between
 * pages/Report.jsx and pages/Review.jsx so the review screen shows
 * exactly the same underlying data the clinician already saw on the
 * report, not a re-derived summary.
 *
 * The "Review Case" entry point itself lives in Report.jsx's header
 * (always visible, on any report -- reviewing isn't restricted to
 * low-confidence cases); this component just surfaces an urgent notice
 * when report.review_required is true, without its own separate button.
 */
export default function ReportSummary({ report }) {
  const diagnoses = [...(report.diagnoses ?? [])].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  const ragEvidence = (report.evidence ?? []).filter((item) => item.type === "rag");
  const liveEvidence = (report.evidence ?? []).filter((item) => item.type === "live");
  const conflicts = report.conflicts ?? [];

  return (
    <>
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <ConfidenceBar confidence={report.confidence} label="Overall confidence" />
        {report.review_required && (
          <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-3">
            <p className="text-sm text-amber-800">
              Confidence is below the review threshold — this case requires clinician review before
              acting on it.
            </p>
          </div>
        )}
      </section>

      {conflicts.length > 0 && (
        <section role="alert" className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-red-800">
            <WarningIcon /> Conflicting evidence
          </h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-red-700">
            {conflicts.map((conflict, index) => (
              <li key={index}>{conflict}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-6">
        <h3 className="text-sm font-semibold text-slate-700">Differential diagnoses</h3>
        <div className="mt-3 space-y-3">
          {diagnoses.length === 0 && (
            <p className="text-sm text-slate-500">No diagnoses were produced for this case.</p>
          )}
          {diagnoses.map((diagnosis, index) => (
            <div key={index} className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    Rank {diagnosis.rank ?? index + 1}
                  </span>
                  <p className="text-base font-medium text-slate-800">{diagnosis.name}</p>
                </div>
              </div>
              <div className="mt-2">
                <ConfidenceBar confidence={diagnosis.confidence} size="sm" />
              </div>
              {diagnosis.supporting_evidence?.length > 0 ? (
                <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
                  {diagnosis.supporting_evidence.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-3 text-sm italic text-slate-400">No cited supporting evidence.</p>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="mt-6">
        <h3 className="text-sm font-semibold text-slate-700">Supporting evidence</h3>

        {ragEvidence.length > 0 && (
          <div className="mt-3">
            <h4 className="text-xs font-medium uppercase tracking-wide text-slate-400">
              Offline knowledge base
            </h4>
            <ul className="mt-2 space-y-2">
              {ragEvidence.map((item, index) => (
                <li key={index} className="rounded-md border border-slate-200 bg-white p-3 text-sm">
                  <p className="text-slate-700">{item.text}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    source: {item.source} · relevance: {Math.round((item.score ?? 0) * 100)}%
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}

        {liveEvidence.length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs font-medium uppercase tracking-wide text-slate-400">Live evidence</h4>
            <ul className="mt-2 space-y-2">
              {liveEvidence.map((item, index) => (
                <li key={index} className="rounded-md border border-slate-200 bg-white p-3 text-sm">
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-blue-700 hover:underline"
                  >
                    {item.title}
                  </a>
                  <p className="mt-1 text-slate-600">{item.summary}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    {item.source}
                    {item.publication_date ? ` · ${item.publication_date}` : ""} · evidence level:{" "}
                    {item.evidence_level}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}

        {ragEvidence.length === 0 && liveEvidence.length === 0 && (
          <p className="mt-3 text-sm text-slate-500">No supporting evidence was retrieved for this case.</p>
        )}
      </section>
    </>
  );
}

function WarningIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4 flex-none">
      <path
        fillRule="evenodd"
        d="M8.257 3.099c.765-1.36 2.72-1.36 3.486 0l6.28 11.18c.75 1.334-.213 2.987-1.744 2.987H3.72c-1.53 0-2.493-1.653-1.743-2.987l6.28-11.18zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-.25-6.25a.75.75 0 00-1.5 0v3.5a.75.75 0 001.5 0v-3.5z"
        clipRule="evenodd"
      />
    </svg>
  );
}
