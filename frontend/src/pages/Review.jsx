import { useState } from "react";
import ReportSummary from "../components/ReportSummary.jsx";
import { submitReview } from "../services/api.js";

const DECISIONS = [
  { value: "accept", label: "Accept", description: "The differential above is correct as presented." },
  { value: "override", label: "Override", description: "Replace the top diagnosis with a different one." },
  { value: "annotate", label: "Annotate", description: "Leave a note without changing the diagnosis." },
];

/**
 * Full HITL review screen: shows the same report data (confidence,
 * conflicts, diagnoses, evidence) via the shared ReportSummary, plus
 * the three review actions. Reachable both automatically (App.jsx
 * routes here when report.review_required is true) and manually (the
 * "Review Case" button on the Report screen).
 *
 * "Override" has no dedicated backend field for the replacement
 * diagnosis -- ReviewRequest only has {decision, annotation, reviewer}
 * -- so the chosen/typed diagnosis is written into `annotation` as
 * "Override diagnosis: <name>", matching the convention documented in
 * backend/api/schemas.py.
 */
export default function Review({ report, onBack, onReviewed }) {
  const [decision, setDecision] = useState("accept");
  const [overrideDiagnosis, setOverrideDiagnosis] = useState("");
  const [annotationNote, setAnnotationNote] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [confirmation, setConfirmation] = useState(null);

  const diagnoses = [...(report.diagnoses ?? [])].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);

    if (!reviewer.trim()) {
      setError("Reviewer name is required.");
      return;
    }

    let annotation = null;
    if (decision === "override") {
      if (!overrideDiagnosis.trim()) {
        setError("Pick or type the diagnosis you're overriding to.");
        return;
      }
      annotation = `Override diagnosis: ${overrideDiagnosis.trim()}`;
    } else if (decision === "annotate") {
      if (!annotationNote.trim()) {
        setError("An annotation note is required.");
        return;
      }
      annotation = annotationNote.trim();
    }

    setSubmitting(true);
    try {
      const result = await submitReview(report.case_id, { decision, annotation, reviewer: reviewer.trim() });
      setConfirmation(result);
    } catch (err) {
      setError(err.message || "Failed to submit the review.");
    } finally {
      setSubmitting(false);
    }
  }

  if (confirmation) {
    return (
      <div className="mx-auto max-w-2xl">
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-6 text-center">
          <h2 className="text-lg font-semibold text-emerald-800">Review recorded</h2>
          <p className="mt-2 text-sm text-emerald-700">
            {confirmation.reviewer} recorded "{confirmation.decision}" for case {confirmation.case_id}.
          </p>
          {confirmation.annotation && (
            <p className="mt-2 rounded-md bg-white px-3 py-2 text-sm text-slate-600">{confirmation.annotation}</p>
          )}
        </div>
        <button
          type="button"
          onClick={onReviewed}
          className="mt-4 w-full rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Back to report
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">Review case</h2>
          <p className="mt-1 text-sm text-slate-500">Case {report.case_id}</p>
        </div>
        <button
          type="button"
          onClick={onBack}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Back to report
        </button>
      </div>

      <div className="mt-6">
        <ReportSummary report={report} />
      </div>

      <section className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="text-sm font-semibold text-slate-700">Clinician decision</h3>

        {error && (
          <div role="alert" className="mt-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-4 space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            {DECISIONS.map((option) => (
              <label
                key={option.value}
                className={`cursor-pointer rounded-md border p-3 text-sm transition ${
                  decision === option.value
                    ? "border-blue-500 bg-blue-50"
                    : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <input
                  type="radio"
                  name="decision"
                  value={option.value}
                  checked={decision === option.value}
                  onChange={(e) => setDecision(e.target.value)}
                  className="sr-only"
                />
                <span className="block font-medium text-slate-800">{option.label}</span>
                <span className="mt-1 block text-xs text-slate-500">{option.description}</span>
              </label>
            ))}
          </div>

          {decision === "override" && (
            <div>
              <span className="block text-sm font-medium text-slate-700">Replacement diagnosis</span>
              {diagnoses.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {diagnoses.map((d, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => setOverrideDiagnosis(d.name)}
                      className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:border-blue-400 hover:text-blue-700"
                    >
                      {d.name}
                    </button>
                  ))}
                </div>
              )}
              <input
                type="text"
                value={overrideDiagnosis}
                onChange={(e) => setOverrideDiagnosis(e.target.value)}
                placeholder="Pick a diagnosis above, or type your own"
                className="mt-2 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          )}

          {decision === "annotate" && (
            <div>
              <label className="block text-sm font-medium text-slate-700" htmlFor="annotation_note">
                Note
              </label>
              <textarea
                id="annotation_note"
                rows={3}
                value={annotationNote}
                onChange={(e) => setAnnotationNote(e.target.value)}
                placeholder="e.g. Recommend follow-up MRI in 3 months."
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-slate-700" htmlFor="reviewer">
              Reviewer name
            </label>
            <input
              id="reviewer"
              type="text"
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="Dr. …"
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-md bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {submitting ? "Submitting…" : "Submit review"}
          </button>
        </form>
      </section>
    </div>
  );
}
