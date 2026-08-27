import ReportSummary from "../components/ReportSummary.jsx";

/**
 * report: the FinalReport JSON from POST/GET /diagnose.
 * submittedSummary: { age, sex, clinicalText, hadImage } echoed from
 * what was submitted on the Case Submission screen -- FinalReport
 * itself doesn't include patient details or a separate ImagingFindings
 * object, so this app remembers what was submitted rather than
 * inventing data that isn't in the API response.
 * onNewCase / onReviewCase: navigation callbacks from App.jsx.
 */
export default function Report({ report, submittedSummary, onNewCase, onReviewCase }) {
  return (
    <div className="mx-auto max-w-3xl">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">Clinician report</h2>
          <p className="mt-1 text-sm text-slate-500">Case {report.case_id}</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onReviewCase}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Review Case
          </button>
          <button
            type="button"
            onClick={onNewCase}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            New case
          </button>
        </div>
      </div>

      {submittedSummary && (
        <section className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-700">Patient summary</h3>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <dt className="text-slate-500">Age</dt>
            <dd className="text-slate-800">{submittedSummary.age || "—"}</dd>
            <dt className="text-slate-500">Sex</dt>
            <dd className="text-slate-800">{submittedSummary.sex || "—"}</dd>
          </dl>
          {submittedSummary.clinicalText && (
            <p className="mt-2 text-sm text-slate-600">{submittedSummary.clinicalText}</p>
          )}
          {submittedSummary.hadImage && (
            <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              A medical image was included with this case and analyzed as part of the evidence
              fusion below.
            </div>
          )}
        </section>
      )}

      <div className="mt-6">
        <ReportSummary report={report} />
      </div>
    </div>
  );
}
