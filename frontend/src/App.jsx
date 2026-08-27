import { useState } from "react";
import CaseSubmission from "./pages/CaseSubmission.jsx";
import Processing from "./pages/Processing.jsx";
import Report from "./pages/Report.jsx";
import Review from "./pages/Review.jsx";
import { diagnoseCase } from "./services/api.js";

// Simple state-based routing between the screens -- avoids adding
// react-router-dom for an app this small. All state lives in React
// state only, nothing in localStorage/sessionStorage.
const SCREENS = {
  SUBMISSION: "submission",
  PROCESSING: "processing",
  REPORT: "report",
  REVIEW: "review",
};

export default function App() {
  const [screen, setScreen] = useState(SCREENS.SUBMISSION);
  const [report, setReport] = useState(null);
  const [submittedSummary, setSubmittedSummary] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(formData, summary) {
    setError(null);
    setSubmittedSummary(summary);
    setScreen(SCREENS.PROCESSING);
    try {
      const result = await diagnoseCase(formData);
      setReport(result);
      // Reachable automatically when review is required, not just via
      // the manual "Review Case" button on the Report screen.
      setScreen(result.review_required ? SCREENS.REVIEW : SCREENS.REPORT);
    } catch (err) {
      setError(err.message || "Something went wrong while analyzing this case.");
      setScreen(SCREENS.SUBMISSION);
    }
  }

  function handleNewCase() {
    setReport(null);
    setSubmittedSummary(null);
    setError(null);
    setScreen(SCREENS.SUBMISSION);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <h1 className="text-xl font-semibold tracking-tight text-slate-800">MedOrchestrate</h1>
          <p className="text-sm text-slate-500">Clinical decision-support prototype — for clinician review only</p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {screen === SCREENS.SUBMISSION && <CaseSubmission onSubmit={handleSubmit} error={error} />}

        {screen === SCREENS.PROCESSING && <Processing />}

        {screen === SCREENS.REPORT && report && (
          <Report
            report={report}
            submittedSummary={submittedSummary}
            onNewCase={handleNewCase}
            onReviewCase={() => setScreen(SCREENS.REVIEW)}
          />
        )}

        {screen === SCREENS.REVIEW && report && (
          <Review report={report} onBack={() => setScreen(SCREENS.REPORT)} onReviewed={() => setScreen(SCREENS.REPORT)} />
        )}
      </main>
    </div>
  );
}
