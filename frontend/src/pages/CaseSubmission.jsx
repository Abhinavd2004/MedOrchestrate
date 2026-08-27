import { useRef, useState } from "react";

const SEX_OPTIONS = ["male", "female", "other"];

/**
 * onSubmit(formData, submittedSummary) -- formData is ready to POST to
 * /diagnose as multipart/form-data; submittedSummary is a plain object
 * ({age, sex, clinicalText, hadImage}) the parent keeps in React state
 * so the Report screen can show a patient summary / note that an image
 * was included -- FinalReport itself doesn't echo back patient details
 * or imaging findings, so the app remembers what was submitted instead
 * of inventing data that isn't in the API response.
 */
export default function CaseSubmission({ onSubmit, error }) {
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [clinicalText, setClinicalText] = useState("");
  const [medicalReport, setMedicalReport] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [formError, setFormError] = useState(null);
  const fileInputRef = useRef(null);

  function handleImageChange(event) {
    const file = event.target.files?.[0] ?? null;
    setImageFile(file);
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImagePreviewUrl(file ? URL.createObjectURL(file) : null);
  }

  function handleRemoveImage() {
    setImageFile(null);
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImagePreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleSubmit(event) {
    event.preventDefault();
    setFormError(null);

    if (!age || !sex) {
      setFormError("Age and sex are required.");
      return;
    }
    if (!clinicalText.trim() && !medicalReport.trim() && !imageFile) {
      setFormError("Provide at least clinical text, a medical report, or an image.");
      return;
    }

    const formData = new FormData();
    formData.append("age", age);
    formData.append("sex", sex);
    if (clinicalText.trim()) formData.append("clinical_text", clinicalText.trim());
    if (medicalReport.trim()) formData.append("medical_report", medicalReport.trim());
    if (imageFile) formData.append("image", imageFile);

    onSubmit(formData, {
      age,
      sex,
      clinicalText: clinicalText.trim(),
      hadImage: Boolean(imageFile),
    });
  }

  const displayedError = formError || error;

  return (
    <div className="mx-auto max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-800">New case</h2>
      <p className="mt-1 text-sm text-slate-500">
        Enter whatever information is available — clinical text, a medical report, an image, or any
        combination.
      </p>

      {displayedError && (
        <div
          role="alert"
          className="mt-4 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {displayedError}
        </div>
      )}

      <form onSubmit={handleSubmit} className="mt-6 space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700" htmlFor="age">
              Age
            </label>
            <input
              id="age"
              type="number"
              min="0"
              max="120"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700" htmlFor="sex">
              Sex
            </label>
            <select
              id="sex"
              value={sex}
              onChange={(e) => setSex(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">Select…</option>
              {SEX_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option.charAt(0).toUpperCase() + option.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="clinical_text">
            Clinical history / presenting complaint
          </label>
          <textarea
            id="clinical_text"
            rows={4}
            value={clinicalText}
            onChange={(e) => setClinicalText(e.target.value)}
            placeholder="e.g. headache and dizziness, history of hypertension"
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="medical_report">
            Medical report / MRI report text
          </label>
          <textarea
            id="medical_report"
            rows={4}
            value={medicalReport}
            onChange={(e) => setMedicalReport(e.target.value)}
            placeholder="Paste relevant report text here, if available"
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="image">
            Medical image (optional)
          </label>
          <input
            id="image"
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="mt-1 block w-full text-sm text-slate-600 file:mr-4 file:rounded-md file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
          />
          {imagePreviewUrl && (
            <div className="mt-3 flex items-center gap-3">
              <img
                src={imagePreviewUrl}
                alt="Selected medical image preview"
                className="h-24 w-24 rounded-md border border-slate-200 object-cover"
              />
              <button
                type="button"
                onClick={handleRemoveImage}
                className="text-sm text-slate-500 underline hover:text-slate-700"
              >
                Remove image
              </button>
            </div>
          )}
        </div>

        <button
          type="submit"
          className="w-full rounded-md bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Analyze Case
        </button>
      </form>
    </div>
  );
}
