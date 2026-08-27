# Synthetic test cases

All patients, symptoms, and history below are **synthetic/de-identified**
-- authored for this prototype, not real patient data. Each file is a
valid `CaseInput` JSON object (see `backend/models/schemas.py`) and can
be loaded directly:

```python
import json
from backend.models.schemas import CaseInput

case = CaseInput(**json.load(open("data/test_cases/CASE002_text_only.json")))
```

Or POSTed to `/diagnose` as multipart form fields (see
`backend/api/routes.py` / `frontend/src/services/api.js` for the exact
field mapping) for a manual/demo run through the real pipeline.

## Matrix coverage

These fixtures are input data only -- outcomes like "high confidence"
or "optimizer runs" depend on what the live pipeline (Claude, RAG,
PubMed) actually returns at request time, which varies. The mapping
below is what each case is *designed* to exercise, not a guaranteed
runtime outcome. For deterministic, guaranteed coverage of every
matrix item (confidence tiers, failure modes, iteration caps), see
`backend/tests/test_integration.py`, which mocks the external boundary
precisely to force each scenario.

| File | Exercises |
|---|---|
| `CASE001.json` | Text only (original Phase 0 fixture) |
| `CASE001_with_image.json` | Text + image (original Phase 0 fixture) |
| `CASE002_text_only.json` | Text only |
| `CASE003_image_only.json` | Image only, no text/report |
| `CASE004_text_and_image.json` | Text + MRI image |
| `CASE005_text_and_medical_report.json` | Text + MRI report (text), both fields |
| `CASE006_medical_report_only.json` | Medical report text only, no clinical_text |
| `CASE007_high_confidence_candidate.json` | Symptoms closely matching the offline knowledge base's terms -- a plausible high-confidence, no-optimizer case |
| `CASE008_low_confidence_candidate.json` | Vague, non-specific symptoms -- a plausible low-confidence, optimizer-triggering case |
| `CASE009_elderly_patient.json` | Elderly patient, anticoagulation + fall history (chronic subdural hematoma differential) |
| `CASE010_pediatric_patient.json` | Pediatric patient |
| `CASE011_with_labs.json` | Populated `labs` dict (anemia-related) |
| `CASE012_extensive_history.json` | Long `history` list, multiple comorbidities |
| `CASE013_minimal_no_content.json` | No clinical_text, no medical_report, no image -- only age/sex (edge case) |
| `CASE014_headache_focused.json` | Headache-predominant presentation (migraine) |
| `CASE015_dizziness_focused.json` | Dizziness-predominant presentation (BPPV) |
| `CASE016_multiple_symptoms_with_image.json` | Multiple symptoms + image + extensive history combined |

## Failure-mode and iteration-cap scenarios

RAG/Qdrant unreachable, PubMed/MedlinePlus unreachable, the optimizer's
3-iteration cap, and invalid-image-upload validation are **behavioral**
scenarios (they depend on simulating an outage or malformed input, not
on case content) -- these are covered by dedicated tests in
`backend/tests/test_integration.py` rather than static fixture files
here, since a JSON file can't itself represent "Qdrant is down."
