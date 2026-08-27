"""API routes.

POST /diagnose calls orchestrator.run_case() SYNCHRONOUSLY and returns
the FinalReport directly in the response body (not a
{"status": "processing"} placeholder) -- this is the simplest version
that proves the pipeline works end to end over HTTP. Async
polling (kick off run_case() in a background task/worker, let clients
poll GET /diagnose/{case_id} until it's ready) is a natural fast-follow
once this works, not implemented here.
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.api.schemas import ReviewRequest, ReviewResponse
from backend.models.schemas import CaseInput, FinalReport, Patient
from backend.services import storage
from backend.services.logging import log_event
from backend.services.orchestrator import run_case

router = APIRouter()

UPLOAD_DIR = Path("data/images/uploads")


def _parse_json_field(raw: str | None, field_name: str, default):
    if raw is None or raw.strip() == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"'{field_name}' must be valid JSON: {exc}"
        ) from exc


def _save_uploaded_image(case_id: str, image: UploadFile) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "upload").suffix or ".png"
    dest = UPLOAD_DIR / f"{case_id}{suffix}"
    dest.write_bytes(image.file.read())
    return str(dest)


@router.post("/diagnose", response_model=FinalReport)
def diagnose(
    case_id: str | None = Form(None),
    age: int = Form(...),
    sex: str = Form(...),
    clinical_text: str | None = Form(None),
    medical_report: str | None = Form(None),
    labs: str | None = Form(None),
    history: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> FinalReport:
    """multipart/form-data so an optional image file can travel alongside
    the JSON-ish case fields in one request. `labs` and `history` are
    JSON-encoded strings (a dict and a list of strings respectively) --
    multipart form fields are flat, so nested/array values are passed
    as JSON text and parsed here, with a 422 on malformed JSON.

    Deliberately a plain `def` (not `async def`): run_case() is a
    synchronous, potentially slow pipeline, and FastAPI runs sync route
    handlers in a threadpool -- keeping this sync avoids blocking the
    event loop for the whole request, which `await`-ing a slow sync
    call from an async def would otherwise do.
    """
    resolved_case_id = case_id or str(uuid.uuid4())

    labs_dict = _parse_json_field(labs, "labs", {})
    if not isinstance(labs_dict, dict):
        raise HTTPException(status_code=422, detail="'labs' must be a JSON object")

    history_list = _parse_json_field(history, "history", [])
    if not isinstance(history_list, list) or not all(isinstance(h, str) for h in history_list):
        raise HTTPException(status_code=422, detail="'history' must be a JSON array of strings")

    image_path: str | None = None
    if image is not None and image.filename:
        try:
            image_path = _save_uploaded_image(resolved_case_id, image)
        except Exception as exc:
            log_event("api_error", endpoint="/diagnose", case_id=resolved_case_id, error=str(exc))
            raise HTTPException(status_code=500, detail="Failed to save uploaded image") from exc

    try:
        case = CaseInput(
            case_id=resolved_case_id,
            patient=Patient(age=age, sex=sex),
            clinical_text=clinical_text,
            medical_report=medical_report,
            image_path=image_path,
            labs=labs_dict,
            history=history_list,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    storage.save_case(case, status="pending")

    try:
        report = run_case(case)
    except Exception as exc:
        # Never leak internals to the client -- log server-side, return a
        # generic 500. run_case() already degrades gracefully internally
        # (see orchestrator.py), so reaching here means something truly
        # unexpected happened.
        storage.save_case(case, status="failed")
        log_event("api_error", endpoint="/diagnose", case_id=resolved_case_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error while processing the case") from exc

    storage.save_case(case, status="completed")
    storage.save_report(resolved_case_id, report)
    return report


@router.get("/diagnose/{case_id}", response_model=FinalReport)
def get_diagnosis(case_id: str) -> FinalReport:
    report = storage.get_report(case_id)
    if report is None:
        # In this synchronous-only first version /diagnose only returns
        # once run_case() has fully finished, so a missing report here
        # always means "no such case_id was ever submitted" (or a typo)
        # -- not "still processing". storage.get_case(case_id).status
        # ("pending"/"completed"/"failed") now exists and could
        # distinguish those states for a future async version; not
        # wired in here since this endpoint doesn't need it yet.
        raise HTTPException(status_code=404, detail=f"No report found for case_id '{case_id}'")
    return report


@router.post("/review/{case_id}", response_model=ReviewResponse)
def submit_review(case_id: str, review: ReviewRequest) -> ReviewResponse:
    if storage.get_report(case_id) is None:
        raise HTTPException(status_code=404, detail=f"No report found for case_id '{case_id}'")

    stored = storage.save_review(case_id, review.model_dump())

    return ReviewResponse(
        case_id=stored["case_id"],
        decision=stored["decision"],
        reviewer=stored["reviewer"],
        annotation=stored["annotation"],
        timestamp=stored["timestamp"],
        recorded=True,
    )


@router.get("/review/{case_id}", response_model=ReviewResponse)
def get_review(case_id: str) -> ReviewResponse:
    """Returns the most recent review for case_id (a case can be
    reviewed more than once; storage.get_reviews returns the full
    history, most recent first), or 404 if it has never been reviewed."""
    reviews = storage.get_reviews(case_id)
    if not reviews:
        raise HTTPException(status_code=404, detail=f"No review found for case_id '{case_id}'")
    review = reviews[0]

    return ReviewResponse(
        case_id=review["case_id"],
        decision=review["decision"],
        reviewer=review["reviewer"],
        annotation=review["annotation"],
        timestamp=review["timestamp"],
        recorded=True,
    )
