"""API-layer request/response models.

Core domain contracts (CaseInput, FinalReport, etc.) live in
backend/models/schemas.py -- import from there rather than redefining
shapes here. This module holds only API-specific wrappers that no
agent needs to know about, like the human review decision below.
"""

from typing import Literal

from pydantic import BaseModel, model_validator


class ReviewRequest(BaseModel):
    decision: Literal["accept", "override", "annotate"]
    # For "annotate", this is the free-text note. For "override", this
    # carries the reviewer's replacement diagnosis (there's no separate
    # structured field for it -- the frontend puts whatever diagnosis
    # the reviewer picked/typed in here, e.g. "Override diagnosis:
    # Migraine with aura"), which is why both require it non-empty.
    annotation: str | None = None
    reviewer: str

    @model_validator(mode="after")
    def _annotation_required_for_annotate_or_override(self) -> "ReviewRequest":
        if self.decision in ("annotate", "override") and not (self.annotation and self.annotation.strip()):
            raise ValueError(
                "'annotation' is required when decision is 'annotate' (the note) "
                "or 'override' (the replacement diagnosis)"
            )
        return self


class ReviewResponse(BaseModel):
    case_id: str
    decision: str
    reviewer: str
    annotation: str | None
    timestamp: str
    recorded: bool
