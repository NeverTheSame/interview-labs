"""Package resume turns what a job seeker gives R3 (an uploaded file or
pasted text) into the plain resume text stored in ``r3_resumes``."""

from .extract import (
    ACCEPTED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    ResumeExtractError,
    extract_text,
    normalize_text,
)

__all__ = ["ACCEPTED_EXTENSIONS", "MAX_UPLOAD_BYTES", "ResumeExtractError", "extract_text", "normalize_text"]
