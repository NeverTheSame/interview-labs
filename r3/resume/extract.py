"""Plain text out of an uploaded resume file.

:func:`extract_text` accepts .txt, .md, .pdf and .docx. Every failure a job
seeker can cause (wrong type, too big, empty, a scanned PDF, a broken file)
raises :class:`ResumeExtractError` with a message meant to be shown to them
as is; ``kind`` lets the web layer pick the HTTP status without parsing it.

DOCX is read with the standard library (it is a zip of XML parts), so only
PDF needs a dependency (pypdf).
"""

from __future__ import annotations

import codecs
import io
import logging
import re
import zipfile
from pathlib import PurePath
from typing import Literal
from xml.etree import ElementTree

log = logging.getLogger("r3.resume.extract")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
# A resume is a few pages; a long PDF is something else and slow to read.
MAX_PDF_PAGES = 20
ACCEPTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
ACCEPTED_LABEL = "PDF (.pdf), Word (.docx), plain text (.txt) or Markdown (.md)"

# A .docx part bigger than this once unzipped is a zip bomb, not a resume.
_MAX_XML_BYTES = 40 * 1024 * 1024

ErrorKind = Literal["unsupported", "too_large", "empty", "no_text", "corrupt"]


class ResumeExtractError(ValueError):
    """The file cannot become resume text; ``str(e)`` is user-readable."""

    def __init__(self, message: str, kind: ErrorKind) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


def extract_text(filename: str, data: bytes) -> str:
    """Return the normalized text of a resume file, picked by its extension."""
    ext = PurePath(filename or "").suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        shown = f"“{ext}” files" if ext else "Files without an extension"
        raise ResumeExtractError(f"{shown} are not supported. Upload a {ACCEPTED_LABEL} file.", "unsupported")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ResumeExtractError(
            f"That file is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB, the upload limit. "
            "Save a smaller copy or paste the text instead.", "too_large")
    if not data:
        raise ResumeExtractError("That file is empty. Choose your resume file again or paste the text.", "empty")
    if ext == ".pdf":
        text = reflow_pdf_lines(normalize_text(_pdf_text(data)))
    elif ext == ".docx":
        text = _docx_text(data)
    else:
        text = normalize_text(_decode_text(data))
        if not text:
            raise ResumeExtractError("That file has no text in it. Choose your resume file or paste the text.", "empty")
        return text
    text = normalize_text(text)
    if not text:
        raise ResumeExtractError(
            "R3 could not find any text in that file. It may be a scanned image; "
            "open it, copy the text and paste it here instead.", "no_text")
    return text


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
_SPACES = re.compile(r"[ \t  -  　]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Tidy text without losing its lines: unify line endings, drop NULs and
    other control characters, trim trailing spaces, keep at most one blank
    line in a row and strip blank lines at both ends. Used for pasted text
    too, so every resume is stored the same way."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace(" ", "\n").replace(" ", "\n\n")
    text = _CONTROL.sub("", text).replace(" ", " ")
    text = _SPACES.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip("\n").rstrip()


def _decode_text(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:4096]:
        # A NUL byte early on means a binary file (often a PDF or Word file
        # renamed), which latin-1 would turn into pages of junk.
        raise ResumeExtractError("That file does not look like plain text. "
                                 f"Upload a {ACCEPTED_LABEL} file or paste the text.", "corrupt")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _corrupt(kind_name: str) -> ResumeExtractError:
    return ResumeExtractError(f"R3 could not read that {kind_name} file; it may be damaged. "
                              "Try saving it again, or paste the text instead.", "corrupt")


_BULLETS = ("-", "•", "*", "–")
_TERMINAL = (".", ":", ";", "!", "?")
_DATE_RANGE = re.compile(r"\b(?:19|20)\d{2}\b.*?(?:–|—|-|\bto\b)\s*(?:[A-Za-z]{3,9}\.?\s+)?"
                         r"(?:(?:19|20)\d{2}\b|present|now|current)", re.IGNORECASE)


# "Senior SRE — Corvane Payments": a role or a school, set apart by a spaced dash.
_TITLE_DASH = re.compile(r"\s[—–]\s")
# A line this long was wrapped by the PDF's margin, not ended by its author.
WRAPPED_WIDTH = 80


def _keeps_own_line(line: str) -> bool:
    """A line re-flow never joins: a bullet, a heading (ALL CAPS, or short,
    title-case and comma-free), a role or school line ("Title — Employer"), a
    line with a date range, a contact line."""
    line = line.strip()
    words = line.split()
    return (line.startswith(_BULLETS) or line.upper() == line or _DATE_RANGE.search(line) is not None
            or any(mark in line for mark in ("@", "|", "http", "www."))
            or (len(words) <= 5 and "," not in line and all(w[0].isupper() for w in words if w[0].isalpha()))
            or (_TITLE_DASH.search(line) is not None and line[:1].isupper() and not line.endswith(_TERMINAL)
                and len(words) <= 12))


def _continues(prev: str, line: str, in_bullet: bool) -> bool:
    """Whether ``line`` is the rest of ``prev``, which the PDF wrapped."""
    if not prev.strip() or not line.strip() or prev.endswith(_TERMINAL) or _keeps_own_line(line):
        return False
    if in_bullet:
        return True  # a bullet runs on until a sentence ends or a new bullet, heading or title starts
    if _keeps_own_line(prev):
        return False
    first = line.lstrip()[0]
    return (first.islower() or (prev.endswith(",") and ":" not in line)
            or (prev.count(",") >= 2 and "," in line and ":" not in line)  # a comma list carried on
            or len(prev) >= WRAPPED_WIDTH)  # a paragraph line cut at the margin


def reflow_pdf_lines(text: str) -> str:
    """Undo a PDF's hard line wraps, conservatively. A line joins the one
    before it only when that one does not end a sentence or clause and the
    line is no bullet, heading, role, dated or contact line; and then only
    when it carries on a bullet, starts lower-case, continues a comma list, or
    follows a line cut at the margin. Blank lines stay. A wrap that starts
    with a capital or a digit is still a wrap (QA D4: "moving hot" /
    "PostgreSQL reads to replicas.")."""
    out: list[str] = []
    in_bullet = False
    for line in text.split("\n"):
        prev = out[-1].rstrip() if out else ""
        if _continues(prev, line, in_bullet):
            out[-1] = f"{prev} {line.strip()}"
        else:
            out.append(line)
            in_bullet = line.strip().startswith(_BULLETS)
    return "\n".join(out)


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader  # imported here so the portal starts without parsing PDFs

    if b"%PDF" not in data[:1024]:
        raise _corrupt("PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeExtractError("That PDF is password-protected. Remove the password or paste the text instead.",
                                     "corrupt")
        if (pages := len(reader.pages)) > MAX_PDF_PAGES:
            raise ResumeExtractError(f"A resume is a few pages; this PDF has {pages}. Upload the resume itself "
                                     f"(at most {MAX_PDF_PAGES} pages).", "too_large")
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except ResumeExtractError:
        raise
    except Exception as e:  # pypdf raises many types on malformed input
        log.info("pdf extract failed: %s: %s", type(e).__name__, e)
        raise _corrupt("PDF") from e


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "word/document.xml" not in names:
                raise _corrupt("Word")
            # Headers first: many resumes keep the name and contact line there.
            parts = sorted(n for n in names if re.fullmatch(r"word/header\d*\.xml", n)) + ["word/document.xml"]
            chunks: list[str] = []
            for name in parts:
                if zf.getinfo(name).file_size > _MAX_XML_BYTES:
                    raise _corrupt("Word")
                chunk = _wordml_text(zf.read(name)).strip()
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
            return "\n\n".join(chunks)
    except ResumeExtractError:
        raise
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError, OSError, ValueError, RecursionError) as e:
        log.info("docx extract failed: %s: %s", type(e).__name__, e)
        raise _corrupt("Word") from e


def _wordml_text(xml: bytes) -> str:
    # Word never writes a DTD; refusing one rules out entity-expansion bombs.
    if b"<!DOCTYPE" in xml[:2048].upper():
        raise _corrupt("Word")
    out: list[str] = []

    def walk(el: ElementTree.Element) -> None:
        tag = el.tag
        if tag == _W + "t":
            out.append(el.text or "")
        elif tag == _W + "tab":
            out.append("\t")
        elif tag in (_W + "br", _W + "cr"):
            out.append("\n")
        for child in el:
            walk(child)
        if tag == _W + "p":
            out.append("\n")

    walk(ElementTree.fromstring(xml))
    return "".join(out)
