"""A plain-text resume as a clean one-column PDF (``GET /resume/{id}/pdf``).

The layout is read from the text alone: the first non-empty line is the
name (set larger) unless it is over :data:`NAME_MAX_CHARS` or the only line
(a resume that lost its line breaks is body text, not a bold block), short
lines in capitals or ending in ":" are section headings (bold), lines
starting with ``-``, ``*`` or ``•`` are bullets, and everything else is a
paragraph line. fpdf2 breaks pages itself.

Text is set in the bundled DejaVu Sans (``fonts/``, Bitstream Vera/DejaVu
licence, see ``fonts/LICENSE``), which covers Latin, Cyrillic, Greek and the
usual typography (curly quotes, dashes, bullets). If the font files are
missing, the text is folded to latin-1 instead (accents stripped, quotes
and dashes straightened, anything else ``?``) so a download never fails.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from fpdf import FPDF

FONT_DIR = Path(__file__).parent / "fonts"
_REGULAR = FONT_DIR / "DejaVuSans.ttf"
_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

NAME_MAX_CHARS = 80

_BULLET = re.compile(r"^\s*[-*•·–]\s+")
_FOLD = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-",
                       "‒": "-", "−": "-", "•": "-", "·": "-", "…": "...", " ": " ", " ": " ",
                       "​": ""})


def to_latin1(text: str) -> str:
    """Fold text into latin-1 for the core fonts: typography to ASCII,
    accents stripped where that helps, anything else ``?``."""
    out = []
    for ch in text.translate(_FOLD):
        if unicodedata.combining(ch):  # a stray stress mark is dropped, not turned into "?"
            continue
        if ord(ch) < 256:
            out.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append(base if base and all(ord(c) < 256 for c in base) else "?")
    return "".join(out)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60 or _BULLET.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    return (s.endswith(":") and len(s.split()) <= 5) or (len(letters) >= 3 and all(c.isupper() for c in letters))


def render_pdf(title: str, content: str, *, unicode_font: bool | None = None) -> bytes:
    """The resume as PDF bytes. ``unicode_font`` forces the font path (True
    = bundled TTF, False = latin-1 fold); by default the TTF is used when
    present."""
    use_ttf = (_REGULAR.exists() and _BOLD.exists()) if unicode_font is None else unicode_font
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    if use_ttf:
        pdf.add_font("Body", "", str(_REGULAR))
        pdf.add_font("Body", "B", str(_BOLD))
        family, clean = "Body", (lambda s: s)
    else:
        family, clean = "Helvetica", to_latin1
    pdf.set_title(clean(title))
    pdf.set_creator("R3")
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    lines = content.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    filled = [i for i, line in enumerate(lines) if line.strip()]
    # A name is short and has the resume below it; a long first line or a
    # one-line text (a resume that lost its line breaks) is set as body text.
    first = filled[0] if len(filled) > 1 and len(lines[filled[0]].strip()) <= NAME_MAX_CHARS else None
    for i, raw in enumerate(lines):
        line = clean(raw.rstrip())
        pdf.set_x(pdf.l_margin)
        if not line.strip():
            pdf.ln(2.5)
        elif i == first:
            pdf.set_font(family, "B", 17)
            pdf.multi_cell(width, 8, line.strip(), align="L")
            pdf.ln(1)
        elif _is_heading(line):
            pdf.ln(2)
            pdf.set_font(family, "B", 11.5)
            pdf.multi_cell(width, 6, line.strip(), align="L")
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, y, pdf.l_margin + width, y)
            pdf.ln(1.5)
        elif m := _BULLET.match(line):
            pdf.set_font(family, "", 10)
            mark = "•" if use_ttf else "-"
            pdf.cell(5, 5, mark)
            pdf.multi_cell(width - 5, 5, line[m.end():].strip(), align="L")
        else:
            pdf.set_font(family, "", 10)
            pdf.multi_cell(width, 5, line.strip(), align="L")
    return bytes(pdf.output())
