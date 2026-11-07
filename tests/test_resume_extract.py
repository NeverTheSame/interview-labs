"""Text extraction from uploaded resume files (r3/resume/extract.py)."""

import io
import zipfile

import pytest

from r3.resume import MAX_UPLOAD_BYTES, ResumeExtractError, extract_text, normalize_text

RESUME = "Jane Doe\nSenior Platform Engineer\n\nBuilt Kubernetes platforms for 8 years."

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(paragraphs: list[str], header: str | None = None) -> bytes:
    def part(paras: list[str], root: str) -> str:
        body = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>" for p in paras)
        return f'<?xml version="1.0" encoding="UTF-8"?><w:{root} xmlns:w="{W_NS}">{body}</w:{root}>'

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", part(paragraphs, "document").replace(
            "<w:p>", "<w:body><w:p>", 1).replace(f"</w:document>", "</w:body></w:document>"))
        if header is not None:
            zf.writestr("word/header1.xml", part([header], "hdr"))
    return buf.getvalue()


def make_pdf(lines: list[str] | None) -> bytes:
    """A minimal one-page PDF with the given lines of Helvetica text (or no
    text at all, like a scanned page), written by hand with a correct xref."""
    ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET" if lines else ""
    stream = ops.encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> "
        b"/Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def test_txt_and_md():
    assert extract_text("cv.txt", RESUME.encode()) == RESUME
    assert extract_text("CV.MD", ("# " + RESUME).encode()) == "# " + RESUME


def test_txt_bom_latin1_and_utf16():
    assert extract_text("cv.txt", b"\xef\xbb\xbf" + "Zoë Müller\nEngineer".encode()) == "Zoë Müller\nEngineer"
    assert extract_text("cv.txt", "José Núñez\nEngineer".encode("latin-1")) == "José Núñez\nEngineer"
    assert extract_text("cv.txt", "Ana\nEngineer".encode("utf-16")) == "Ana\nEngineer"


def test_normalize_keeps_lines_and_collapses_blank_runs():
    raw = "Jane\x00 Doe  \r\n\r\n\r\n\r\nEngineer\t\n\n\n\n- Go \n\n"
    assert normalize_text(raw) == "Jane Doe\n\nEngineer\n\n- Go"
    assert normalize_text("\n\n  indented\n") == "  indented"


def test_docx_paragraphs_and_header():
    data = make_docx(["Senior Platform Engineer", "Built Kubernetes platforms &amp; CI"], header="Jane Doe · jane@x.io")
    assert extract_text("resume.docx", data) == ("Jane Doe · jane@x.io\n\nSenior Platform Engineer\n"
                                                 "Built Kubernetes platforms & CI")


def test_docx_tabs_breaks_and_skipped_deletions():
    xml = (f'<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/>'
           "<w:t>C</w:t><w:delText>gone</w:delText></w:r></w:p></w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    assert extract_text("r.docx", buf.getvalue()) == "A\tB\nC"


def test_pdf_text():
    text = extract_text("resume.pdf", make_pdf(["Jane Doe", "Senior Platform Engineer"]))
    assert "Jane Doe" in text and "Senior Platform Engineer" in text


@pytest.mark.parametrize("filename, data, kind, needle", [
    ("resume.pages", b"x", "unsupported", ".pdf"),
    ("resume", b"x", "unsupported", "without an extension"),
    ("resume.exe", b"MZ", "unsupported", "Word (.docx)"),
    ("resume.txt", b"", "empty", "empty"),
    ("resume.pdf", b"", "empty", "empty"),
    ("resume.txt", b"   \n\n\t \n", "empty", "no text"),
    ("resume.pdf", make_pdf(None), "no_text", "paste"),
    ("resume.docx", make_docx([""]), "no_text", "scanned"),
    ("resume.pdf", b"%PDF-1.4\nthis is not really a pdf", "corrupt", "damaged"),
    ("resume.pdf", b"PK\x03\x04 a zip renamed", "corrupt", "damaged"),
    ("resume.docx", b"not a zip at all", "corrupt", "damaged"),
    ("resume.docx", make_docx(["x"])[:60], "corrupt", "damaged"),
    ("resume.txt", b"%PDF\x00\x01\x02binary", "corrupt", "plain text"),
])
def test_errors(filename, data, kind, needle):
    with pytest.raises(ResumeExtractError) as e:
        extract_text(filename, data)
    assert e.value.kind == kind
    assert needle in str(e.value)


def test_docx_without_document_part_and_with_doctype():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/other.xml", "<x/>")
    with pytest.raises(ResumeExtractError, match="damaged"):
        extract_text("r.docx", buf.getvalue())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", '<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>')
    with pytest.raises(ResumeExtractError, match="damaged"):
        extract_text("r.docx", buf.getvalue())


def test_five_megabyte_limit():
    assert extract_text("cv.txt", b"a" * MAX_UPLOAD_BYTES).startswith("aaa")
    with pytest.raises(ResumeExtractError) as e:
        extract_text("cv.txt", b"a" * (MAX_UPLOAD_BYTES + 1))
    assert e.value.kind == "too_large" and "5 MB" in str(e.value)


def test_pdf_hard_wraps_are_reflowed():
    text = extract_text("resume.pdf", make_pdf([
        "EXPERIENCE", "Built a multi-region platform on AWS and Kubernetes. Led a", "platform team of 5 engineers.",
        "SKILLS", "Prometheus, Grafana, Loki,", "PostgreSQL, Kafka, Terraform"]))
    assert text.split("\n") == [
        "EXPERIENCE", "Built a multi-region platform on AWS and Kubernetes. Led a platform team of 5 engineers.",
        "SKILLS", "Prometheus, Grafana, Loki, PostgreSQL, Kafka, Terraform"]


@pytest.mark.parametrize("text", [
    "EXPERIENCE\nplatform work across teams",
    "Staff Site Reliability Engineer — Northwind Logistics, Vancouver (2021 – present)\nowned the on-call rotation",
    "Built the platform and\n- cut paging by 40%",
    "- Ran the on-call rotation for\n• reduced costs",
    "* Ran the on-call rotation for\n– reduced costs",
    "Jane Doe\njane@example.com | +1 604 555 0100 | linkedin.com/in/janedoe",
    "Vancouver, BC,\njane@example.com",
    "Led a\n\nplatform team of 5",
    "Led the platform team.\nplatform work continued",
    "Languages: Python, Go,\nCloud: AWS, GCP",
    "Site Reliability Engineer\nplatform team",
])
def test_reflow_leaves_bullets_headings_dates_and_contacts(text):
    from r3.resume.extract import reflow_pdf_lines

    assert reflow_pdf_lines(text) == text


def test_only_pdfs_are_reflowed():
    assert extract_text("resume.txt", b"Led a\nplatform team") == "Led a\nplatform team"


# QA D4 (2026-10-14): Dana's PDF kept wraps that start with a capital or a digit as their
# own lines, so bullets became fragments ("PostgreSQL reads to replicas.") and the skill
# "Incident Response" became two skills.
DANA_WRAPS = [
    ("• Cut p99 checkout API latency by 38% by tuning Envoy connection pools and moving hot\n"
     "PostgreSQL reads to replicas.",
     "• Cut p99 checkout API latency by 38% by tuning Envoy connection pools and moving hot "
     "PostgreSQL reads to replicas."),
    ("• Introduced SLOs and error budgets for 45 customer-facing services; paging volume dropped\n"
     "52% within two quarters.",
     "• Introduced SLOs and error budgets for 45 customer-facing services; paging volume dropped "
     "52% within two quarters."),
    ("Kubernetes, Terraform, AWS, GCP, Prometheus, Grafana, Go, Python, PostgreSQL, Incident\n"
     "Response, SLOs and Error Budgets, Linux, CI/CD",
     "Kubernetes, Terraform, AWS, GCP, Prometheus, Grafana, Go, Python, PostgreSQL, Incident "
     "Response, SLOs and Error Budgets, Linux, CI/CD"),
    ("Senior Site Reliability Engineer with 10 years of experience running payment and logistics platforms on "
     "Kubernetes across AWS and GCP. Looking for a remote\nSenior SRE or Platform Engineering role.",
     "Senior Site Reliability Engineer with 10 years of experience running payment and logistics platforms on "
     "Kubernetes across AWS and GCP. Looking for a remote Senior SRE or Platform Engineering role."),
]


@pytest.mark.parametrize(("wrapped", "whole"), DANA_WRAPS)
def test_wraps_that_start_with_a_capital_or_a_digit_rejoin(wrapped, whole):
    from r3.resume.extract import reflow_pdf_lines

    assert reflow_pdf_lines(wrapped) == whole


@pytest.mark.parametrize("text", [
    "• Built the Prometheus and Grafana stack\nSenior Site Reliability Engineer — Corvane Payments",
    "• Ran incident command for 70+ incidents\nAug 2020 – Present",
    "• Ran incident command for 70+ incidents\nEDUCATION",
    "Kubernetes, Terraform, AWS\nB.Eng. Software Engineering — McMaster University, 2015",
])
def test_a_title_date_or_heading_after_an_unpunctuated_bullet_stays_apart(text):
    from r3.resume.extract import reflow_pdf_lines

    assert reflow_pdf_lines(text) == text


def test_a_skill_wrapped_across_lines_stays_one_skill():
    from r3.resume.extract import reflow_pdf_lines

    text = reflow_pdf_lines("SKILLS\nKubernetes, PostgreSQL, Incident\nResponse, SLOs and Error Budgets, Linux")
    assert text == "SKILLS\nKubernetes, PostgreSQL, Incident Response, SLOs and Error Budgets, Linux"
