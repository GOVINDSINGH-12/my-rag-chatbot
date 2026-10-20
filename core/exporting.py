"""Create downloadable files from an assistant response or chat transcript."""

from html import escape
from io import BytesIO


def create_export_file(
    content: str, file_format: str, *, is_conversation: bool = False
) -> tuple[bytes, str, str]:
    """Return file bytes, a download filename, and its browser MIME type."""
    file_prefix = "adhivai-chat" if is_conversation else "adhivai-answer"
    if file_format == "txt":
        return content.encode("utf-8"), f"{file_prefix}.txt", "text/plain"

    if file_format != "pdf":
        raise ValueError(f"Unsupported export format: {file_format}")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate
    except ImportError as exc:
        raise RuntimeError(
            "PDF export needs ReportLab. Install dependencies from requirements.txt."
        ) from exc

    # Escape model text before using ReportLab's paragraph markup parser.
    safe_content = escape(content).replace("\n", "<br/>")
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="AdhivAI chat" if is_conversation else "AdhivAI answer",
    )
    document.build([Paragraph(safe_content, getSampleStyleSheet()["BodyText"])])
    return output.getvalue(), f"{file_prefix}.pdf", "application/pdf"
