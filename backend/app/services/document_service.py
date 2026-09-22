from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns one dictionary per page so that we can preserve
    page-level citation information.
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are currently supported.")

    return extract_pdf_text(str(path))
