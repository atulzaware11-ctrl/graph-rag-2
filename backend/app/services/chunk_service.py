def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split extracted page text into overlapping chunks.

    Keeping the page number allows us to cite the original PDF later.
    """

    chunks = []

    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page": page_number,
                        "text": chunk,
                    }
                )

                chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks
