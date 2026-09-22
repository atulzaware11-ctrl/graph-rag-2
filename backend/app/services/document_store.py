class DocumentStore:

    def __init__(self):
        self.documents: list[dict] = []

    def add_documents(
        self,
        documents: list[dict],
    ) -> None:

        self.documents.extend(documents)

    def get_documents(self) -> list[dict]:

        return self.documents


document_store = DocumentStore()
