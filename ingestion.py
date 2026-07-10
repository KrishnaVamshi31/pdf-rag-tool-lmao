from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        documents.append(
            Document(page_content=text, metadata={"source": file_path, "page": i + 1})
        )
    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # same hierarchy as before: paragraph -> line -> sentence -> word -> char
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                Document(page_content=chunk, metadata={**doc.metadata, "chunk_index": idx})
            )
    return all_chunks


if __name__ == "__main__":
    docs = load_pdf("sample.pdf")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages -> split into {len(chunks)} chunks")
    print("--- Chunk 0 ---")
    print(chunks[0].page_content)
    print(chunks[0].metadata)
    print("--- Chunk 1 ---")
    print(chunks[1].page_content)
    print(chunks[1].metadata)