import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.aicredits.in/v1",
    api_key=os.getenv("AICREDITS_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful assistant answering questions strictly based on the provided document context.

Rules:
- Only use information found in the context below to answer.
- If the answer isn't in the context, say "I couldn't find that in the document" — don't guess or use outside knowledge.
- When relevant, mention which page the information came from.
- Be concise and direct."""


def format_context(results):
    blocks = []
    for doc, score in results:
        page = doc.metadata.get("page", "unknown")
        blocks.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(query: str, vector_store, k: int = 4):
    results = vector_store.similarity_search(query, k=k)
    context = format_context(results)

    user_message = f"""Context from the document:

{context}

---

Question: {query}"""

    response = client.chat.completions.create(
        model="anthropic/claude-3-haiku",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.choices[0].message.content
    sources = sorted(set(doc.metadata.get("page") for doc, _ in results))
    return answer, sources


# ---- everything below this line replaces your old if __name__ block ----
if __name__ == "__main__":
    from ingestion import load_pdf, split_documents
    from embeddings_store import build_vector_store

    print("Loading PDF...")
    docs = load_pdf("sample.pdf")
    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    store = build_vector_store(chunks)
    print("Vector store built")

    query = "how does mutual induction work in a transformer"
    print(f"Sending query to Claude via AICredits...")

    answer, sources = answer_question(query, store)

    print(f"Question: {query}\n")
    print(f"Answer: {repr(answer)}")
    print(f"Sources: pages {sources}")