# test_rag.py
import os

from models.embeddings import EmbeddingClient
from utils.rag import build_knowledge_base, retrieve_relevant_chunks


def main():
    docs_dir = os.path.join("data", "docs")
    print(f"Building knowledge base from: {docs_dir}")

    embed_client = EmbeddingClient()
    vectorstore = build_knowledge_base(docs_dir, embed_client)
    print(f"Vectorstore built. Num chunks: {len(vectorstore['chunks'])}")

    while True:
        query = input("\nEnter a query (or 'q' to quit): ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break

        results = retrieve_relevant_chunks(
            query=query,
            embed_client=embed_client,
            vectorstore=vectorstore,
            top_k=3,
        )

        print("\nTop matches:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Source: {r['source']}")
            print(f"Score: {r['score']:.3f}")
            print(f"Text:\n{r['text'][:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
