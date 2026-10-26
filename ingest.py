import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

KNOWLEDGE_DIR = Path("knowledge")
COLLECTION = "novacart_kb"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

openai_client = OpenAI()
qdrant = QdrantClient(url="http://127.0.0.1:6333")


def split_into_chunks(text: str, source: str) -> list[dict]:
    parts = text.split("\n## ")
    title = parts[0].strip().lstrip("#").strip()
    chunks = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        heading = heading.strip()
        body = body.strip()
        if body:
            chunks.append({
                "text": f"{title} - {heading}\n{body}",
                "source": source,
                "section": heading,
            })
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(split_into_chunks(text, path.name))
    print(f"Loaded {len(chunks)} chunks from {KNOWLEDGE_DIR}/")

    vectors = embed([chunk["text"] for chunk in chunks])
    print(f"Created {len(vectors)} embeddings, each with {len(vectors[0])} numbers")

    if qdrant.collection_exists(collection_name=COLLECTION):
        qdrant.delete_collection(collection_name=COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['source']}#{chunk['section']}")),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Stored {len(points)} chunks in Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    main()