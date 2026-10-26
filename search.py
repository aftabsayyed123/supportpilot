from ingest import COLLECTION, embed, qdrant


def search(question: str, limit: int = 3):
    question_vector = embed([question])[0]
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=question_vector,
        limit=limit,
        with_payload=True,
    )
    return results.points


if __name__ == "__main__":
    questions = [
        "Can I change where my parcel is going? It already shipped.",
        "I was charged twice, when do I get my money back?",
        "Do you sell gift vouchers?",
        "What is the capital of France?",
    ]

    for question in questions:
        print(f"\nQUESTION: {question}")
        for point in search(question):
            print(f"  {point.score:.3f}  {point.payload['source']} > {point.payload['section']}")