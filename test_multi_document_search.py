from multi_document_retriever import MultiDocumentRetriever
from vector_db import VectorDatabase


print("=" * 70)
print("REAL MULTI-DOCUMENT SEARCH TEST")
print("=" * 70)

try:
    # Initialize existing Vector Database
    db = VectorDatabase()
    db.load()

    print("\n✓ Vector database loaded.")
    print(f"  Chunks available: {len(db.documents)}")

    # Initialize Multi-Document Retriever
    retriever = MultiDocumentRetriever(db)

    print("✓ MultiDocumentRetriever initialized.")

    # ---------------------------------------------------------
    # PUT YOUR ACTUAL USER ID HERE
    # ---------------------------------------------------------

    USER_ID = "Anshika Jain2"

    # ---------------------------------------------------------
    # Test question
    # ---------------------------------------------------------

    query = "What is an encoder?"

    print("\n" + "-" * 70)
    print("QUESTION")
    print("-" * 70)
    print(query)

    # ---------------------------------------------------------
    # Search across ALL documents
    # ---------------------------------------------------------

    results = retriever.retrieve(
        user_id=USER_ID,
        query=query,
        selected_document_ids=None,
        top_k=5,
    )

    print("\n" + "-" * 70)
    print("RETRIEVED RESULTS")
    print("-" * 70)

    print(f"\nNumber of results: {len(results)}")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 50)

        if isinstance(result, dict):

            document = (
                result.get("document_name")
                or result.get("source")
                or result.get("filename")
                or "Unknown"
            )

            page = (
                result.get("page")
                or result.get("page_number")
                or "Unknown"
            )

            score = (
                result.get("score")
                or result.get("distance")
                or "Unknown"
            )

            text = (
                result.get("text")
                or result.get("content")
                or result.get("chunk")
                or ""
            )

            print("Document:", document)
            print("Page:", page)
            print("Score:", score)
            print("Text:", str(text)[:500])

        else:
            print(result)

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if results:
        print("✓ REAL MULTI-DOCUMENT RETRIEVAL WORKING")
    else:
        print("⚠ No results returned.")
        print("  Check the USER_ID and document ownership.")

    print("=" * 70)

except Exception as e:

    print("\n" + "=" * 70)
    print("TEST FAILED")
    print("=" * 70)

    print(f"\nError Type: {type(e).__name__}")
    print(f"Error: {e}")