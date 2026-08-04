from rag.retrieval_service import retrieve
from graph.schemas.intent_sechema import RefinedQuery    # عدّل المسار حسب مكانها الفعلي


def print_result(query_text: str, refined_queries: list[RefinedQuery]):
    print("=" * 70)
    print(f"Query: {query_text}")
    print("=" * 70)

    state = {"refined_queries": refined_queries}
    result = retrieve(state)
    print(refined_queries)

    print(f"\nTop score: {result.top_score}")
    print(f"Number of results: {len(result.results)}\n")

    print("--- Results ---")
    for r in result.results:
        print(f"  [{r.source}] {r.name}  (type={r.type}, score={r.score})")

    print("\n--- Context sent to LLM ---")
    print(result.context)
    print("\n")


if __name__ == "__main__":
    # مؤقتًا بنبني الـ refined_queries يدويًا لحد ما نربط بالـ intent step فعليًا
    test_cases = [
        (
            "عايز اعمل Cpc و inr و urin و rbs و uric acid و iron و tsh و فينامين د و كالسيوم",
             [
                RefinedQuery(query="CBC", aliases=["Cpc", "Complete Blood Count"], keywords=[], description=""),
                RefinedQuery(query="INR", aliases=["inr"], keywords=[], description=""),
                RefinedQuery(query="Urine Analysis", aliases=["urin"], keywords=[], description=""),
                RefinedQuery(query="RBS", aliases=["rbs", "Random Blood Sugar"], keywords=[], description=""),
                RefinedQuery(query="Uric Acid", aliases=[], keywords=[], description=""),
                RefinedQuery(query="Iron", aliases=[], keywords=[], description=""),
                RefinedQuery(query="TSH", aliases=[], keywords=[], description=""),
                RefinedQuery(query="Vitamin D", aliases=["فيتامين د"], keywords=[], description=""),
               
            ],
        ),
    ]

    for query_text, refined_queries in test_cases:
        print_result(query_text, refined_queries)