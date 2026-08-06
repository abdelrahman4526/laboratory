
from .schemas import SearchResult


AGREEMENT_BOOST = 0.05


def merge_results(*result_lists: list[SearchResult]) -> list[SearchResult]:
    
    by_key: dict[tuple[int, str], dict] = {}

    for result_list in result_lists:
        for result in result_list:
            key = (result.id, result.type.value)
            if key not in by_key:
                by_key[key] = {
                    "id": result.id,
                    "type": result.type,
                    "name": result.name,
                    "best_score": result.score,
                    "sources": {result.source},
                }
            else:
                entry = by_key[key]
                entry["best_score"] = max(entry["best_score"], result.score)
                entry["sources"].add(result.source)

    merged: list[SearchResult] = []
    for entry in by_key.values():
        extra_methods = len(entry["sources"]) - 1
        boosted_score = min(1.0, entry["best_score"] + extra_methods * AGREEMENT_BOOST)
        merged.append(SearchResult(
            id=entry["id"],
            type=entry["type"],
            name=entry["name"],
            score=round(boosted_score, 3),
            source="+".join(sorted(entry["sources"])),
        ))

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged