"""Hand-labelled arithmetic example. No model, network, or API calls."""

def ranked_precision(labels):
    if not labels:
        return None
    relevant = sum(labels)
    if relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, useful in enumerate(labels, start=1):
        hits += useful
        if useful:
            total += hits / rank
    return total / relevant


def supported_fraction(labels):
    return sum(labels) / len(labels) if labels else None


for order in ([1, 0, 1], [0, 1, 1], [1, 1, 0]):
    print(f"{order}: {ranked_precision(order):.4f}")

print("faithfulness:", supported_fraction([1, 0]))
print("context_recall:", supported_fraction([1, 1]))
