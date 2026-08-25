import re


def process_results(doc, results):
    target = ['A', 'B', 'C', 'D'][doc["answer"]]
    prediction = results[0]

    match = re.search(r"^\s*[a-zA-Z]", prediction)
    if match and match.group().strip().lower() == target.lower():
        return { "exact_match": 1.0 }
    else:
        return { "exact_match": 0.0 }