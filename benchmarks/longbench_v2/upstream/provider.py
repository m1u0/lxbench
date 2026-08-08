"""Minimal LongBench v2 direct-answer grading surface.

Extracted from THUDM/LongBench pred.py and result.py. See SOURCE.md and LICENSE.
"""

import re


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def score_predictions(predictions):
    difficulty_counts = {"easy": 0, "hard": 0}
    difficulty_correct = {"easy": 0, "hard": 0}
    length_counts = {"short": 0, "medium": 0, "long": 0}
    length_correct = {"short": 0, "medium": 0, "long": 0}

    for prediction in predictions:
        difficulty = prediction["difficulty"]
        length = prediction["length"]
        correct = int(prediction["judge"])
        difficulty_counts[difficulty] += 1
        difficulty_correct[difficulty] += correct
        length_counts[length] += 1
        length_correct[length] += correct

    def percentage(correct, total):
        return round(100 * correct / total, 1) if total else 0.0

    return {
        "overall_accuracy": percentage(
            sum(difficulty_correct.values()), len(predictions)
        ),
        "difficulty": {
            name: percentage(difficulty_correct[name], difficulty_counts[name])
            for name in ("easy", "hard")
        },
        "context_length": {
            name: percentage(length_correct[name], length_counts[name])
            for name in ("short", "medium", "long")
        },
    }
