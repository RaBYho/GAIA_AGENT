"""
Reimplementation simplifiee du scorer officiel GAIA (exact match avec normalisation).
Basee sur la logique publiee dans le depot GAIA (question_scorer).
"""

import string


def _normalize_number(text: str):
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_string(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    articles = {"a", "an", "the"}
    words = [w for w in text.split() if w not in articles]
    return " ".join(words)


def question_scorer(prediction: str, ground_truth: str) -> bool:
    """Retourne True si prediction correspond a ground_truth selon les regles GAIA
    (nombres compares avec tolerance, chaines normalisees, listes separees par virgule)."""
    gt_number = _normalize_number(ground_truth)
    pred_number = _normalize_number(prediction)

    if gt_number is not None:
        if pred_number is None:
            return False
        return abs(gt_number - pred_number) < 1e-6

    if "," in ground_truth:
        gt_items = [_normalize_string(x) for x in ground_truth.split(",")]
        pred_items = [_normalize_string(x) for x in prediction.split(",")]
        return gt_items == pred_items

    return _normalize_string(prediction) == _normalize_string(ground_truth)
