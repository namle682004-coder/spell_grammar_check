"""Compute ROUGE and BERTScore metrics."""
from __future__ import annotations

from loguru import logger


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    import evaluate
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {k: round(v, 4) for k, v in results.items()}


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    import evaluate
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang,
    )
    return {
        "bertscore_precision": round(sum(results["precision"]) / len(results["precision"]), 4),
        "bertscore_recall": round(sum(results["recall"]) / len(results["recall"]), 4),
        "bertscore_f1": round(sum(results["f1"]) / len(results["f1"]), 4),
    }


def compute_gleu(predictions: list[str], references: list[str]) -> dict:
    """Compute Sentence-level GLEU score for Grammar Error Correction."""
    try:
        from nltk.translate.gleu_score import sentence_gleu

        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens = pred.split()
            r_tokens = ref.split()
            if p_tokens and r_tokens:
                scores.append(sentence_gleu([r_tokens], p_tokens))
        if scores:
            return {"gleu_score": round(sum(scores) / len(scores), 4)}
    except Exception as e:
        logger.debug(f"NLTK GLEU fallback to pure Python implementation: {e}")

    from collections import Counter

    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens = pred.split()
        r_tokens = ref.split()
        if not p_tokens or not r_tokens:
            continue

        p_ngrams = Counter(
            tuple(p_tokens[i : i + n])
            for n in range(1, 5)
            for i in range(len(p_tokens) - n + 1)
        )
        r_ngrams = Counter(
            tuple(r_tokens[i : i + n])
            for n in range(1, 5)
            for i in range(len(r_tokens) - n + 1)
        )
        tpfp = sum(p_ngrams.values())
        tpfn = sum(r_ngrams.values())
        overlap = p_ngrams & r_ngrams
        tp = sum(overlap.values())
        denom = max(tpfp, tpfn)
        scores.append(tp / denom if denom > 0 else 0.0)

    avg_gleu = sum(scores) / len(scores) if scores else 0.0
    return {"gleu_score": round(avg_gleu, 4)}


def compute_cer_wer(predictions: list[str], references: list[str]) -> dict:
    """Compute Character Error Rate (CER) and Word Error Rate (WER)."""
    import difflib

    def _levenshtein(s1: list | str, s2: list | str) -> int:
        matcher = difflib.SequenceMatcher(None, s1, s2)
        return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")

    total_char_dist = 0
    total_ref_chars = 0
    total_word_dist = 0
    total_ref_words = 0

    for pred, ref in zip(predictions, references):
        c_dist = _levenshtein(pred, ref)
        total_char_dist += c_dist
        total_ref_chars += max(len(ref), 1)

        w_pred = pred.split()
        w_ref = ref.split()
        w_dist = _levenshtein(w_pred, w_ref)
        total_word_dist += w_dist
        total_ref_words += max(len(w_ref), 1)

    cer = total_char_dist / total_ref_chars if total_ref_chars > 0 else 0.0
    wer = total_word_dist / total_ref_words if total_ref_words > 0 else 0.0

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
    }


def compute_length_stats(predictions: list[str]) -> dict:
    lengths = [len(p.split()) for p in predictions]
    return {
        "avg_pred_length_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_pred_length": min(lengths) if lengths else 0,
        "max_pred_length": max(lengths) if lengths else 0,
    }


def compute_all_metrics(predictions: list[str], references: list[str]) -> dict:
    logger.info("Computing ROUGE...")
    rouge = compute_rouge(predictions, references)
    logger.info("Computing BERTScore...")
    bs = compute_bertscore(predictions, references)
    logger.info("Computing GLEU & CER/WER...")
    gleu = compute_gleu(predictions, references)
    cer_wer = compute_cer_wer(predictions, references)
    length = compute_length_stats(predictions)
    return {**rouge, **bs, **gleu, **cer_wer, **length}

