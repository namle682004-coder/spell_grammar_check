from src.evaluation.metrics import compute_cer_wer, compute_gleu, compute_length_stats


def test_compute_gleu():
    preds = ["Tôi đang học Python."]
    refs = ["Tôi đang học Python."]
    res = compute_gleu(preds, refs)
    assert "gleu_score" in res
    assert res["gleu_score"] == 1.0


def test_compute_cer_wer():
    preds = ["Tôi học python"]
    refs = ["Tôi đang học Python"]
    res = compute_cer_wer(preds, refs)
    assert "cer" in res
    assert "wer" in res
    assert res["cer"] >= 0
    assert res["wer"] >= 0


def test_compute_length_stats():
    preds = ["Tôi đang học Python"]
    stats = compute_length_stats(preds)
    assert stats["avg_pred_length_words"] == 4
