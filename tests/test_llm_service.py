from src.services.llm_service import LLMService


def test_extract_corrections_diff_alignment():
    service = LLMService()

    original = "Tôi học python"
    corrected = "Tôi đang học ngôn ngữ Python"

    errors = service._extract_corrections(original, corrected)

    # Should detect insertions/replacements without word index shifting
    assert len(errors) > 0
    
    # Check that "Tôi" and "học" were recognized as matching and didn't trigger cascaded errors
    originals_flagged = [e["original"] for e in errors]
    assert "Tôi" not in originals_flagged
    assert "học" not in originals_flagged


def test_extract_corrections_exact_match():
    service = LLMService()

    original = "Tôi học Python"
    corrected = "Tôi học Python"

    errors = service._extract_corrections(original, corrected)
    assert len(errors) == 0
