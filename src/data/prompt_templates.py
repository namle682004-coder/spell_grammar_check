PROMPTS = {
    "simple": """Sửa lỗi chính tả và ngữ pháp trong câu sau. Chỉ trả về câu đã sửa.

Câu gốc: {text}
Câu đã sửa:""",

    "few_shot": """Sửa lỗi chính tả và ngữ pháp trong câu.

Ví dụ 1:
Input: "Toi yeu Ha Noi."
Output: "Tôi yêu Hà Nội."

Ví dụ 2:
Input: "Em hoc bai o nha."
Output: "Em học bài ở nhà."

Ví dụ 3:
Input: "Toi di hoc bang xe dap."
Output: "Tôi đi học bằng xe đạp."

Input: {text}
Output:""",

    "paragraph": """Sửa lỗi chính tả và ngữ pháp cho từng câu trong đoạn văn sau. Giữ nguyên cấu trúc đoạn văn, chỉ sửa lỗi.

Đoạn văn cần sửa:
{text}

Đoạn văn đã sửa:""",

    "english": """Fix grammar and spelling errors. Output only corrected sentence.

Input: {text}
Output:""",

    "vietnamese_friendly": """Bạn là chuyên gia tiếng Việt. Hãy sửa lỗi chính tả và ngữ pháp.

Câu cần sửa: {text}
Câu đã sửa:""",
}


def get_prompt(template_name: str, text: str) -> str:
    """Get formatted prompt for grammar correction."""
    if template_name not in PROMPTS:
        template_name = "simple"

    return PROMPTS[template_name].format(text=text)
