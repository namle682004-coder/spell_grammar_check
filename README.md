# Spell & Grammar Correction with LLM

Production-ready API and inference pipeline for Vietnamese / English spelling and grammar correction. Built with FastAPI, PostgreSQL, prompt-first inference, and optional vLLM deployment.

---

## Quickstart

```bash
cp .env.example .env
uv python install 3.11
uv sync --group dev
uv run uvicorn src.main:app --host 0.0.0.0 --port 8080
```

### Docker quickstart

```bash
docker compose up -d
```

## FastAPI SaaS API

Ngoài ML pipeline, dự án có lớp API production với auth, quota, và PostgreSQL:

| Service    | Port     | Mô tả                                            |
| ---------- | -------- | ------------------------------------------------ |
| FastAPI    | **8080** | REST API (`/v1/check`, auth, quota, corrections) |
| vLLM       | **8000** | Inference server (GPU, optional)                 |
| PostgreSQL | 5432     | User data, API keys, requests                    |

```bash
# 1. Start database
docker compose up -d spell_grammar_db

# 2. Run migrations
bash scripts/migrate_db.sh

# 3. Start API (port 8080)
cp .env.example .env   # set DATABASE_URL, JWT_SECRET
uv run uvicorn src.main:app --host 0.0.0.0 --port 8080

# 4. (Optional) Start vLLM and point API to it
export VLLM_BASE_URL=http://localhost:8000
bash scripts/run_deploy_vllm.sh
```

Auth: dùng `X-API-Key` header cho API calls, hoặc JWT Bearer token cho dashboard routes (`/v1/auth/*`).

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Yêu Cầu Phần Cứng](#yêu-cầu-phần-cứng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt Nhanh](#cài-đặt-nhanh)
- [Hướng Dẫn Chi Tiết](#hướng-dẫn-chi-tiết)
  - [Bước 1: Cài Đặt Môi Trường](#bước-1-cài-đặt-môi-trường)
  - [Bước 2: Test Prompting với Model Nhỏ](#bước-2-test-prompting-với-model-nhỏ)
  - [Bước 3: Đánh Giá Kết Quả Prompting](#bước-3-đánh-giá-kết-quả-prompting)
  - [Bước 4: (Tuỳ Chọn) Finetune với LoRA](#bước-4-tuỳ-chọn-finetune-với-lora)
  - [Bước 5: So Sánh Kết Quả](#bước-5-so-sánh-kết-quả)
  - [Bước 6: Triển Khai Model](#bước-6-triển-khai-model)
- [Cấu Hình](#cấu-hình)
- [Xử Lý Lỗi](#xử-lý-lỗi)
- [Kế Hoách Mở Rộng](#kế-hoạch-mở-rộng)

---

## 🎯 Tổng Quan

Dự án này thực hiện **2 giai đoạn** cho bài toán sửa lỗi chính tả & ngữ pháp:

| Giai Đoạn        | Mô Tả                                 | Thời Gian | GPU Cần          |
| ---------------- | ------------------------------------- | --------- | ---------------- |
| **1. Prompting** | Test zero-shot/few-shot với model nhỏ | 5-15 phút | RTX 3050 (6-8GB) |
| **2. Finetune**  | Finetune LoRA nếu prompting không đạt | 2-4 giờ   | 8GB+ hoặc cloud  |

### Tại Sao Làm Theo Cách Này?

- ✅ **Tiết kiệm thời gian & tài nguyên** - Prompting mất vài phút, finetune mất vài giờ
- ✅ **Xác thực nhanh** - Đánh giá nhanh model base có phù hợp với bài toán hay không
- ✅ **Có baseline** - Có số liệu metric trước khi đầu tư finetune
- ✅ **Tối ưu chi phí** - RTX 3050 chạy inference tốt cho model 1B-3B

---

## 💻 Yêu Cầu Phần Cứng

### Chạy Prompting (Inference)

| GPU          | VRAM  | Model Hỗ Trợ                  | Tốc Độ    |
| ------------ | ----- | ----------------------------- | --------- |
| RTX 3050 6GB | 6GB   | Qwen2.5-1.5B, Phi-3-mini-3.8B | Nhanh     |
| RTX 3050 8GB | 8GB   | Qwen2.5-3B, Phi-3-mini-3.8B   | Nhanh     |
| RTX 3060+    | 12GB+ | Llama-3.2-3B, Mistral-7B      | Rất Nhanh |

### Finetune LoRA

- **Tối thiểu**: RTX 3050 8GB cho model 1.5B
- **Khuyến nghị**: 16GB+ cho model 3B+
- **Thay thế**: Dùng Google Colab Pro hoặc cloud GPU (Vast.ai, RunPod)

---

## 📁 Cấu Trúc Dự Án

```
spell-gramar-check/
├── configs/                           # File cấu hình
│   ├── eval_prompting.yaml            # Cấu hình đánh giá prompting
│   ├── eval_finetuned.yaml            # Cấu hình đánh giá sau finetune
│   └── train_finetune.yaml            # Cấu hình finetune
│
├── data/                              # Dữ liệu
│   ├── raw/                           # Dữ liệu thô
│   ├── processed/                     # Dữ liệu đã xử lý
│   ├── test_samples.json              # Mẫu test cố định
│   └── vietnamese_errors.json         # Lỗi tiếng Việt mẫu
│
├── outputs/                           # Kết quả thí nghiệm
│   ├── prompting_results/             # Kết quả từ prompting
│   │   ├── predictions.json
│   │   └── metrics.json
│   ├── adapters/                      # LoRA adapters sau finetune
│   ├── logs/                          # Log training
│   └── metrics/                       # Metrics đánh giá
│
├── scripts/                           # Script chạy
│   ├── setup.sh                       # Cài đặt môi trường
│   ├── run_prompting_test.sh          # Test prompting
│   ├── run_prepare_data.sh            # Chuẩn bị dữ liệu finetune
│   ├── run_train.sh                   # Chạy finetune
│   ├── run_eval_prompting.sh          # Đánh giá prompting
│   ├── run_eval_finetuned.sh          # Đánh giá sau finetune
│   ├── run_deploy.sh                  # Triển khai model
│   └── tail_log.sh                    # Xem log realtime
│
├── src/                               # Source code
│   ├── cli/                           # Command line interface
│   ├── data/                          # Xử lý dữ liệu
│   │   └── prompt_templates.py        # Prompt templates cho grammar correction
│   ├── evaluation/                    # Đánh giá model
│   │   └── metrics.py                 # GLEU, CER, BERTScore
│   ├── inference/                     # Inference
│   │   ├── generate_base.py           # Chạy với base model (prompting)
│   │   └── generate_finetuned.py      # Chạy với finetuned model
│   ├── training/                      # Finetune với Unsloth
│   └── utils/                         # Utility functions
│
├── tests/                             # Unit tests
├── docker/                            # Docker cho deployment
├── .github/                           # CI/CD
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Cài Đặt Nhanh

```bash
# 1. Clone repository
git clone https://github.com/yourusername/grammar_correction.git
cd grammar_correction

# 2. Chạy script cài đặt
bash scripts/setup.sh

wsl -d Ubuntu-22.04

# 3. Activate environment
source .venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

---

## 📖 Hướng Dẫn Chi Tiết

### Bước 1: Cài Đặt Môi Trường

```bash
bash scripts/setup.sh
```

Script này sẽ:

- Tạo virtual environment
- Cài đặt PyTorch với CUDA support
- Cài đặt Unsloth, transformers, vLLM
- Cài đặt các thư viện cần thiết khác

### Bước 2: Test Prompting với Model Nhỏ

**Chạy test với model nhỏ nhất trước:**

```bash
# Test với Qwen2.5-1.5B (khuyến nghị cho RTX 3050 6GB)
bash scripts/run_prompting_test.sh --model Qwen/Qwen2.5-1.5B-Instruct

# Hoặc test với model 3B nếu có 8GB VRAM
bash scripts/run_prompting_test.sh --model Qwen/Qwen2.5-3B-Instruct

# Test với Phi-3-mini (3.8B) - cần 8GB VRAM
bash scripts/run_prompting_test.sh --model microsoft/Phi-3-mini-4k-instruct
```

**File test mẫu (`data/test_samples.json`):**

```json
[
  {
    "id": "1",
    "input": "Toi di hoc bang xe dap.",
    "expected": "Tôi đi học bằng xe đạp.",
    "language": "vi"
  },
  {
    "id": "2",
    "input": "She go to school yesterday.",
    "expected": "She went to school yesterday.",
    "language": "en"
  },
  {
    "id": "3",
    "input": "Anh ây rat thong minh.",
    "expected": "Anh ấy rất thông minh.",
    "language": "vi"
  }
]
```

**Prompt templates mặc định:**

```python
# src/data/prompt_templates.py

PROMPTS = {
    "simple": """Sửa lỗi chính tả và ngữ pháp trong câu sau. Chỉ trả về câu đã sửa, không giải thích.

Câu gốc: {text}
Câu đã sửa:""",

    "few_shot": """Sửa lỗi chính tả và ngữ pháp trong câu.

Ví dụ 1:
Input: "Toi yeu Ha Noi."
Output: "Tôi yêu Hà Nội."

Ví dụ 2:
Input: "Em hoc bai o nha."
Output: "Em học bài ở nhà."

Input: {text}
Output:""",

    "english": """Fix grammar and spelling errors in the following sentence. Output only the corrected sentence.

Input: {text}
Output:"""
}
```

### Bước 3: Đánh Giá Kết Quả Prompting

```bash
bash scripts/run_eval_prompting.sh
```

**Metrics đánh giá:**

| Metric        | Mô Tả                       | Ngưỡng Tốt |
| ------------- | --------------------------- | ---------- |
| **GLEU**      | Độ chính xác của câu đã sửa | > 0.85     |
| **CER**       | Character Error Rate        | < 0.10     |
| **WER**       | Word Error Rate             | < 0.15     |
| **BERTScore** | Độ tương đồng ngữ nghĩa     | > 0.92     |

**Xem kết quả:**

```bash
cat outputs/prompting_results/metrics.json
```

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_template": "simple",
  "metrics": {
    "gleu": 0.87,
    "cer": 0.08,
    "wer": 0.12,
    "bertscore": 0.94
  },
  "inference_time_ms": 245.3,
  "tokens_per_second": 45.2
}
```

**📌 Quyết định có finetune hay không:**

- ✅ **Prompting đủ tốt** (GLEU > 0.85, CER < 0.10) → Dùng prompting, không cần finetune
- ⚠️ **Cần cải thiện** (GLEU 0.70-0.85) → Thử prompt khác, thêm few-shot examples
- 🔴 **Prompting yếu** (GLEU < 0.70) → Cần finetune

### Bước 4: (Tuỳ Chọn) Finetune với LoRA

**Chỉ finetune nếu prompting không đạt yêu cầu.**

#### 4.1 Chuẩn bị dữ liệu finetune

```bash
# Format dữ liệu cho finetune
bash scripts/run_prepare_data.sh
```

Dữ liệu finetune format:

```json
[
  {
    "instruction": "Sửa lỗi chính tả và ngữ pháp trong câu sau.",
    "input": "Toi di hoc bang xe dap.",
    "output": "Tôi đi học bằng xe đạp."
  },
  {
    "instruction": "Correct the grammar and spelling errors.",
    "input": "She go to school yesterday.",
    "output": "She went to school yesterday."
  }
]
```

#### 4.2 Chạy finetune

```bash
# Finetune với model 1.5B (phù hợp RTX 3050 6GB)
bash scripts/run_train.sh --model Qwen/Qwen2.5-1.5B-Instruct

# Hoặc với model 3B (cần 8GB VRAM)
bash scripts/run_train.sh --model Qwen/Qwen2.5-3B-Instruct
```

**Cấu hình finetune mặc định:**

```yaml
# configs/train_finetune.yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  load_in_4bit: true # Quantization để tiết kiệm VRAM

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  batch_size: 4
  gradient_accumulation: 2
  learning_rate: 2e-4
  epochs: 3
  max_seq_length: 512
```

#### 4.3 Theo dõi training

```bash
# Xem log realtime
bash scripts/tail_log.sh

# Hoặc dùng tensorboard
tensorboard --logdir outputs/logs/
```

### Bước 5: So Sánh Kết Quả

```bash
# Đánh giá model đã finetune
bash scripts/run_eval_finetuned.sh

# So sánh prompting vs finetuned
python src/evaluation/compare.py
```

**Kết quả so sánh:**

```
========================================
COMPARISON: Prompting vs Finetuned
========================================

Metric        | Prompting | Finetuned | Improvement
--------------|-----------|-----------|------------
GLEU          | 0.87      | 0.94      | +0.07
CER           | 0.08      | 0.04      | -0.04
WER           | 0.12      | 0.06      | -0.06
BERTScore     | 0.94      | 0.97      | +0.03

Inference Time: Prompting 245ms, Finetuned 267ms (+9%)
```

### Bước 6: Triển Khai Model

#### 6.1 Deploy với vLLM (tốc độ cao)

```bash
bash scripts/run_deploy.sh
```

#### 6.2 API endpoint

```bash
# Model sẽ chạy tại http://localhost:8000

# Test API
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Sửa lỗi: Toi di hoc bang xe dap.",
    "max_tokens": 100
  }'
```

#### 6.3 Docker deployment

```bash
# Build image
docker build -t grammar-correction -f docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 grammar-correction
```

---

## ⚙️ Cấu Hình

### Chọn Model

```yaml
# configs/eval_prompting.yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct" # Thay đổi ở đây
  trust_remote_code: true
  device_map: "auto"
```

### Thay Đổi Prompt Template

```python
# src/data/prompt_templates.py
# Thêm prompt mới ở đây
PROMPTS["vietnamese_formal"] = """Bạn là chuyên gia ngôn ngữ tiếng Việt.
Hãy sửa lỗi chính tả và ngữ pháp trong câu dưới đây.

Câu cần sửa: {text}
Câu đã sửa:"""
```

### Điều Chỉnh VRAM

```yaml
# configs/eval_prompting.yaml
model:
  load_in_4bit: true # Bật để tiết kiệm VRAM
  load_in_8bit: false # Tắt nếu muốn tăng độ chính xác
```

---

## 🛠️ Xử Lý Lỗi

| Lỗi                  | Nguyên Nhân                 | Cách Fix                                                                    |
| -------------------- | --------------------------- | --------------------------------------------------------------------------- |
| `CUDA out of memory` | Model quá lớn so với VRAM   | Giảm batch size, bật 4bit quantization, dùng model nhỏ hơn                  |
| `Model not found`    | Tên model sai               | Kiểm tra tên trên HuggingFace                                               |
| `Slow inference`     | Model chạy trên CPU         | Kiểm tra CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
| `Tokenizer error`    | Tokenizer không tương thích | Thêm `trust_remote_code: true` trong config                                 |

### Kiểm Tra GPU

```bash
# Kiểm tra VRAM trống
nvidia-smi

# Kiểm tra PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

---

## 📈 Kế Hoạch Mở Rộng

### Giai Đoạn 1: RTX 3050 (Local)

- [x] Prompting với Qwen2.5-1.5B
- [ ] Thêm few-shot examples cho tiếng Việt
- [ ] Tối ưu prompt templates

### Giai Đoạn 2: Finetune (Cloud GPU)

- [ ] Finetune với dataset tiếng Việt (10k+ samples)
- [ ] Test với model 7B (dùng cloud)
- [ ] A/B test prompting vs finetuned

### Giai Đoạn 3: Production

- [ ] Deploy với vLLM
- [ ] Thêm API rate limiting
- [ ] Monitoring với Prometheus + Grafana
- [ ] CI/CD pipeline

---

## 📊 Dataset Gợi Ý

| Dataset       | Ngôn Ngữ    | Số Lượng | Link                                                    |
| ------------- | ----------- | -------- | ------------------------------------------------------- |
| **VnCoreNLP** | Tiếng Việt  | 10k+     | [Link](https://github.com/vncorenlp/VnCoreNLP)          |
| **VLSP 2021** | Tiếng Việt  | 5k       | [Link](https://vlsp.org.vn/)                            |
| **FCE**       | Tiếng Anh   | 34k      | [Link](https://www.cl.cam.ac.uk/research/nl/bea2019st/) |
| **Lang8**     | Đa ngôn ngữ | 1M+      | [Link](https://github.com/fei-cheng/lang8-l1)           |

---

## 📝 License

MIT License - Free for commercial and personal use.

---

## 🙏 Tài Liệu Tham Khảo

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [vLLM](https://vllm.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 🎯 Fine-tuning cho Văn Bản Dài & Dataset Tiếng Việt

Khi xử lý văn bản dài (>5 câu hoặc nhiều trang), phương pháp **Fine-tuning (LoRA/QLoRA)** mang lại độ chính xác cao hơn rõ rệt so với zero-shot / few-shot prompting:

| Loại input           | Khả năng xử lý qua Prompting (Qwen 1.5B) | Giải pháp khuyến nghị |
| -------------------- | ---------------------------------------- | --------------------- |
| Câu ngắn (1-2 câu)   | ✅ Tốt                                   | Prompting / Base Model |
| Đoạn văn (3-5 câu)   | ⚠️ Trung bình (dễ lệch ngữ cảnh)         | Few-shot / LoRA Fine-tune |
| Văn bản dài (>5 câu) | ❌ Kém (dễ hallucination)                | Fine-tuning bắt buộc  |

---

### Dataset Tiếng Việt cho Fine-tuning

| Dataset | Số lượng mẫu | Đặc điểm | Ứng dụng |
| --- | --- | --- | --- |
| **[ShynBui/Vietnamese_spelling_error](https://huggingface.co/datasets/ShynBui/Vietnamese_spelling_error)** | 422,209 | Quy mô lớn, đa dạng lỗi thực tế | Fine-tuning chính |
| **[PaulTran/vietnamese_spelling_error_detection](https://huggingface.co/datasets/PaulTran/vietnamese_spelling_error_detection)** | 10,000+ | Phân loại lỗi chi tiết (Telex/VNI) | Đánh giá & Benchmark |
| **[bmd1905/error-correction-vi](https://huggingface.co/datasets/bmd1905/error-correction-vi)** | 50,000+ | Dữ liệu biên tập từ báo chí | Huấn luyện bổ trợ |

---

### Định dạng dữ liệu huấn luyện (Instruction Format)

```json
[
  {
    "instruction": "Sửa lỗi chính tả và ngữ pháp trong câu sau.",
    "input": "côn viec kin doanh thì rất kho khan",
    "output": "Công việc kinh doanh thì rất khó khăn"
  },
  {
    "instruction": "Sửa lỗi chính tả và ngữ pháp trong câu sau.",
    "input": "toi dang là sinh diên nam hai",
    "output": "Tôi đang là sinh viên năm hai"
  }
]
```

---

### Lộ trình Fine-tuning mở rộng
1. **Tải dataset**: Sử dụng `ShynBui/Vietnamese_spelling_error` từ Hugging Face Hub.
2. **Format dữ liệu**: Chuyển đổi về cấu trúc instruction theo chuẩn ChatML / Alpaca.
3. **Mô hình nền**: `Qwen2.5-1.5B-Instruct` hoặc `Qwen2.5-3B-Instruct`.
4. **Tối ưu huấn luyện**: Tận dụng **LoRA / QLoRA 4-bit** với **Unsloth** để kiểm soát VRAM dưới 6-8GB.
5. **Đánh giá benchmark**: Đo lường định lượng qua bộ chỉ số **GLEU**, **CER/WER**, và **BERTScore**.

