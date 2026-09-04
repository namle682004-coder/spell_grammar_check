# Spell & Grammar Correction with LLM

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![vLLM](https://img.shields.io/badge/vLLM-Serving-4B8BBE)](https://vllm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/namle682004-coder/spell_grammar_check/actions/workflows/ci.yml/badge.svg)](https://github.com/namle682004-coder/spell_grammar_check/actions)

Production-ready API and machine learning pipeline for automated Vietnamese and English spelling and grammar correction. The system integrates parameter-efficient fine-tuning (PEFT/QLoRA), high-throughput vLLM serving, comprehensive NLP evaluation metrics (GLEU, CER/WER, BERTScore), and an enterprise REST API with multi-tier quota management.

---

## 📌 Key Features

- **Dual-Phase Correction Engine:**
  - **Prompt-First Baseline:** Zero-shot and few-shot inference for compact SLMs (Qwen2.5-1.5B/3B, Phi-3 Mini).
  - **PEFT / QLoRA Pipeline:** 4-bit quantized Low-Rank Adaptation using **Unsloth** and **TRL (SFTTrainer)** to minimize VRAM footprints during fine-tuning.
- **High-Throughput Inference:** Continuous batching and PagedAttention powered by **vLLM** for low-latency production deployment.
- **NLP Evaluation Harness:** Automated quantitative benchmarking measuring **GLEU** (Grammar Error Correction), **Character & Word Error Rate (CER/WER)**, and **BERTScore** to prevent semantic drift.
- **Enterprise SaaS API:** Asynchronous **FastAPI** backend with PostgreSQL persistence, dual authentication (API Key & JWT Bearer), tiered rate limiting, and token quota tracking.
- **Production Infrastructure:** Fully containerized stack with **Docker Compose**, schema versioning via **Alembic**, and modern Python package management using **uv**.

---

## 🏗️ Architecture

```
[ Client / Web App ]
         │
         ▼  (HTTP / HTTPS)
┌──────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                     │
│  - Dual Auth: API Key (`X-API-Key`) & JWT Bearer         │
│  - Middleware: Quota enforcement, Rate limiting, Logging │
│  - Routing: `/v1/check`, `/v1/auth`, `/v1/quota`         │
└──────────────┬────────────────────────────┬──────────────┘
               │                            │
      (Read / Write State)          (Forward Inference)
               │                            │
               ▼                            ▼
┌──────────────────────────────┐ ┌───────────────────────────┐
│     PostgreSQL Database      │ │      Inference Engine     │
│  - User profiles & API keys  │ │  - Primary: vLLM Server   │
│  - Quota allocations         │ │  - Fallback: Local Engine │
│  - Request logs & audits     │ │  - In-memory response LRU │
└──────────────────────────────┘ └───────────────────────────┘
```

---

## 📁 Repository Structure

```
spell_grammar_check/
├── configs/                     # Experiment, training, and evaluation configurations
│   ├── eval_prompting.yaml      # Baseline prompting parameters
│   ├── eval_finetuned.yaml      # Fine-tuned model evaluation setup
│   └── train_finetune.yaml      # LoRA / QLoRA training hyperparameters
├── data/                        # Datasets, raw inputs, and processed evaluation sets
│   ├── test_samples.json        # Standardized benchmark samples
│   └── vietnamese_errors.json   # Curated Vietnamese error test samples
├── docker/                      # Container definitions
│   └── Dockerfile               # Production multi-stage Docker build
├── docker-compose.yml           # Multi-service composition (API, DB, vLLM)
├── pyproject.toml               # Project dependencies, build system, and tool configs
├── scripts/                     # Operational automation scripts
│   ├── migrate_db.sh            # Database migration executor
│   ├── run_deploy_vllm.sh       # vLLM inference server runner
│   ├── run_eval_base.sh         # Base model evaluation trigger
│   ├── run_eval_finetuned.sh    # Fine-tuned model evaluation trigger
│   ├── run_train.sh             # Training execution script
│   └── setup.sh                 # Environment initialization script
├── src/
│   ├── api/                     # FastAPI routing, middleware, and dependencies
│   ├── cli/                     # CLI entry points for data, train, eval, and compare
│   ├── data/                    # Data loaders, prompt templates, and preprocessors
│   ├── deployment/              # Deployment specs (K8s, Terraform, vLLM runner)
│   ├── evaluation/              # Benchmark harnesses (GLEU, CER/WER, BERTScore)
│   ├── inference/               # Model predictors, decoding, batching, and cache
│   ├── services/                # Core business logic (auth, usage, LLM service)
│   ├── storage/                 # Database models, Alembic migrations, repositories
│   └── training/                # LoRA attachment, Unsloth loader, trainer setup
└── tests/                       # Unit, integration, and benchmark test suites
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Linux or WSL2 (Ubuntu 22.04 recommended)
- Python `>= 3.11`
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA GPU with CUDA 12.1+ (Optional for local inference; CPU supported for API and unit tests)

### 2. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/namle682004-coder/spell_grammar_check.git
cd spell_grammar_check

# Copy environment configuration
cp .env.example .env

# Install Python 3.11 and sync dependencies via uv
uv python install 3.11
uv sync --group dev
```

### 3. Database Initialization

Start the PostgreSQL service and run Alembic migrations:

```bash
# Start PostgreSQL via Docker Compose
docker compose up -d spell_grammar_db

# Run schema migrations
bash scripts/migrate_db.sh
```

### 4. Launch API Server

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

Interactive API documentation:
- **Swagger UI:** `http://localhost:8080/docs`
- **ReDoc:** `http://localhost:8080/redoc`

---

## 🐳 Docker Deployment

To spin up the multi-container stack (API, PostgreSQL, optional vLLM):

```bash
# Launch background services
docker compose up -d

# Inspect service logs
docker compose logs -f api
```

| Service | Container Port | Host Port | Description |
| :--- | :--- | :--- | :--- |
| **FastAPI** | `8080` | `8080` | REST API Gateway (`/v1/check`, auth, quota) |
| **PostgreSQL** | `5432` | `5432` | Relational persistence for accounts, keys, audits |
| **vLLM** *(optional)* | `8000` | `8000` | Continuous batching inference engine |

---

## 📡 API Reference

### Health Check

```bash
curl -s http://localhost:8080/health
```

```json
{
  "status": "healthy"
}
```

### Spelling & Grammar Correction (`POST /v1/check`)

Authentication requires an API key in the `X-API-Key` header:

```bash
curl -X POST http://localhost:8080/v1/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sg_your_api_key_here" \
  -d '{
    "text": "Toi di hoc bang xe dap.",
    "model": "qwen2.5-1.5b",
    "type": "full"
  }'
```

**Response Example:**
```json
{
  "success": true,
  "original_text": "Toi di hoc bang xe dap.",
  "corrected_text": "Tôi đi học bằng xe đạp.",
  "total_errors": 4,
  "errors": [
    {
      "original": "Toi",
      "replacement": "Tôi",
      "type": "spelling",
      "position": 0
    },
    {
      "original": "di",
      "replacement": "đi",
      "type": "spelling",
      "position": 4
    },
    {
      "original": "hoc",
      "replacement": "học",
      "type": "spelling",
      "position": 7
    },
    {
      "original": "bang",
      "replacement": "bằng",
      "type": "spelling",
      "position": 11
    }
  ],
  "processing_time_ms": 182,
  "tokens_used": 28
}
```

---

## 🧪 Model Pipeline: Training & Evaluation

### 1. Data Preparation
Format raw datasets into instruction-following JSON schema:

```bash
uv run python -m src.cli.prepare_data \
  --input-path data/raw/shynbui \
  --output-dir data/processed/shynbui
```

### 2. Baseline Prompting Evaluation
Evaluate zero-shot and few-shot performance on the base model:

```bash
bash scripts/run_eval_base.sh
```

Results are exported to `outputs/prompting_results/metrics.json`.

### 3. Model Fine-Tuning (QLoRA 4-bit)
Train parameter-efficient LoRA adapters using Unsloth:

```bash
bash scripts/run_train.sh --config configs/train_finetune.yaml
```

**Hyperparameters (`configs/train_finetune.yaml`):**
```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  load_in_4bit: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 2e-4
  num_train_epochs: 3
  max_seq_length: 512
```

### 4. Benchmark & Comparison
Evaluate and benchmark model predictions against gold references:

```bash
# Evaluate fine-tuned checkpoint
bash scripts/run_eval_finetuned.sh

# Run comparative benchmark report
uv run python -m src.cli.compare
```

**NLP Evaluation Metrics:**
- **GLEU (Google-BLEU):** Sentence-level n-gram precision/recall for grammar corrections.
- **CER / WER:** Character Error Rate and Word Error Rate against gold references.
- **BERTScore:** Semantic preservation check to prevent hallucination or semantic drift.

---

## 🚀 Model Serving with vLLM

To serve fine-tuned weights using the high-throughput vLLM engine:

```bash
export MODEL_PATH=outputs/adapters/vietnamese-grammar-qwen1.5b-shynbui-lora/final_model
export VLLM_PORT=8000

bash scripts/run_deploy_vllm.sh
```

Point the API gateway to vLLM by configuring `VLLM_BASE_URL=http://localhost:8000` in `.env`.

---

## 🛠️ Testing & Code Quality

```bash
# Run unit and integration tests
uv run pytest tests/ -v

# Run linter
uv run ruff check src tests

# Check code formatting
uv run ruff format --check src tests
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
