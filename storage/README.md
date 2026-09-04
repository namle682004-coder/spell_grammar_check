# Usage & user data storage

Thư mục này lưu dữ liệu runtime từ API (không commit vào git).

## Cấu trúc

```
storage/
├── api_requests/          # Lịch sử gọi API (JSONL theo ngày)
├── llm_usage/             # Token LLM mỗi request (JSONL theo ngày)
├── users/                 # Profile tổng hợp theo client_id
├── sessions/              # Session tracking (JSONL theo ngày)
├── errors/                # Request lỗi (JSONL theo ngày)
├── aggregates/daily/      # Thống kê gộp theo ngày
└── _examples/             # Mẫu schema record
```

## File naming

| Thư mục | Pattern | Ví dụ |
|---------|---------|-------|
| `api_requests/` | `YYYY-MM-DD.jsonl` | `2026-05-26.jsonl` |
| `llm_usage/` | `YYYY-MM-DD.jsonl` | `2026-05-26.jsonl` |
| `users/` | `{client_id}.json` | `a1b2c3d4e5f6g7h8.json` |
| `aggregates/daily/` | `YYYY-MM-DD.json` | `2026-05-26.json` |

## Client ID

- Mặc định: hash 16 ký tự đầu của `API_KEY` (không lưu key thô).
- Tuỳ chọn: header `X-Client-Id` để gắn user/app riêng.
- Session: header `X-Session-Id` (dự phòng mở rộng).

## Cấu hình

```env
USAGE_STORAGE_ROOT=./storage
```

## Code (Python package — không đặt code vào thư mục này)

- ORM / Postgres: `src/storage/database.py`, `src/storage/models/`
- Ghi file JSONL: `src/storage/repository.py`
- Middleware: `src/api/middleware/usage_middleware.py`

> Thư mục `storage/` chỉ chứa **dữ liệu runtime** (jsonl/json).  
> Đừng tạo `storage/database.py` — dùng `src/storage/` để tránh lỗi import.

Mỗi response có header `X-Request-Id` để trace log.
