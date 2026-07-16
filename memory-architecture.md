# Kiến trúc Tích hợp Memory Layer

**Ngày cập nhật:** 2026-07-15  
**Phạm vi:** Tích hợp đầy đủ lớp bộ nhớ Redis + Qdrant cho hệ thống bot đa tác vụ

## 1. Mục tiêu kiến trúc

Hệ thống memory layer được thiết kế để:
- Ghi nhớ ngữ cảnh dài hạn theo từng người dùng.
- Truy xuất thông tin liên quan cho truy vấn mới.
- Cá nhân hóa câu trả lời dựa trên lịch sử tương tác.
- Giảm gọi API lặp lại nhờ cơ chế cache.

Nguyên tắc cốt lõi:
- Truy xuất theo ngữ nghĩa (semantic retrieval).
- Phân tách dữ liệu theo `user_id` và `source_type`.
- Suy giảm độ quan trọng theo thời gian (importance decay).
- Kết hợp nhiều lớp nhớ: Redis (nhanh, ngắn hạn) + Qdrant (ngữ nghĩa, dài hạn).

## 2. Trạng thái hiện tại

Hiện tại luồng orchestrator đã có vị trí đọc/ghi memory nhưng đang ở mức stub:
- `rehydrate_context`: chưa truy xuất thực tế.
- `episodic_writer`: chưa ghi thực tế.

Kiến trúc hiện tại:
- User Message -> Orchestrator Graph -> Agents -> Reply
- Đã có hạ tầng Redis/Qdrant client nhưng chưa đi vào luồng xử lý đầy đủ.

## 3. Kiến trúc đích

Luồng xử lý mục tiêu:
1. User gửi tin nhắn.
2. `rehydrate_context` đọc ngữ cảnh:
   - Working memory (Redis `wm:`, mục 11): nạp lại N lượt hội thoại gần nhất + sticky module nếu còn hiệu lực.
   - Cache (Redis, mục 5.2): kiểm tra cache truy vấn chính xác.
   - Qdrant: tìm kiếm hybrid dense (cosine) + sparse (BM25), trọng số 0.7/0.3, filter `user_id` + `source_type` trước, re-rank top-20 bằng cross-encoder.
3. Router phân loại intent (LLM multi-label) — nếu có `sticky_module` còn hiệu lực từ bước 2, ưu tiên route lại module đó thay vì luôn phân loại lại từ đầu.
4. Dispatch agent song song theo intent, mỗi agent nhận memory đã lọc theo nguồn phù hợp.
5. `format_response` hợp nhất phản hồi.
6. Ghi lại kết quả (2 node độc lập, phục vụ mục tiêu khác nhau):
   - `working_memory_writer` (mục 11): append lượt hội thoại vào `wm:{user_id}:turns`, refresh/set sticky, refresh TTL trượt.
   - `episodic_writer`: chunking ngữ nghĩa (200-400 token, overlap 50 token) → tạo embedding → upsert Qdrant → cache Redis theo TTL.
7. Trả kết quả cho người dùng.

## 4. Trigger points trong hệ thống

- Trigger A (READ context): `orchestrator/graph.py` tại node `rehydrate_context` — đọc cả working memory (turns/sticky) lẫn cache/Qdrant.
- Trigger B (WRITE episodic): `orchestrator/graph.py` tại node `episodic_writer`.
- Trigger C (CACHE AGENT): trong từng module agent (đặc biệt search).
- Trigger D (NODE CACHE): trong các node xử lý chi tiết như crawl URL.
- Trigger E (WRITE session): node mới `working_memory_writer` (mục 11), chạy cùng lúc với Trigger B.

## 5. Thành phần kỹ thuật chính

### 5.1 Embedding Provider
- Tạo module embedding riêng (`app/infra/providers/embedding.py`).
- Dùng mô hình `multilingual-e5-large` (1024 chiều).
- Cung cấp API:
  - `init_embeddings()`
  - `embed_text()`
  - `embed_texts()`
  - `close_embeddings()`

### 5.2 Redis — Cache Layer
- Cache truy vấn gần đây (TTL ngắn, ví dụ 1 giờ).
- Cache kết quả xử lý/nội dung crawl (TTL trung bình, ví dụ 4-24 giờ).
- Ưu tiên đọc Redis trước để giảm độ trễ.
- Namespace tách biệt với working memory (mục 11): cache theo key nội dung (hash query/URL), không theo `user_id`. Vai trò session/hội thoại (lịch sử lượt chat, sticky module) thuộc về working memory — xem mục 11, không mô tả lại ở đây.

### 5.3 Qdrant (Long-term Episodic Memory)
- Lưu chunk văn bản + vector embedding + metadata.
- Metadata chuẩn:
  - `user_id`, `source_type`, `timestamp`, `importance`, `language`, `topics`
  - Mở rộng theo intent: `mood_score`, `urls`, `category`, `amount`...
- Truy xuất hybrid: dense (cosine) + sparse (BM25), trọng số kết hợp 0.7/0.3.
- Luôn filter theo `user_id` trước khi tính similarity.
- Re-rank top-20 kết quả bằng cross-encoder trước khi đưa vào context assembly (giảm nhiễu so với chỉ dùng similarity thô).

## 6. Dữ liệu và truyền context

### 6.1 Khi đọc context
- Input: `user_id` + câu hỏi hiện tại.
- Output: `retrieved_memories[]` gồm payload + score + source type.

### 6.2 Khi fan-out cho agent
- Mỗi agent chỉ nhận memory thuộc `source_type` tương ứng.
- Tránh nhiễu context chéo giữa các module.

### 6.3 Khi ghi episodic
- Ghi theo từng intent đã xử lý.
- Mỗi đoạn nội dung sau chunking sẽ tạo 1 point riêng trong Qdrant.

## 7. Nâng cấp theo pha

### Pha 1: Foundation
- Tạo embedding client.
- Kích hoạt init/close ở vòng đời ứng dụng.
- Cài đặt `episodic_writer` + helper extract/chunk.

### Pha 2: Orchestrator Integration
- Cài đặt `rehydrate_context` đọc Redis (cache + working memory) + Qdrant.
- Cập nhật `fan_out` để lọc memory theo agent, đọc `sticky_module` nếu có.
- Cài đặt working memory (mục 11): helper `wm:` trong `app/infra/memory/working.py`, node `working_memory_writer`.

### Pha 3: Module Enhancements
- Search agent hỗ trợ cache hit và pre-populate documents.
- Crawl node có cache theo URL.
- Summary node tăng chất lượng nhờ context đã truy xuất.

### Pha 4: Advanced Memory Ops
- Job suy giảm `importance` hằng ngày: giảm 10%/30 ngày nếu điểm nhớ không được truy xuất lại (dùng cron 03:00 daily đã đăng ký trong `app/bot/scheduler.py`).
- Job nén memory hằng tuần (cron Chủ nhật 04:00): gộp các chunk có `importance` dưới ngưỡng 0.2 thành bản tóm tắt.
- Working memory (Redis session) không thuộc phạm vi decay này — nó tự hết hạn qua TTL trượt, không cần job riêng (mục 11.3, 11.6).

## 8. Độ tin cậy và khả năng suy giảm mềm

Yêu cầu vận hành:
- Mọi thao tác memory phải có `try/except`, không làm bot crash.
- Nếu Redis/Qdrant lỗi: hệ thống vẫn trả lời theo luồng không-memory.
- Logging rõ ràng cho cache hit/miss, retrieval, write success/fail.

## 9. Chỉ số đánh giá thành công

- Cache hit rate > 50% cho truy vấn lặp.
- Giảm độ trễ > 30% khi cache hit.
- Không hồi quy hiệu năng khi memory tắt.
- Tỉ lệ thao tác memory thành công ở mức cao (mục tiêu 99.9%).

## 10. Danh sách file tác động chính

- `app/infra/providers/embedding.py` (mới)
- `app/infra/memory/working.py` (thêm helper turns/sticky, mục 11.7)
- `app/main.py`
- `app/orchestrator/graph.py` (rehydrate/episodic + node mới `working_memory_writer`)
- `app/orchestrator/router.py` (đọc `sticky_module` từ metadata, mục 11.7)
- `app/modules/search/agent.py`
- `app/modules/search/node/crawl_node.py`
- `app/modules/search/node/summary_node.py`
- `app/bot/scheduler.py`
- `pyproject.toml`

## 11. Kiến trúc xử lý Working Memory (session ngắn hạn)

### 11.1 Vấn đề hiện tại

- `bot/handlers.py:handle_text` khởi tạo `AgentState` mới hoàn toàn cho mỗi tin nhắn (chỉ có 1 `HumanMessage`) — không có lịch sử hội thoại nào được nạp lại giữa các lượt.
- `app/infra/memory/working.py` hiện chỉ expose 1 Redis client thô (`init_redis`/`get_redis_client`/`close_redis`) — chưa có quy ước key hay helper đọc/ghi session nào.
- `config.yaml` của từng module đã khai báo `routing.sticky` + `routing.sticky_turns`, nhưng `orchestrator/router.py` (`classify_intents`, `resolve_agent_names`) hoàn toàn không đọc các field này — sticky routing chưa có nơi lưu trạng thái để hoạt động được.

Cache (mục 5.2) và working memory là 2 vai trò khác nhau trên cùng hạ tầng Redis: cache tránh gọi lại API tốn kém (theo key nội dung), còn working memory phải giữ **trạng thái phiên theo user** (lịch sử hội thoại, sticky module) để nối liền ngữ cảnh giữa các lượt nhắn tin. Phần dưới đây thiết kế riêng cho working memory.

### 11.2 Phân biệt Working Memory (session) vs Cache (mục 5.2)

| Khía cạnh | Working Memory (session) | Cache (5.2) |
|---|---|---|
| Mục đích | Nối ngữ cảnh hội thoại + sticky intent giữa các lượt | Tránh gọi lại API/embedding tốn kém |
| Khóa | theo `user_id` (namespace `wm:`) | theo nội dung (hash query/URL) |
| TTL | Trượt theo phiên hoạt động (vd 1800s, refresh mỗi lượt) | Cố định theo loại dữ liệu (1h / 4-24h) |
| Nội dung | N lượt hội thoại gần nhất + trạng thái sticky module | Kết quả truy vấn/crawl thô |
| Ghi bởi | Node mới `working_memory_writer` (chạy cạnh `episodic_writer`) | Từng agent/node tự cache khi cần |
| Đọc bởi | `rehydrate_context` (đầu request) và `_fan_out`/`route` (đọc sticky) | Agent/node tương ứng trước khi gọi API ngoài |

### 11.3 Cấu trúc key và schema Redis (namespace riêng: `wm:`)

- `wm:{user_id}:turns` — danh sách N lượt gần nhất (vd tối đa 10), mỗi phần tử `{"role": "human"|"ai", "content": str, "ts": <iso8601>}`. Khi vượt ngưỡng: cắt bớt lượt cũ nhất (trim), không tóm tắt tự động ở giai đoạn foundation.
- `wm:{user_id}:sticky` — hash `{"module": <agent_name>, "turns_left": <int>}`, chỉ được set khi module vừa trả lời có `routing.sticky: true` trong `config.yaml`; giảm `turns_left` mỗi lượt, xoá khi về 0 hoặc khi người dùng được route rõ ràng sang module khác.
- TTL trượt: mỗi lần đọc hoặc ghi đều `EXPIRE` lại các key trên theo session timeout — hết hạn nghĩa là hết phiên, không cần job dọn dẹp riêng (khác với decay/compress ở mục 7, vốn áp dụng cho Qdrant).

### 11.4 Luồng đọc — mở rộng `rehydrate_context`

1. Đọc `wm:{user_id}:turns`, dựng lại lịch sử `messages` (prepend trước `HumanMessage` mới nhất) để router và agent có ngữ cảnh multi-turn thay vì chỉ thấy 1 câu đơn lẻ.
2. Đọc `wm:{user_id}:sticky`; nếu `turns_left > 0`, đưa `module` vào `state["metadata"]["sticky_module"]` để `_fan_out`/`classify_intents` ưu tiên route lại đúng module đó thay vì luôn phân loại lại từ đầu bằng LLM.
3. Nếu Redis lỗi hoặc miss (phiên mới) → coi như không có lịch sử, không chặn luồng chính, tuân theo nguyên tắc suy giảm mềm ở mục 8.

### 11.5 Luồng ghi — node mới `working_memory_writer`

- Chạy sau `format_response`, song song về mặt logic với `episodic_writer` (2 node độc lập, không phụ thuộc nhau vì phục vụ mục tiêu khác nhau: 1 cái là ngắn hạn/theo phiên, 1 cái là dài hạn/toàn cục).
- Append lượt `human` (câu hỏi) và lượt `ai` (câu trả lời cuối) vào `wm:{user_id}:turns`, trim nếu vượt N lượt.
- Nếu agent vừa dispatch thuộc module có `routing.sticky: true` → set/refresh `wm:{user_id}:sticky` với `turns_left = routing.sticky_turns` đọc từ `config.yaml` của module đó.
- Refresh TTL trượt cho toàn bộ key `wm:{user_id}:*`.

### 11.6 Quan hệ với Episodic Memory (Qdrant, mục 5.3)

- Working memory không tự động "thăng hạng" (promote) lên Qdrant — đây là 2 lớp độc lập, khác payload và vòng đời.
- Chỉ nội dung đã qua `episodic_writer` (chunking + embedding) mới vào Qdrant; working memory chỉ tồn tại trong phiên Redis và biến mất khi hết TTL.
- Nếu sau này cần giữ lại tóm tắt hội thoại quan trọng sau khi phiên hết hạn, cần một bước tổng hợp (summarize) riêng trước khi key bị xoá — có thể coi là mở rộng của Pha 4 (mục 7) nhưng ở cấp phiên hội thoại thay vì cấp chunk văn bản.

### 11.7 Điểm chỉnh sửa cần thiết khi triển khai (tham khảo cho phase sau, chưa thực hiện)

- `app/infra/memory/working.py`: bổ sung helper `get_recent_turns()`, `append_turn()`, `get_sticky()`, `set_sticky()` bên cạnh client Redis thô hiện có.
- `app/orchestrator/graph.py`: `_rehydrate_context` đọc turns + sticky; thêm node `working_memory_writer` chạy sau `_format_response`.
- `app/orchestrator/router.py`: `_fan_out` và/hoặc `classify_intents` cần đọc `state["metadata"]["sticky_module"]` để tôn trọng sticky routing thay vì luôn phân loại lại từ đầu bằng LLM.

## 12. Kết luận

Kiến trúc memory layer hướng tới một hệ thống bot có khả năng ghi nhớ dài hạn, truy xuất ngữ nghĩa chính xác theo từng người dùng, và tối ưu chi phí/độ trễ bằng cache đa lớp. Lộ trình triển khai theo pha giúp giảm rủi ro, đảm bảo tương thích ngược và cho phép mở rộng dần từ nền tảng đến tính năng nâng cao.
