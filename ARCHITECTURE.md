# L3A Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

Hệ thống là **rule-based, tất định** (không dùng LLM): cùng dữ liệu MCP → cùng output. Mã nguồn:

| Module | Vai trò |
| --- | --- |
| `src/student_agent/workflow.py` | `solve_case()` + `Coordinator` (điều phối, adjudication, ghép output) |
| `src/student_agent/agents.py` | Specialist agents, phân quyền tool, policy decision, verifier |
| `src/student_agent/analysis.py` | Hàm thuần: scope dữ liệu theo timeline, facts, candidate issues |
| `src/student_agent/a2a.py` | Giao thức A2A in-process (envelope, bus, hop budget, timeout) |
| `src/student_agent/evidence.py` | `EvidenceRegistry` theo từng case |
| `src/student_agent/cli.py` | `day09 run`: session MCP riêng mỗi case, retry, trace buffer, concurrency |

## 1. System overview

```text
inputs/<case_id>.json
        │
        ▼
  Coordinator ──(tool discovery: list_tools)
        │ task_assigned: collect_order_facts
        ▼
  Order/item agent ── get_order, get_order_items ──► OrderFacts (scoped items)
        │ handoff: ORDER_FACTS_READY
        ▼
  Coordinator ──────────────┬───────────────────────────────┐   (tuần tự)
        │ collect_payment_facts                 assess_delivery│
        ▼                                                     ▼
  Payment agent                                   Shipment agent
   get_payment_timeline, get_refund_timeline       get_shipment_summary
   ► PaymentFacts (captures, mismatch, refunds)    ► ShipmentFacts (on_time / seller_delay /
        │ handoff: PAYMENT_<VERDICT>                 logistics_delay / not_delivered, conflicts)
        └──────────────────────┬──────────────────────────────┘ handoff: DELIVERY_<VERDICT>
                               ▼
  Coordinator: candidate_issues() + select_primary_issue()   (claim vs evidence)
        │ task_assigned: decide_resolution (notes: primary_issue, candidates)
        ▼
  Policy agent ── get_policy(policy_version) ──► policy_decided (status, action, refund, parties)
        │ handoff: <RECOMMENDED_ACTION>
        ▼
  (chỉ khi seller chịu trách nhiệm) Order/item agent ── get_sellers ──► SELLERS_CONFIRMED
        ▼
  Coordinator ghép draft output ──► Verifier ──► verification_completed ──► case_finalized
                                                                   │
                         outputs/<case_id>.json  +  traces/trace.jsonl
```

Mọi mũi tên giữa agent là một A2A message và được ghi vào trace (`task_assigned` / `handoff`).
Mọi kết quả tool được agent dùng đều sinh `tool_result_consumed` kèm `evidence_ref`.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Output/handoff |
| --- | --- | --- | --- |
| Coordinator (`coordinator`) | Input case, kết quả các specialist | Tool discovery; giao việc; chọn `primary_issue` từ candidate issues (evidence thắng lời khai); ghép output; hạ confidence nếu verifier fail | `task_assigned` tới từng agent; output cuối; `case_received` / `case_finalized` |
| Order/item (`order-agent`) | `claimed_order_id` | Lấy order, kiểm tra `order_id` khớp; scope item theo timeline; tổng tiền, freight, seller (từ dòng item) | `OrderFacts` → `handoff ORDER_FACTS_READY` / `ORDER_NOT_FOUND` |
| Payment (`payment-agent`) | `OrderFacts` | Scope payment events quanh `order_approved_at`, refund events trong `[purchase, opened_at]`; phát hiện split hợp lệ, duplicate capture, reconciliation mismatch, refund pending/failed | `PaymentFacts` → `handoff PAYMENT_<VERDICT>` |
| Shipment (`shipment-agent`) | `OrderFacts` | Trễ hay đúng hạn từ timestamp của order; quy trách nhiệm seller (bàn giao carrier sau `shipping_limit`) hay logistics; đối chiếu shipment events, ghi conflict | `ShipmentFacts` → `handoff DELIVERY_<VERDICT>` |
| Policy (`policy-agent`) | `primary_issue`, facts | Tra rule `EC_POLICY_V1`: `case_status`, `recommended_action`, loại bên chịu trách nhiệm; tính refund **từ dữ liệu của chính order**; resolve `party_id` seller từ evidence (không copy seller ví dụ trong policy) | `policy_decided` + `handoff <ACTION>` |
| Verifier (`verifier`) | Draft output, evidence registry | Kiểm tra invariants (mục 6) | `verification_completed` `VERIFICATION_PASSED` / `VERIFICATION_FAILED` |

Phân quyền tool (`TOOL_SCOPES` trong `agents.py`, vi phạm → `PermissionError`):

| Actor | Tool được gọi |
| --- | --- |
| Coordinator | không (chỉ `list_tools`) |
| Order/item | `get_order`, `get_order_items`, `get_sellers` (chỉ khi seller chịu trách nhiệm) |
| Payment | `get_payment_timeline`, `get_refund_timeline` |
| Shipment | `get_shipment_summary` |
| Policy | `get_policy` |
| Verifier | không |

Không dùng `get_order_payments` (trùng thông tin với `get_payment_timeline`), `get_product_context` (không liên quan nghiệp vụ) và `get_customer_history` (order không trả `customer_unique_id`).

## 3. A2A protocol

- **Envelope** (`A2AMessage`): `case_id`, `sender`, `recipient`, `kind` (`task` | `result`), `task`, `payload`, `evidence_refs`, `decision_code`, `message_id`, `reply_to`.
- **Correlation**: mỗi case có một `A2ABus` riêng gắn với `case_id`; message mang `case_id` khác bị từ chối (`A2AError`). `reply_to` nối result với task.
- **Handoff**: `bus.request()` ghi `task_assigned` (actor → target, `decision_code` = tên task), gọi handler của agent đích, rồi ghi `handoff` (agent → coordinator, `decision_code` = kết luận, kèm `evidence_refs`). Handoff tới policy mang `notes` quan sát được: `primary_issue`, `claimed_topic`, `candidates`.
- **Điều kiện handoff**: payment/shipment chỉ chạy khi order-agent trả `OrderFacts`; policy chỉ chạy sau khi coordinator đã phán quyết primary issue.
- **Timeout**: mỗi lượt agent bị giới hạn `AGENT_TURN_TIMEOUT_S = 240s` (`asyncio.wait_for`).
- **Chống vòng lặp**: luồng là DAG cố định; thêm hop budget `MAX_HOPS_PER_CASE = 40` mỗi case (thực tế 10–12 hop).
- Trace chỉ chứa event, decision code và attribute số/đếm; không có nội dung suy luận.

## 4. Evidence lifecycle

1. `EvidenceGateway.call()` gửi `case_id` đúng của case, validate response theo `mcp-evidence-response-v1.schema.json`.
2. Agent đăng ký envelope vào `EvidenceRegistry` **tạo mới cho mỗi lần thử của mỗi case** (ref, tool, domain, data, result_hash). Ref được copy nguyên văn, không bao giờ tự tạo.
3. Sau khi phân tích, agent emit `tool_result_consumed` (actor, `tool_name`, `evidence_refs=[ref]`, attribute như số dòng in-scope/out-of-scope).
4. **Scope theo timeline order** trước khi dùng dữ liệu: payment events trong `[approved_at − 1h, approved_at + 24h]`; refund events trong `[purchase, opened_at]`; shipping limit trong `[purchase, estimated_delivery]`; shipment events trong `[purchase, max(opened_at, delivered) + 1 ngày]`; dòng trùng y hệt bị loại. Dòng ngoài scope không được dùng làm căn cứ.
5. Output chỉ trích evidence liên quan tới kết luận (`EVIDENCE_TOOLS[primary_issue]`), và chỉ các ref đã được consume trong trace. `claim_assessments[].evidence_refs` là tập con của `evidence_refs`.
6. Output và trace của một case chỉ được ghi khi lần thử đó thành công. Không có cache evidence giữa các case hay các lần chạy.

## 5. Failure policy

| Failure | Retry? | Fallback | Trace event/code |
| --- | --- | --- | --- |
| MCP timeout / mất kết nối (`httpx2.TransportError`, `MCPError`, `TimeoutError`, `ExceptionGroup` của chúng) | Có: làm lại **cả case** trên session mới, tối đa 3 lần, backoff 2s/4s (idempotent vì chỉ đọc) | Hết lượt → `day09 run` dừng với lỗi, không ghi output đoán | Lần thử lỗi bị bỏ (buffer), lần thành công có `case_received` với `attributes.attempt` |
| Not found (tool trả `is_error`, ví dụ `get_refund_timeline` khi order không có refund) | Không (là câu trả lời, không phải sự cố) | Coi như không có record; nếu thiếu `get_order` / payment / shipment → `insufficient_evidence`, `needs_investigation`, refund 0, confidence ≤ 0.35 | `tool_result_consumed` `TOOL_NO_RECORDS`; `ORDER_NOT_FOUND` |
| Tool không có trong discovery | Không | Như not found | `TOOL_NOT_DISCOVERED` |
| Source conflict (shipment event trái timestamp order; freight item ≠ số tiền đã capture; status lệch giữa tool) | Không | Timestamp/record của order là nguồn chuẩn; refund không vượt số tiền đã capture; ghi vào `data_conflicts`; confidence −0.03 mỗi conflict | `data_conflicts[].resolution_code` (`ORDER_TIMESTAMPS_AUTHORITATIVE`, `REFUND_CAPPED_AT_CAPTURED_AMOUNT`, …) |
| Invalid specialist result / verifier fail | Không | Giữ kết luận nhưng hạ confidence ≤ 0.4; không bịa dữ liệu thay thế | `verification_completed` `VERIFICATION_FAILED` + `first_problem` |
| Vượt hop budget / message sai case | Không | Case lỗi, run dừng | `A2AError` |

## 6. Verification invariants

Verifier (`VerifierAgent.check`) kiểm tra trước khi finalize; CLI kiểm tra schema lần nữa trước khi ghi file:

- **Schema**: output hợp lệ theo `l3a-output-v2.schema.json`.
- **Case scope**: `case_id` đúng; `affected_entities.order_ids` ⊆ {`claimed_order_id`} đã xác minh.
- **Evidence ownership**: mọi ref (output + claim) có trong registry của case này; mọi ref đã có `tool_result_consumed` trong trace; ref của claim ⊆ ref của output.
- **Required evidence**: domain bắt buộc theo issue (ví dụ `refund_failed` cần `refund` + `policy`) đều có mặt.
- **Money totals**: tổng `refund_lines` = `recommended_refund_brl` (tới cent).
- **Status/refund/action consistency**: có action và không trùng; `no_action` ⇒ refund 0, không có refund line; refund > 0 ⇒ `action_required`.
- **Responsibility**: `late_delivery_seller` phải có party seller; `late_delivery_logistics` không đổ lỗi seller; seller được quy trách nhiệm phải nằm trong `affected_entities.seller_ids`.
- **Confidence bounds**: trong [0, 1]; `insufficient_evidence` không được > 0.6.

Confidence: 0.93 khi claim được evidence xác nhận; −0.10 nếu evidence trái lời khai; −0.08 mỗi issue cạnh tranh; −0.03 mỗi data conflict; trần 0.97; `insufficient_evidence` 0.35.

## 7. Reproducibility

- Không dùng model/LLM, không có random seed: logic tất định. Event id / timestamp trong trace là giá trị duy nhất theo lần chạy.
- Python ≥ 3.11 (đã chạy trên 3.13, conda env `lab-vin-env`). Dependency theo `pyproject.toml`: `mcp>=2,<3`, `httpx2>=2,<3`, `jsonschema[format]>=4.25,<5`, `python-dotenv>=1.1,<2` (đã kiểm với mcp 2.2.0, httpx2 2.12.0).
- Concurrency: mặc định **1 case một lúc** (`DAY09_CONCURRENCY` để đổi), mỗi case một MCP session, và **mỗi session chỉ có một tool call đang chạy**. Bản đầu chạy 3 case song song + payment/shipment song song trong case; bản nộp đó chỉ đạt ~94% ở provenance, nên chuyển sang tuần tự để audit trail phía server không bị nhập nhằng.
- Ngân sách: 6 tool call mỗi case (`get_order`, `get_order_items`, `get_payment_timeline`, `get_refund_timeline`, `get_shipment_summary`, `get_policy`) + `get_sellers` khi seller chịu trách nhiệm (bản v5 bỏ call này và mất điểm evidence). Timeout HTTP 300s, lượt agent 240s, tối đa 3 lần thử mỗi case.
- Lệnh chạy:

```bash
python -m pip install -e ".[dev]"
cp .env.example .env   # điền COMPETITION_API_URL, COMPETITION_TEAM_API_KEY, MCP_ENDPOINT
day09 validate-inputs
day09 run && day09 validate
day09 package --output dist/submission.zip
ruff check . && pytest -q
```

- Không ghi API key vào output, trace hay tài liệu; `validate` chặn chuỗi `sk-team-...` trong submission.
