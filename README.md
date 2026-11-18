# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| POST | `/probes` | 用随附样图对固定模型做批量 1/2/4 的计时探测，返回各批计算耗时与吞吐量。探测进行中再次调用返回 **409** |
| GET | `/stats` | 容量/状态计数（演示与观测用，含 `probe_active`） |
| GET | `/health` | 存活探针 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel

curl -X POST http://localhost:8000/probes
# {
#   "probe_id": "...", "sample_count": 4, "batch_sizes": [1, 2, 4],
#   "trials": [
#     {"batch_size": 1, "images": 1, "compute_ms": 6.0, "forward_ms": 6.07,
#      "throughput_images_per_s": 166.7},
#     {"batch_size": 2, ...}, {"batch_size": 4, ...}
#   ]
# }
```

`/probes` 响应字段：

- `compute_ms`：该批次占用唯一计算通道的**墙钟耗时**（设置演示用 `INFERENCE_DELAY_MS` 时含注入延迟）。
- `forward_ms`：推理线程内测得的**纯模型前向耗时**（不含注入延迟）。
- `throughput_images_per_s`：按 `batch_size / compute_ms` 计算的吞吐量，供选择微批大小。

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。

## 探测（/probes）规则实现（与需求逐条对应）

- **样图与预处理**：读取 `samples/`（默认 4 张，按文件名排序），逐张走与 `/predict`
  完全相同的 `preprocess()`，组成 1、2、4 张的批次试跑；首次正式计时代前有一次
  **不计入结果的预热**前向，避免一次性内存/JIT 开销污染 1 张批次的耗时。
- **唯一计算通道**：探测批次与正式订单由同一个 worker 协程派发到同一个单线程执行器，
  不存在第二条推理路径；任何时刻只有一批在做前向计算。
- **订单优先**：每轮探测试跑（含预热）开始前，worker 先把**所有已受理的待推理订单**
  排空；探测期间新提交的订单也会在下一轮试跑前先完成。
- **不可抢占**：已在运行的批次（无论订单还是探测）在线程执行器内不可中断，探测只能
  等当前批次实际结束后再开始，订单的截止/结果/容量规则不受影响。
- **不生成订单、不占容量**：探测张量保存在独立的 `ProbeState` 中，不进入订单队列、
  不计入 16 张容量，`/stats` 的 `batches_started` 也不统计探测批次。
- **期间 API 可用**：阻塞计算仍在独立线程，探测期间提交/查询/取消均在毫秒级响应，
  16 张容量可被正式订单正常用满（满额仍返回 429）。
- **并发探测 409、失败可重试**：全局同时只允许一个探测，第二个请求返回 **409**；
  模型失败返回 **500** 且探测槽立即释放，可直接重试；客户端断连时 worker 会在当前
  不可抢占批次结束后放弃探测并释放槽位。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟（探测的 `compute_ms` 含此延迟，`forward_ms` 不含） |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |
| `SAMPLES_DIR` | `samples` | `/probes` 使用的随附样图目录（容器内 `/srv/samples`） |

慢推理演示：`docker compose up` 前把 `docker-compose.yml` 里的 `INFERENCE_DELAY_MS`
改为 `2000`，推理约 2 秒/批，期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察）。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟；以及 `/probes` 的计时载荷、
无订单/容量副作用、样图预处理一致性、并发探测 409 与失败/完成后重试、
订单优先排空、运行批次不可抢占、探测期间 API 响应与 16 张容量仍可用。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
