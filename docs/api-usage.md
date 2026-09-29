# Add / Search 接口使用说明（对接自测用）

> **给谁看**：要手工或写脚本调这两个端点的人。目标：3 分钟内跑通一次写入 + 取回。
> **权威定义**：契约本身以 [`contract.md`](./contract.md) §1（= PRD §2.1）为准；
> 本文只讲"怎么调、怎么判对错"，冲突时以契约文档为准。

---

## 0. 服务地址与鉴权

| 项 | 值 |
| --- | --- |
| Base URL | **`https://tianximem.mbgtest.lenovomm.com`**（对外入口，HTTPS） |
| 直连地址 | `http://10.193.135.28:28088`（内网直连同一个服务；两个入口行为一致，任选其一） |
| 端点 | `POST /add`、`POST /search`、`GET /health`（探活，不碰数据） |
| 认证 | **不需要**。没有 API key、不用带 `Authorization` 头（两个入口都不校验） |
| 请求体 | JSON（记得带 `Content-Type: application/json`） |

> ⚠ 服务本身**没有任何鉴权**：`user_id` 只是**数据隔离**字段，不是登录凭据。
> 别往里写真实隐私数据。

---

## 1. 先探活

```bash
curl -s https://tianximem.mbgtest.lenovomm.com/health
# {"status":"ok"}
```

只要 HTTP 2xx 就算活着。**注意它不会去探下游**（Qdrant、模型网关），所以
`/health` 绿了不代表此刻能写入——真正的判据是下面第 2 步的 Add 是否 200。

---

## 2. `POST /add` —— 写入一批消息

### 请求字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | string | 本批的唯一 ID。**任意字符串都行，服务端不解析它的格式**（见下面 ⚠ 第 1 条） |
| `user_id` | string | 数据隔离键。**只有同一个 `user_id` 写进去的东西，才能被同一个 `user_id` 搜到** |
| `session_id` | string | 会话分组（同一次对话的几批用同一个值）。**不是检索过滤条件** |
| `messages` | array | 按**发生顺序**排列，至少 1 条 |
| `messages[].role` | string | 例如 `user` / `assistant` / `system` |
| `messages[].content` | string | 正文（可为空串，不会因此报错） |
| `messages[].timestamp` | int，可选 | **Unix 毫秒**。给了它，返回的 `content` 里会带一个日粒度日期锚点 |

### 示例

```bash
curl -s -X POST https://tianximem.mbgtest.lenovomm.com/add \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "test-alice-20260928:chunk-0",
    "user_id": "test-alice",
    "session_id": "s1",
    "messages": [
      {"role": "user", "content": "I moved to Hangzhou last Wednesday.", "timestamp": 1697900000000},
      {"role": "assistant", "content": "Got it — Hangzhou, and I will remember that."}
    ]
  }'
```

### 成功响应（200）

```json
{
  "success": true,
  "request_id": "test-alice-20260928:chunk-0",
  "user_id": "test-alice",
  "session_id": "s1"
}
```

`request_id` / `user_id` / `session_id` **原样回显**。**判成功看 `success == true` 且三者与请求逐字一致**，
不要只看 HTTP 200。

### ⚠ 三个必须知道的点

1. **`request_id` 是不透明字符串，任意形状都收。**
   服务端**不解析它**——不要求 `chunk-<数字>`、不要求序号落在结尾（`"r_3115a1…"`、`"abc"`、
   `"foo:bar"` 都一样收）。它**整个**参与记忆位置的计算，另一半是这条消息在**本批**里的下标。
   ⇒ 用什么形状由调用方定，**换形状不会让这一批写不进去**。
2. **同一个 `request_id` + 同一份 payload 重复发是幂等的**：照样返回 200，但不会重复写入
   ⇒ **每次测试用新的 `request_id`**（最简单：带上时间戳或计数）。
3. **响应返回时就已经持久化、立刻可搜**——Add 成功之后马上 Search 就能查到，不需要等待。

---

## 3. `POST /search` —— 按名次取回证据

### 请求字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `user_id` | string | 同一个隔离键。搜**别人**的 `user_id` 只会得到 `[]` |
| `query` | string | 查询语句，**非空白**即可；服务端不改写它 |
| `top_k` | int ≥ 1 | 最多返回几条（线上固定传 100；自测建议 10） |

### 示例

```bash
curl -s -X POST https://tianximem.mbgtest.lenovomm.com/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test-alice", "query": "Where did Alice move to?", "top_k": 10}'
```

### 成功响应（200）

```json
{
  "data": [
    {
      "id": "3f2a9c41e7b0d85a1c6f4e2b9d07a3c5f8e1b46d20c793a4f5e8b1d6c3a9072e",
      "content": "[2023-10-22] Q: I moved to Hangzhou last Wednesday (October 18, 2023).\nA: [assistant] Got it — Hangzhou, and I will remember that.",
      "created_at": "2023-10-22",
      "score": 1.0
    }
  ]
}
```

| 字段 | 说明 |
| --- | --- |
| `id` | 稳定字符串 ID（64 位十六进制），同一位置永远是同一个 ID，可用于去重 |
| `content` | **渲染后的记忆原文**（`Q:` / `A:` 行首标记；**同一次 `add` 内**相邻的若干条会被合并成一段返回） |
| `created_at` | **总是存在**。日粒度日期（`2023-10-22`）；消息没带 `timestamp` 时是空串 `""`。**不会是秒级时间戳** |
| `score` | 随名次单调递减的**占位值**（第 1 名 = `1.0`），**不是**相似度、不要拿它做阈值判断 |

### ⚠ 必须知道的点

- 结果**按名次降序**，`len(data) <= top_k` 一定成立；没有命中就是**空数组 `[]`**（不是 `null`）。
- **它不会回答你的问题**——`content` 是检索到的原始证据，答案要由调用方（AML）自己生成。
  所以别指望 `query` 用自然语言提问就返回一句答案。
- 返回的 `content` 可能与写入的 `messages` 不完全逐字相同：会加日期锚点、把**同一次 `add` 内**的相邻条目合并成一段，
  这是**有意为之**（让下游模型能正确读出时间）。

---

## 4. 端到端最小例子（复制即可跑）

```bash
BASE=https://tianximem.mbgtest.lenovomm.com
UID=test-$(date +%s)          # 每次跑用新的 user_id，避免和历史数据混在一起

# ① 写入（request_id 是**任意字符串**，这里带个后缀只是为了好认）
curl -s -X POST $BASE/add -H 'Content-Type: application/json' -d "{
  \"request_id\": \"$UID:chunk-0\",
  \"user_id\": \"$UID\",
  \"session_id\": \"s1\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"My favourite band is Queen.\", \"timestamp\": 1697900000000},
    {\"role\": \"assistant\", \"content\": \"Noted.\"}
  ]
}"
echo

# ② 取回
curl -s -X POST $BASE/search -H 'Content-Type: application/json' -d "{
  \"user_id\": \"$UID\", \"query\": \"What band does the user like?\", \"top_k\": 5
}"
echo

# ③ 隔离自检：换个 user_id 搜，应当得到 {\"data\": []}
curl -s -X POST $BASE/search -H 'Content-Type: application/json' -d '{
  "user_id": "someone-else", "query": "band", "top_k": 5
}'
echo
```

Python 版（`pip install httpx`）：

```python
import httpx

BASE = "https://tianximem.mbgtest.lenovomm.com"
UID = "test-python-1"

with httpx.Client(base_url=BASE, timeout=60) as c:
    assert c.get("/health").json() == {"status": "ok"}

    r = c.post("/add", json={
        "request_id": f"{UID}:chunk-0",
        "user_id": UID,
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "My favourite band is Queen.",
             "timestamp": 1697900000000},
            {"role": "assistant", "content": "Noted."},
        ],
    })
    r.raise_for_status()
    body = r.json()
    assert body["success"] is True and body["request_id"] == f"{UID}:chunk-0"

    r = c.post("/search", json={"user_id": UID, "query": "favourite band?", "top_k": 5})
    r.raise_for_status()
    data = r.json()["data"]
    assert isinstance(data, list) and len(data) <= 5
    for item in data:
        print(item["score"], item["created_at"], repr(item["content"][:80]))
```

---

## 5. 出错了怎么判

| 状态码 | 含义 | 怎么办 |
| --- | --- | --- |
| `200` | 成功 | Add 再看 `success == true` 与三个回显字段；Search 再看 `data` 是数组 |
| `422` | 请求体不合法：缺字段、`messages` 为空、`top_k < 1`、`query` 全空白 | 按响应里的 `detail` 改请求，**重发同一份请求没有意义** |
| `409` | **同一个 `request_id` 收到了不同的 payload**——服务端拒绝这一批，两份都不会写坏 | **重发没有意义**（结果永远一样）；换一个新的 `request_id` 再发 |
| `500` | 服务内部错误 | 看响应 `detail`；**可以直接重试**（这一批没被确认） |
| `503` | 依赖（向量库 / 模型网关）暂时不可用 | **可以直接重试**，接口是幂等的 |
| 连不上 / 超时 | 网络或服务没起 | 先打 `/health`；再确认域名可解析、或直连 `10.193.135.28:28088` |

**重试是安全的**：同一 `request_id` + **同一份** payload 重发不会写重。Add 的失败**不会**留下"半条"记忆
（要么整批生效，要么下次重试补上）。

⚠ 反过来：同一个 `request_id` 配**不同的** payload 会被判成冲突并回 **409**。
服务端刻意不"挑一份落库"——那会让另一份记忆**凭空消失**，而检索侧完全看不出来。

---

## 6. 自测注意事项

- **数据是持久的**：写进去就留在服务的记忆库里，没有"清空"接口。
  自测请统一用带前缀的 `user_id`（如 `test-<你的名字>-<日期>`），别用平台正式跑数用的 ID。
- **跨 `user_id` 检索被禁止**：查不到别人的数据是**正确行为**，不是 bug。
- **`session_id` 不是过滤条件**：Search 只按 `user_id` 隔离。它是分组字段（参与记忆 ID 的计算），
  但**既不参与检索过滤，也不决定内容怎么合并**——合并只按"一次 `add`"划界，跨 `add` 永不合并。
- **一批别塞太多**：线上口径是每批 ≤20 条消息或 2,000 词（切分由平台做）。
  自测保持在这个量级，否则形状与线上不一致。
- **时钟**：`timestamp` 用 **Unix 毫秒**；不传也能写，只是返回的 `created_at` 会是 `""`。
- 想一次验完整契约（14 条自动检查），在仓库里跑 `make contract-check`
  （打的是本地服务；要打远程加 `--base-url`）。
