"""`eval/smoke/preflight.py` 的静态一致性（**不联网、不起服务**）。

真正"打真 HTTP"的那部分由 `make contract-check` 跑（`contract.md` §4 的清单），
这里只钉住三件**可以离线断言**的事：

1. `eval/` 的边界——**不 import `src/`**（否则契约层就退化成进程内调用了）
2. 与 `src/` 的**有意重复**必须是一致的（环境变量名、临时集合名）
3. 退出码语义与 CLI 可用

> 单元测试验证**函数**，`make contract-check` 验证**真的 HTTP 响应**——两者都要
> （[`./CLAUDE.md`](./CLAUDE.md) 结尾那条）。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tianximem.common import config as cfg

_PREFLIGHT_PATH = Path(__file__).resolve().parents[1] / "eval" / "smoke" / "preflight.py"
_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def preflight():
    """导入 `eval.smoke.preflight`（模块级代码只定义东西，不会跑服务）。"""
    sys.path.insert(0, str(_REPO))
    try:
        from eval.smoke import preflight as module
    finally:
        sys.path.pop(0)
    return module


# ── 1. eval/ 的边界：不 import src/ ────────────────────────────────────


def test_preflight_does_not_import_src() -> None:
    """**`eval/` 的边界是结构性的**：本文件不得 `import tianximem`。

    理由两条（[`../../eval/CLAUDE.md`](../../eval/CLAUDE.md)）：

    * §13 要求 B1（ReFind 原版）在我们自己的 harness 里重跑，而它只以"另一个服务"的形式
      存在 ⇒ 若 harness 走进程内调用，B1 就成了特例，两条基线不可比；
    * **契约层只有走 HTTP 才碰得到**——进程内调用根本看不到 `/add` 的 422
      与 `data` 的序列化。

    ⚠ 用 AST 而不是文本匹配：注释与 docstring 里出现 `tianximem` 是**正常的**
    （本文件的说明就在提它），把它们算成违规会让规则变成噪音。
    """
    tree = ast.parse(_PREFLIGHT_PATH.read_text(encoding="utf-8"), filename=str(_PREFLIGHT_PATH))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("tianximem"):
                offenders.append(f"L{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tianximem"):
                    offenders.append(f"L{node.lineno}: import {alias.name}")

    assert offenders == [], (
        "eval/smoke/preflight.py 不得 import src/（eval/CLAUDE.md 的边界）：\n  "
        + "\n  ".join(offenders)
    )


def test_preflight_module_is_pure_at_import(preflight) -> None:
    """导入它**不产生副作用**（不起服务、不读环境、不连网）。"""
    assert callable(preflight.main)
    assert preflight.EXIT_OK == 0


# ── 2. 与 src/ 的有意重复必须一致 ──────────────────────────────────────


@pytest.mark.parametrize(
    ("preflight_attr", "config_attr"),
    [
        ("_ENV_QDRANT_URL", "ENV_QDRANT_URL"),
        ("_ENV_EMBED_BASE_URL", "ENV_EMBED_BASE_URL"),
        ("_ENV_EMBED_API_KEY", "ENV_EMBED_API_KEY"),
        ("_ENV_CONFIG_DIR", "ENV_CONFIG_DIR"),
        ("_ENV_PROFILE", "ENV_PROFILE"),
        ("_ENV_SQLITE_PATH", "ENV_SQLITE_PATH"),
        ("_ENV_EMBED_CACHE_DIR", "ENV_EMBED_CACHE_DIR"),
        ("_ENV_WORKERS", "ENV_WORKERS"),
    ],
)
def test_env_names_match_the_single_declaration(
    preflight, preflight_attr: str, config_attr: str
) -> None:
    """环境变量名的**单一声明**在 `common/config.py`；preflight 里那份必须与它逐字相同。

    `eval/` 不能 import `src/`（上一条），所以这里**必须**有一处重复。
    重复本身不危险——**漂移才危险**，而这条用例就是那道保险
    （与 `RRF_K` 两处相等是同一种做法）。
    """
    assert getattr(preflight, preflight_attr) == getattr(cfg, config_attr)


def test_preflight_uses_a_collection_nobody_else_uses(preflight) -> None:
    """预检的集合名**必须与开发期、提交期的都不同**。

    否则预检写进去的记忆会**永久留在 `memories_dev` 里**，污染之后每一次检索——
    而那正是 `open-questions.md` **V9** 的形状（遗留 point 按 id upsert，静默不报错）。
    """
    used = {preflight._PREFLIGHT_COLLECTION}
    assert preflight._PREFLIGHT_COLLECTION not in {"memories", "memories_dev"}
    assert len(used) == 1  # 只有一个名字，不是"某几个中的一个"


def test_preflight_collection_is_not_the_yaml_default(preflight) -> None:
    """再钉一层：预检集合名不等于 `configs/default.yaml` 里的那个。"""
    text = (_REPO / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert f"collection: {preflight._PREFLIGHT_COLLECTION}" not in text


# ── 3. 退出码语义 ──────────────────────────────────────────────────────


def test_exit_codes_are_distinct_and_nonzero_on_failure(preflight) -> None:
    """**"没跑成"与"跑了但没过"必须可区分**，且两者都非零。

    `make contract-check` 是一条门禁：任何非零都必须让 make 失败——
    门禁静默放行比门禁不存在更危险。
    """
    codes = {preflight.EXIT_OK, preflight.EXIT_CHECKS_FAILED, preflight.EXIT_PRECONDITION_FAILED}
    assert len(codes) == 3, "三个退出码必须互不相同"
    assert preflight.EXIT_OK == 0
    assert preflight.EXIT_CHECKS_FAILED != 0
    assert preflight.EXIT_PRECONDITION_FAILED != 0


def test_cli_parses(preflight) -> None:
    """`--help` 可用（CLI 形状本身是被 make 依赖的那一层）。"""
    result = subprocess.run(
        [sys.executable, str(_PREFLIGHT_PATH), "--help"],
        capture_output=True,
        # ⚠ 必须显式给 UTF-8：`preflight` 会把自己的 stdio 切到 UTF-8（见
        # `_force_utf8_stdio`），而 `text=True` 单独用会按**本机 locale**（这里 GBK）解码
        # ⇒ 读线程抛 UnicodeDecodeError 死掉，`result.stdout` 直接变成 None。
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "contract.md" in result.stdout


def test_preconditions_fail_loudly_when_the_service_cannot_start(tmp_path: Path) -> None:
    """**服务起不来时必须 exit 2，而不是"跳过检查后报通过"。**

    这是本文件最要紧的一条：预检的全部价值在于"**它是门禁**"。一个在缺前置时静默放行的
    门禁，会让 Smoke 的额度直接暴露在契约违规之下。

    **怎么做到不依赖机器状态**：把 `TIANXIMEM_ENV_FILE` 指到一个不存在的路径
    ⇒ 连 `.env` 也不会被读到 ⇒ 密钥必然为空 ⇒ 服务启动即失败。
    （配置校验由服务自己负责，见 `preflight._launch_service` 的说明。）
    """
    result = subprocess.run(
        [sys.executable, str(_PREFLIGHT_PATH)],
        capture_output=True,
        # ⚠ 同上：必须显式给 UTF-8，否则读线程按**本机 locale**（这里 GBK）解码而死。
        encoding="utf-8",
        errors="replace",
        timeout=180,
        cwd=str(_REPO),
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith(("AML_", "TIANXIMEM_"))},
            "TIANXIMEM_ENV_FILE": str(tmp_path / "no-such.env"),
        },
    )
    assert result.returncode == 2, (
        f"服务起不来时应当 exit 2，实际 exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "前置条件不满足" in result.stdout
    # 服务自己那句报错要被原样转述出来（否则排错得自己再去翻日志）
    assert "embed.base_url" in result.stdout or "AML_EMB_BASE_URL" in result.stdout


# ── 4. 检查**真的会失败吗**（否则这个门禁是摆设）─────────────────────────
#
# 一个永远不会 FAIL 的检查等于没有检查。下面每一条都塞一个**故意违规**的假服务，
# 确认对应的检查确实报 FAIL；最后用 `mode="ok"` 做**阳性对照**——
# 证明这个假服务不是"无论如何都失败"，也就是证明这些用例不是空过的。


class FakeService:
    """最小的假 `/add` + `/search`，可以按 `mode` **故意制造契约违规**。"""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.added: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:  # noqa: ANN001
        if request.url.path == "/add":
            payload = json.loads(request.content)
            # ⚠ 真的服务按 `request_id` **幂等**（§2.2）；假服务必须照样做，
            #    否则"重放不重复写入"那条检查**没有阳性对照**可言。
            #    `merged_duplicate` 是那条检查的违规模式：重放被当成新的一批。
            if self.mode == "merged_duplicate" or not self._already(payload["request_id"]):
                self.added.append(payload)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": payload["request_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                },
            )
        if request.url.path == "/search":
            body = json.loads(request.content)
            return httpx.Response(200, json={"data": self._search(body)})
        return httpx.Response(404)

    def _already(self, request_id: str) -> bool:
        return any(m["request_id"] == request_id for m in self.added)

    def _search(self, body: dict) -> object:
        if self.mode == "null_data":
            return None
        user = body["user_id"]
        top_k = body["top_k"]
        mine = [m for m in self.added if m["user_id"] == user]

        if self.mode == "merged_duplicate":
            # 违规点：**重复写入**（见 `/add`）。而检索侧照**段模型**合并
            # ⇒ 多出来的那一行**不表现为多一项**，而是**同一段变宽**（同一份原文出现两次）。
            # 这正是"只比 id 与条数会漏掉它"的那个场景。
            merged = "\n".join(
                f"Q: {m['messages'][0]['content']}\nA: [assistant] {m['messages'][1]['content']}"
                for m in mine
            )
            return [{"id": f"{user}:0", "content": merged, "created_at": "", "score": 1.0}]

        # 违规点：无视 top_k，把该 user 的全部返回
        count = len(mine) if self.mode == "ignore_top_k" else min(top_k, len(mine))

        items = []
        for i, message in enumerate(mine[:count]):
            question = message["messages"][0]["content"]
            answer = message["messages"][1]["content"] if len(message["messages"]) > 1 else ""
            if self.mode == "generated":
                content = "The train leaves at 09:42."  # 像答案、不像记忆
            else:
                content = f"Q: {question}\nA: [assistant] {answer}"
            if self.mode == "fused_score":
                # 融合分数的量级（RRF k=61 ≈ 0.033），而不是名次倒数
                score: float = 0.032786883 - i * 0.001
            elif self.mode == "fused_tail":
                # rank=0 是 1.0（合法）、也严格递减，但尾巴掉进融合分数的量级
                score = 1.0 if i == 0 else 0.032 - i * 0.001
            else:
                score = 1.0 / (i + 1)
            item: dict[str, object] = {
                "id": f"{user}:{i}",
                "content": content,
                "created_at": "",
                "score": score,
            }
            if self.mode == "extra_field":
                item["text"] = content
            if self.mode == "missing_created_at":
                del item["created_at"]
            items.append(item)
        return items


def _client_for(mode: str):
    import httpx

    return httpx.Client(
        base_url="http://fake",
        transport=httpx.MockTransport(FakeService(mode)),
        timeout=5.0,
    )


def _run_one(mode: str, method: str) -> None:
    """只跑**指定的那一条**检查（跑全部的话，一处违规会连带弄脏别的检查）。"""
    from eval.smoke.preflight import Preflight

    preflight_module = __import__("eval.smoke.preflight", fromlist=["Preflight"])
    service = FakeService(mode)
    import httpx

    client = httpx.Client(
        base_url="http://fake", transport=httpx.MockTransport(service), timeout=5.0
    )
    pf = Preflight("http://fake", nonce="t", client=client)
    try:
        getattr(pf, method)()
    finally:
        pf.close()
    assert preflight_module is not None


@pytest.mark.parametrize(
    ("mode", "method", "expected"),
    [
        ("null_data", "check_search_returns_a_list_of_exactly_four_fields", "必须是数组"),
        ("extra_field", "check_search_returns_a_list_of_exactly_four_fields", "字段集合必须恰好"),
        (
            "missing_created_at",
            "check_search_returns_a_list_of_exactly_four_fields",
            "字段集合必须恰好",
        ),
        ("ignore_top_k", "check_count_is_within_top_k", "契约错误"),
        ("fused_score", "check_score_is_rank_derived", "rank=0 的 score 必须是 1.0"),
        ("fused_tail", "check_score_is_rank_derived", "量级像融合分数"),
        ("generated", "check_content_properties", "问题原文"),
        # ⚠ 段模型下"重复写入"**不再表现为多一项**，而是**同一段变宽**——
        #    所以这条违规模式刻意让检索侧照段模型合并（见 `FakeService._search`）。
        ("merged_duplicate", "check_replay_does_not_write_again", "正文变了"),
    ],
)
def test_check_fails_on_a_violating_service(mode: str, method: str, expected: str) -> None:
    """**每一条检查在对应的违规下都必须 FAIL。** 见本节的标题。"""
    from eval.smoke.preflight import CheckFailed

    with pytest.raises(CheckFailed, match=expected):
        _run_one(mode, method)


@pytest.mark.parametrize(
    "method",
    [
        "check_search_returns_a_list_of_exactly_four_fields",
        "check_count_is_within_top_k",
        "check_score_is_rank_derived",
        "check_content_properties",
        "check_add_echoes_the_three_fields",
        "check_replay_does_not_write_again",
    ],
)
def test_checks_pass_on_a_conforming_service(method: str) -> None:
    """**阳性对照**：同一个假服务在 `mode="ok"` 下这些检查必须**全部通过**。

    没有这一条，上面那组用例是空过的——一个"无论如何都报 FAIL"的检查也能让它们通过。
    """
    _run_one("ok", method)


def test_run_aggregates_failures_and_keeps_going() -> None:
    """`run()` 必须**把全部检查跑完**再汇报，而不是第一条失败就退出。

    一次跑完看到所有问题，比修一个跑一次快得多——而预检是每次改检索逻辑都要过的门。
    """
    import httpx
    from eval.smoke.preflight import CheckFailed, Preflight

    service = FakeService("null_data")
    client = httpx.Client(
        base_url="http://fake", transport=httpx.MockTransport(service), timeout=5.0
    )
    pf = Preflight("http://fake", nonce="t", client=client)
    try:
        report = pf.run()
    finally:
        pf.close()

    assert report.results, "什么检查都没跑"
    assert report.failed, "在明显违规的假服务上竟然全部通过"
    # 全跑完了：结果条数 == 注册的检查条数（不是第一条就短路）
    assert len(report.results) >= 10
    assert CheckFailed is not None
