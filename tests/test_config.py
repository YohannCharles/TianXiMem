"""`common/config.py` —— 配置的唯一入口（§15 / §12.1 R1 对冲 3）。

## 这个文件里最重要的一条

`test_only_config_reads_the_environment` —— 它**静态地**扫描 `src/tianxi_am/**/*.py`，
断言除 `common/config.py` 之外**没有任何模块访问 `os.environ` / `os.getenv`**。
"唯一入口"是一个可以被写成断言的**结构性质**，而不是一条靠自觉的约定。

## 本文件全部用例都**不依赖真实环境变量**

每个用例都显式构造 `env` dict 传进 `load_config()`；涉及 yaml 的用 `tmp_path` 现造。
⇒ 在谁的机器上、`.env` 里有什么，都不影响这里的结果。
"""

from __future__ import annotations

import ast
import multiprocessing
import re
from pathlib import Path

import pytest

from tianxi_am.common import config as config_module
from tianxi_am.common.config import (
    ENV_CONFIG_DIR,
    ENV_EMBED_API_KEY,
    ENV_EMBED_BASE_URL,
    ENV_FILE,
    ENV_PROFILE,
    ENV_SQLITE_PATH,
    ENV_WORKERS,
    RRF_K,
    AppConfig,
    ConfigError,
    assert_single_process,
    load_config,
)

#: 一份"最小可启动"的环境（只给**必需**的那两项，其余走内置默认值）。
_MIN_ENV: dict[str, str] = {
    ENV_EMBED_BASE_URL: "http://unused/v1",
    ENV_EMBED_API_KEY: "k",
}

_SRC = Path(__file__).resolve().parents[1] / "src" / "tianxi_am"
_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """一个自带的 `configs/`：先照抄仓库里的真文件。

    这样用例既**不依赖 cwd**（不会因为从别的目录跑 pytest 而找不到 `configs/`），
    又测的是**仓库里那份真的 yaml**——而不是一个只在这个测试里存在的假配置。
    """
    real = _REPO / "configs"
    for name in ("default.yaml", "local.yaml"):
        (tmp_path / name).write_text((real / name).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def _load(config_dir: Path, **env: str) -> AppConfig:
    return load_config({**_MIN_ENV, **env}, config_dir=config_dir)


# ── 1. 唯一入口：**结构性质，不是约定** ────────────────────────────────

_CONFIG_MODULE = _SRC / "common" / "config.py"


def _env_reads(tree: ast.Module) -> list[tuple[int, str]]:
    """找出真正**访问**环境变量的语句（`os.environ` / `os.getenv` / `from os import environ`）。

    ⚠ 用 AST 而不是文本匹配：docstring 与注释里出现 `os.environ` 字样是**正常的**
    （本文档就在讲它），把它们算成违规会让这条用例变成噪音而被删掉。
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in {"environ", "getenv"}:
                hits.append((node.lineno, f"os.{node.attr}"))
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in {"environ", "getenv"}:
                    hits.append((node.lineno, f"from os import {alias.name}"))
    return hits


def test_only_config_reads_the_environment() -> None:
    """**除 `common/config.py` 外，全包不得访问环境变量。**

    这不是洁癖：读取点一多，"这次跑的是哪套值"就无法从一个地方回答，
    而 §13 的每一条对照实验都依赖那个回答。
    """
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path == _CONFIG_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, what in _env_reads(tree):
            offenders.append(f"{path.relative_to(_SRC)}:{lineno} 用了 {what}")

    assert offenders == [], (
        "只有 common/config.py 能读环境变量（见 common/CLAUDE.md 与本文件顶部）：\n  "
        + "\n  ".join(offenders)
    )


def test_config_module_itself_reads_only_through_load_config() -> None:
    """反向确认：`config.py` **确实**读了环境变量（否则上面那条是空过的）。"""
    tree = ast.parse(_CONFIG_MODULE.read_text(encoding="utf-8"))
    assert _env_reads(tree), "config.py 竟然没有访问 os.environ ——那它不是唯一入口而是一个空壳"


def test_no_module_reads_config_files_directly() -> None:
    """除 `common/config.py` 外**没有**模块自己去读 `configs/*.yaml`。

    否则"配置从哪来"就有了第二条路径，与"唯一入口"是同一个问题。
    """
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path == _CONFIG_MODULE:
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"yaml\.safe_load|yaml\.load\b", source):
            offenders.append(str(path.relative_to(_SRC)))

    assert offenders == [], f"这些模块自己解析 yaml 了：{offenders}"


def test_every_env_name_is_declared_in_the_env_example() -> None:
    """**每一个 `ENV_*` 常量都必须在 `.env.example` 里出现。**

    ⚠ 这条挡的是**改名漂移**：代码里改了变量名、`.env.example` 没跟着改，
    于是照着模板填 `.env` 的人**填的是另一个变量**——服务照常启动，
    只是那个值从来没被读到（表现为"配了却没生效"，而**不报错**）。

    反向不检查（`.env.example` 里的名字都在代码里用）：水印式的注释与
    暂时没有消费者的变量都允许先待在模板里（例如归档 pipeline 用的 `AML_*` 组）。
    """
    example = (_REPO / ".env.example").read_text(encoding="utf-8")
    declared = {
        name: value
        for name, value in vars(config_module).items()
        if name.startswith("ENV_") and isinstance(value, str)
    }
    assert declared, "一个 ENV_* 常量都没找到 —— 这条用例是空过的"

    missing = [
        f"{name} = {value}" for name, value in sorted(declared.items()) if value not in example
    ]
    assert missing == [], (
        "这些环境变量名没写进 `.env.example`（照着模板填的人会漏掉它们）：\n  "
        + "\n  ".join(missing)
    )


# ── 配置键的**唯一声明处**：`docs/config-reference.md` ────────────────

_CONFIG_REFERENCE = _REPO / "docs" / "config-reference.md"

#: 文档里被反引号包着、又**不是配置键**的点分串（文件名那类）。
_NOT_A_CONFIG_KEY_SUFFIX = (".yaml", ".yml", ".md", ".py", ".json", ".sql", ".toml")


def _code_yaml_keys() -> set[str]:
    """AST 扫 `config.py` 的 `_group(..., where=..., allowed={...})`，还原它接受的**点分键**。

    `where` 就是前缀——**这不是猜的，是那条调用约定本身**（`where="storage.sqlite"`
    配 `allowed={"busy_timeout_ms"}` ⇒ `storage.sqlite.busy_timeout_ms`）。

    只保留**叶子**：`storage.sqlite` 这类中间段不是键，是 `storage.sqlite.*` 的前缀。
    """
    tree = ast.parse(_CONFIG_MODULE.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_group"):
            continue
        where: str | None = None
        allowed: set[str] | None = None
        for kw in node.keywords:
            if kw.arg == "where" and isinstance(kw.value, ast.Constant):
                where = kw.value.value
            elif kw.arg == "allowed" and isinstance(kw.value, ast.Set):
                allowed = {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
        if where and allowed:
            keys |= {f"{where}.{name}" for name in allowed}
    return {k for k in keys if not any(other != k and other.startswith(k + ".") for other in keys)}


def _documented_dotted_keys() -> set[str]:
    """文档里所有反引号包着的点分键（滤掉 `.yaml` / `.md` 这类文件名）。"""
    text = _CONFIG_REFERENCE.read_text(encoding="utf-8")
    return {
        key
        for key in re.findall(r"`([a-z][a-z_]*(?:\.[a-z][a-z_]*)+)`", text)
        if not key.endswith(_NOT_A_CONFIG_KEY_SUFFIX)
    }


def _switch_table_wired_keys() -> list[tuple[str, str]]:
    """§2 开关表里标 ✅ 的行 → `(开关名, 它声称接线的键)`。

    表格**最后一格**是"接线"列：`✅ \\`rerank.enabled\\`` 表示那个开关真的能关。
    """
    lines = _CONFIG_REFERENCE.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## 2. 消融开关")]
    assert starts, "没找到 §2「消融开关」一节 —— 文档结构改了？"
    start = starts[0]
    ends = [i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")]
    end = ends[0] if ends else len(lines)

    wired: list[tuple[str, str]] = []
    for line in lines[start:end]:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        # 表头与分隔行：首格不是反引号包着的开关名
        if len(cells) < 6 or not cells[0].startswith("`"):
            continue
        last = cells[-1]
        if not last.startswith("✅"):
            continue
        wired += [(cells[0].strip("`"), key) for key in re.findall(r"`([a-z][a-z_.]*)`", last)]
    return wired


def test_every_yaml_key_the_code_accepts_is_declared_in_config_reference() -> None:
    """**代码接受的每个 yaml 键都必须在 `docs/config-reference.md` 里出现。**

    那份文档是配置键的**唯一声明处**。新增一个键却不写文档，
    下一个人就只能在代码里考古——而 §13 的每条对照都要回答"这次跑的是哪套值"。

    ⚠ 这与上面 `.env.example` 那条是**同一种漂移**，只是对象换成 yaml。

    ⚠ **反向不检查**：文档里有、代码里没有的键**大量存在且合法**——它们是 ⬜ 待接线的
    落点（`checker.*` / `agent.*` / `neighbor.enabled` …）。而"文档声称已接线、代码却不认"
    那一条由下面 `test_switch_table_...` 单独钉。
    """
    code = _code_yaml_keys()
    assert code, "一个 yaml 键都没解析出来 —— 这条用例是空过的"

    missing = sorted(code - _documented_dotted_keys())
    assert missing == [], (
        "这些 yaml 键代码接受、而 `docs/config-reference.md` 里没写"
        "（那份文档是唯一声明处，新增键要同时写进去）：\n  " + "\n  ".join(missing)
    )


def test_switch_table_only_claims_wired_keys_that_the_code_accepts() -> None:
    """**§2 开关表里标 ✅ 的键，代码必须真的接受。**

    那张表的"接线"列是**唯一一处声明"哪个消融开关真的能关"**的地方，而且是机器可读的
    （`✅ \\`rerank.enabled\\``）。标了 ✅ 而代码不认，就等于**承诺了一个做不到的消融**——
    它的表现是"照文档改了配置、行为一点没变"，**不报错**。

    ⚠ 只覆盖 §2：文档其余部分在 heading 上标 ✅/⬜，粒度更粗也更易变，**没有自动化**。
    """
    wired = _switch_table_wired_keys()
    assert wired, "§2 表里一个 ✅ 都没解析出来 —— 这条用例是空过的"

    code = _code_yaml_keys()
    offenders = [f"{switch} 声称接了 {key}" for switch, key in wired if key not in code]
    assert offenders == [], (
        "§2 的开关表标了 ✅、但 `common/config.py` 不接受这些键：\n  " + "\n  ".join(offenders)
    )


# ── 2. 默认配置可加载 ──────────────────────────────────────────────────


def test_default_profile_loads_the_repo_yaml(config_dir: Path) -> None:
    """`configs/default.yaml` 能加载，且关键值与文档一致。"""
    cfg = _load(config_dir)

    assert cfg.profile == "default"
    assert cfg.storage.sqlite.busy_timeout_ms == 5000
    assert cfg.storage.qdrant.collection == "memories"
    assert cfg.retrieval.prefetch_limit == 200
    assert cfg.retrieval.rrf.k == RRF_K
    assert cfg.retrieval.rrf.weights == (0.5, 0.5)
    assert cfg.retrieval.query_instruction == ""
    assert cfg.server.workers == 1
    # 2026-09-25 起默认为 True（**D21 推翻了 §11.3 的"不加"**，有 346 题的反例）——
    # 这条断言是那个决定在本仓的**回归位**：谁把它改回去，这里立刻红。
    assert cfg.packaging.inject_abs_time is True


def test_packaging_switch_rejects_a_non_bool(config_dir: Path) -> None:
    """`packaging.inject_abs_time` 只有开/关两种取值——写成日期格式串必须**响亮失败**。

    它是一个**对照臂的开关**，不是"用哪种日期"的选择器：口径只有 `day_granularity` 一处
    （`common/render.py`）。放一个宽松的值进来，T1 的两臂就会同时在两个维度上不同，
    于是那个对照再也归因不了——而**分数看起来完全正常**。
    """
    from tianxi_am.common.config import ConfigError, load_config

    other = config_dir / "bad_switch"
    other.mkdir()
    (other / "default.yaml").write_text(
        "packaging:\n  inject_abs_time: '2026-07-26'\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="packaging.inject_abs_time"):
        load_config({"TIANXI_SQLITE_PATH": "x.db", "AML_EMB_API_KEY": "k"}, config_dir=other)


def test_config_dir_env_redirects_where_yaml_is_read(config_dir: Path) -> None:
    """`TIANXI_CONFIG_DIR` **重定向 yaml 的读取目录**。

    ⚠ 这条不只是"多测一个变量"：它是**替代集合**能不能做的前提——§13 的每个 arm
    都需要一份冻结的配置（`configs/runs/`），而"指定另一份配置跑"靠的就是这个变量。
    """
    other = config_dir / "frozen_run"
    other.mkdir()
    (other / "default.yaml").write_text(
        "storage:\n  qdrant:\n    collection: arm_a\n", encoding="utf-8"
    )

    # ⚠ 不传 `config_dir=` 参数——显式参数优先于环境变量，传了就测不到这个变量
    cfg = load_config({**_MIN_ENV, ENV_CONFIG_DIR: str(other)})

    assert cfg.storage.qdrant.collection == "arm_a"
    # 其余键仍走内置默认值（那一份 default.yaml 只写了 collection）
    assert cfg.retrieval.prefetch_limit == 200


def test_explicit_config_dir_argument_beats_the_env_var(config_dir: Path) -> None:
    """显式参数**优先于**环境变量——调用方比环境更具体。"""
    cfg = _load(config_dir, **{ENV_CONFIG_DIR: str(config_dir / "nowhere")})
    assert cfg.storage.qdrant.collection == "memories"


def test_repo_config_dir_loads_without_an_explicit_config_dir() -> None:
    """不传 `config_dir` 时走默认的 `configs/`——**直接从仓库根跑得起来**。"""
    repo_root_configs = _REPO / "configs" / "default.yaml"
    assert repo_root_configs.exists(), "configs/default.yaml 必须存在，否则默认路径加载不了"

    cfg = load_config(_MIN_ENV, config_dir=_REPO / "configs")
    assert cfg.profile == "default"


def test_missing_default_yaml_fails_loudly(tmp_path: Path) -> None:
    """**缺 `default.yaml` 必须报错**，不静默退回内置默认值。

    静默退回的后果：跑的是**没人声明过**的一套值，而"配置化"成了一句空话。
    """
    with pytest.raises(ConfigError, match="配置文件不存在"):
        load_config(_MIN_ENV, config_dir=tmp_path / "nowhere")


def test_unknown_profile_fails_loudly(config_dir: Path) -> None:
    """选了一个不存在的 profile ⇒ 报错，**不静默回退到 default**。

    静默回退会让"我明明选了 submit"变成一个查不出来的问题。
    """
    with pytest.raises(ConfigError, match="配置文件不存在"):
        _load(config_dir, **{ENV_PROFILE: "submit"})


# ── 3. profile 叠加 ────────────────────────────────────────────────────


def test_profile_overlays_only_what_it_names(config_dir: Path) -> None:
    """`local.yaml` **只覆盖它写了的键**，其余继承 `default.yaml`。

    这正是 profile 存在的理由：让"开发期与提交期差在哪"一眼可见
    （`configs/CLAUDE.md`）。若它整体替换，那份信息就丢了。
    """
    base = _load(config_dir)
    local = _load(config_dir, **{ENV_PROFILE: "local"})

    assert local.profile == "local"
    assert local.storage.qdrant.collection == "memories_dev"  # ← local.yaml 改了它
    # 其余一律继承
    assert local.retrieval.rrf.weights == base.retrieval.rrf.weights
    assert local.retrieval.prefetch_limit == base.retrieval.prefetch_limit
    assert local.models.embedder == base.models.embedder


# ── 4. 环境变量覆盖 ────────────────────────────────────────────────────


def test_env_overrides_paths_and_secrets(config_dir: Path) -> None:
    """环境变量**能**覆盖它拥有的那几项：路径、端点、密钥、worker 数。"""
    cfg = _load(
        config_dir,
        **{
            ENV_SQLITE_PATH: "/tmp/other.db",
            "TIANXI_QDRANT_URL": "http://qdrant.internal:6333",
            "TIANXI_EMBED_CACHE_DIR": "/tmp/other_cache",
            ENV_EMBED_BASE_URL: "http://gw/v1/",
            ENV_EMBED_API_KEY: "sk-xyz",
        },
    )

    assert cfg.storage.sqlite.path == "/tmp/other.db"
    assert cfg.storage.qdrant.url == "http://qdrant.internal:6333"
    assert cfg.cache.embed.dir == "/tmp/other_cache"
    assert cfg.embed_base_url == "http://gw/v1/"
    assert cfg.embed_api_key == "sk-xyz"


def test_empty_env_var_does_not_wipe_a_default(config_dir: Path) -> None:
    """**空字符串不算覆盖**。

    `.env` 里一行 `TIANXI_SQLITE_PATH=` 不该把路径变成空串——那会让服务在建库时
    才炸，而不是在配置阶段。空值一律退回默认（或报错）。
    """
    cfg = _load(config_dir, **{ENV_SQLITE_PATH: ""})
    assert cfg.storage.sqlite.path == "var/tianxi.db"


def test_thresholds_are_not_overridable_from_env(config_dir: Path) -> None:
    """**阈值只能来自 yaml**——环境变量里放一个同名的也没用。

    这条是"每个键只有一个家"的守门人：两处都能设的值，最终会变成
    "跑出来的结果和 yaml 里写的不一样，而没人知道为什么"。
    """
    cfg = _load(
        config_dir,
        TIANXI_PREFETCH_LIMIT="999",
        TIANXI_RRF_K="2",
        AML_EMB_MODEL="some-other-model",
    )

    assert cfg.retrieval.prefetch_limit == 200  # yaml 说了算
    assert cfg.retrieval.rrf.k == RRF_K
    assert cfg.models.embedder == "Qwen/Qwen3-Embedding-8B"


def test_env_is_not_mutated_by_loading() -> None:
    """`load_config()` **不写**环境变量（它只读）。"""
    env = dict(_MIN_ENV)
    load_config(env, config_dir=_REPO / "configs")
    assert env == _MIN_ENV


# ── 5. 不合法值：全部**拒绝**而不是警告 ─────────────────────────────────


def test_illegal_rrf_k_is_rejected(config_dir: Path) -> None:
    """**`rrf.k` 不是可调项**（D5）——填错必须拒绝启动，而不是警告后照用。

    填错的后果是"不报错但结果错"：两种写法的名次都看起来正常，只有分数差一个数量级。
    """
    (config_dir / "default.yaml").write_text(
        (_REPO / "configs" / "default.yaml").read_text(encoding="utf-8").replace("k: 61", "k: 60"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="retrieval.rrf.k"):
        _load(config_dir)


def test_rrf_k_is_the_same_constant_in_both_places() -> None:
    """`RRF_K` 在 `common/config.py` 与 `retrieve/fusion.py` 各有一份 ⇒ **必须相等**。

    两份是**有意**的：配置层拒绝非法值（启动时），策略层拒绝非法参数（直接调用方）。
    但值本身只能有一个——这条用例就是那道保险。见 D5。
    """
    from tianxi_am.retrieve.fusion import RRF_K as STRATEGY_K

    assert STRATEGY_K == RRF_K == 61


def test_model_default_is_the_same_in_both_places() -> None:
    """`models.embedder` 的默认值在两处各有一份 ⇒ **必须相等**（同上一条的理由）。"""
    from tianxi_am.embed.qwen3_embedding import DEFAULT_MODEL

    assert AppConfig().models.embedder == DEFAULT_MODEL


def test_workers_must_be_one(config_dir: Path) -> None:
    """**`workers != 1` 拒绝启动**（§15）。

    ⚠ **D25 之后原因变了**（结论没变）：旧理由是"`SessionLocks` 是进程内锁，
    多 worker 下每个 worker 各有各的锁 ⇒ 静默失效"。`SessionLocks` 已删，
    那条不再成立；现在拒的是**"放开多 worker 需要的验证一件都没做"**
    （并发写压力、`busy_timeout` 争用、多进程各自的 Qdrant 客户端）。
    ⇒ 这条不能只是文档警告，因为**去掉它的诱惑比从前更大了**。
    """
    with pytest.raises(ConfigError, match="server.workers"):
        _load(config_dir, **{ENV_WORKERS: "4"})

    with pytest.raises(ConfigError, match="必须是整数"):
        _load(config_dir, **{ENV_WORKERS: "many"})

    assert _load(config_dir, **{ENV_WORKERS: "1"}).server.workers == 1
    assert _load(config_dir, **{ENV_WORKERS: " 1 "}).server.workers == 1  # 容忍空白


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("retrieval.prefetch_limit", 0),
        ("retrieval.prefetch_limit", -1),
        ("storage.sqlite.busy_timeout_ms", 0),
        ("rerank.timeout_seconds", 0),
    ],
)
def test_non_positive_numbers_are_rejected(config_dir: Path, key: str, value: int) -> None:
    """非正数一律拒绝（它们各自都有明确的语义下限）。"""
    text = (_REPO / "configs" / "default.yaml").read_text(encoding="utf-8")
    leaf = key.rsplit(".", 1)[-1]
    (config_dir / "default.yaml").write_text(
        re.sub(rf"^(\s*{leaf}:)\s*\d+", rf"\g<1> {value}", text, flags=re.MULTILINE),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=re.escape(key)):
        _load(config_dir)


def test_bad_weights_are_rejected(config_dir: Path) -> None:
    """权重：必须两个、不得为负、不得同时为 0（那等于不检索）。"""
    text = (_REPO / "configs" / "default.yaml").read_text(encoding="utf-8")

    for bad, match in (
        ("[1.0]", "必须是两个数"),
        ("[0.5]", "必须是两个数"),
        ("[-1.0, 2.0]", "不得为负"),
        ("[0, 0]", "同时为 0"),
        ('["a", "b"]', "必须是两个数"),
    ):
        (config_dir / "default.yaml").write_text(
            text.replace("weights: [0.5, 0.5]", f"weights: {bad}"), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match=match):
            _load(config_dir)


# ── 6. 缺必填项：错误必须**说清楚缺什么、从哪来** ───────────────────────


@pytest.mark.parametrize("missing", [ENV_EMBED_BASE_URL, ENV_EMBED_API_KEY])
def test_missing_required_secret_names_the_variable(config_dir: Path, missing: str) -> None:
    """缺密钥/端点时，错误里必须**点名那个变量**。

    "配置错误"这种消息等于没报——§2.15 的单请求预算是 30 分钟，
    而这类错在启动时就该被消灭。
    """
    env = {k: v for k, v in _MIN_ENV.items() if k != missing}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env, config_dir=config_dir)

    message = str(excinfo.value)
    assert missing in message  # 点名了变量
    assert ".env.example" in message  # 告诉了去哪填


def test_blank_required_secret_is_treated_as_missing(config_dir: Path) -> None:
    """**空白不算填了**——`.env` 里留一行 `AML_EMB_API_KEY=` 是常见状态。"""
    with pytest.raises(ConfigError, match=ENV_EMBED_API_KEY):
        _load(config_dir, **{ENV_EMBED_API_KEY: "   "})


# ── 7. yaml 的键名与路径：拼错要响 ──────────────────────────────────────


def test_unknown_yaml_key_fails_loudly(config_dir: Path) -> None:
    """**拼错的键必须响亮失败。**

    `prefetch_limt: 999` 被静默忽略 ⇒ 跑的是 200，而没有任何人会发现。
    """
    (config_dir / "default.yaml").write_text("retrieval:\n  prefetch_limt: 999\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="不认识的键"):
        _load(config_dir)


def test_env_owned_keys_are_rejected_in_yaml(config_dir: Path) -> None:
    """**路径 / 端点 / 密钥 / worker 数不能写进 yaml**——它们只在 `.env` 里。

    写进去会得到一个**有两个家**的键，而那正是"结果与配置对不上"的成因。
    """
    for yaml_text, match in (
        ("storage:\n  sqlite:\n    path: data/x.db\n", "storage.sqlite"),
        ("storage:\n  qdrant:\n    url: http://x\n", "storage.qdrant"),
        ("cache:\n  embed:\n    dir: /tmp/c\n", "不认识的段"),
        ("server:\n  workers: 4\n", "不认识的段"),
    ):
        (config_dir / "default.yaml").write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ConfigError, match=match):
            _load(config_dir)


def test_malformed_yaml_fails_loudly(config_dir: Path) -> None:
    """yaml 语法错也要响亮（不是 `yaml` 自己的 traceback 了事）。"""
    (config_dir / "default.yaml").write_text("retrieval: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="解析"):
        _load(config_dir)


def test_wrong_type_fails_loudly(config_dir: Path) -> None:
    """类型不对时错误里带**键名与收到的值**。"""
    (config_dir / "default.yaml").write_text(
        "retrieval:\n  prefetch_limit: many\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="retrieval.prefetch_limit"):
        _load(config_dir)


# ── 8. 进程形态守卫 ────────────────────────────────────────────────────


def test_single_process_guard_passes_in_this_process() -> None:
    """当前进程（pytest）不是多进程 worker ⇒ 守卫**不得**误报。"""
    assert_single_process()


def _probe_guard(queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """在**真子进程**里跑守卫，把结果放回队列（必须是模块级函数：spawn 要可 pickle）。"""
    try:
        assert_single_process()
    except ConfigError as exc:
        queue.put(str(exc))
    else:
        queue.put(None)


def test_single_process_guard_rejects_a_real_child_process() -> None:
    """**真**子进程里守卫必须响——否则它对 `--workers 4` 就没有任何效力。

    ⚠ 用真的 `multiprocessing` 子进程，而不是 monkeypatch：monkeypatch 只能证明
    "如果 `parent_process()` 非 None 我们会报错"，证明不了**我们真的能看出**自己是子进程。
    uvicorn 的 `--workers N`（N>1）与 `--reload` 派生子进程走的是同一套机制
    （`multiprocessing.get_context("spawn").Process`），所以这条是那个场景的忠实替身。
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    proc = context.Process(target=_probe_guard, args=(queue,))
    proc.start()
    proc.join(timeout=120)

    assert proc.exitcode == 0, "子进程自己崩了——守卫不该以异常之外的方式失败"
    message = queue.get(timeout=30)
    assert message is not None, "守卫在真子进程里**没有**响 —— 它对多 worker 无效"
    assert "--workers 1" in message


def test_run_parallel_in_this_session_is_not_a_child() -> None:
    """`threading.Thread` **不是** `multiprocessing` 子进程 ⇒ 守卫不得误伤线程池用法。

    这条钉住的是服务真正的运行形态：FastAPI 的 `def` 路由跑在**线程池**里，
    那条路径上守卫必须是 pass。
    """
    import threading

    results: list[object] = []

    def worker() -> None:
        try:
            assert_single_process()
            results.append(None)
        except ConfigError as exc:  # pragma: no cover — 出现了就是 bug
            results.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=30)
    assert results == [None]


# ── 9. `.env` 的读取 ───────────────────────────────────────────────────
#
# ⚠ 这一组**全部不碰真实环境、也不读仓库根的 `.env`**：`load_config()` 只在 `env=None`
# （生产路径）时读 `.env`，而这里一律用 `env_file=` 指到 `tmp_path`。


def test_dotenv_is_read_on_the_production_path(config_dir: Path, tmp_path: Path) -> None:
    """**`env=None` 时会读 `.env`**——否则 `make serve` 会在启动时"缺密钥"。

    ⚠ 守的是一个真缺口：`.env` 被文档声明成"密钥的家"，而 `uv run` 不加载 `.env`、
    Makefile 也不 include 它——少了这一步就没有任何东西读它。
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AML_EMB_BASE_URL=https://gw.example/v1\nAML_EMB_API_KEY=sk-from-dotenv\n",
        encoding="utf-8",
    )

    cfg = load_config(config_dir=_REPO / "configs", env_file=env_file)

    assert cfg.embed_base_url == "https://gw.example/v1"
    assert cfg.embed_api_key == "sk-from-dotenv"


def test_dotenv_parses_our_own_file_format(config_dir: Path, tmp_path: Path) -> None:
    """`.env.example` 里用到的写法都要能解析：空行、注释、`export`、引号、行内注释。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 一整行注释\n"
        "\n"
        "   \n"
        "AML_EMB_BASE_URL=https://gw/v1\n"
        "export AML_EMB_API_KEY=sk-exported\n"
        'TIANXI_SQLITE_PATH="var/quoted.db"\n'
        "TIANXI_QDRANT_URL=http://localhost:6333   # 这一行是行内注释\n"
        "TIANXI_EMBED_CACHE_DIR=/tmp/cache#不是注释（# 前没有空白）\n",
        encoding="utf-8",
    )

    cfg = load_config(config_dir=_REPO / "configs", env_file=env_file)

    assert cfg.embed_base_url == "https://gw/v1"
    assert cfg.embed_api_key == "sk-exported"
    assert cfg.storage.sqlite.path == "var/quoted.db"  # 引号被剥掉
    assert cfg.storage.qdrant.url == "http://localhost:6333"  # 行内注释被剥掉
    assert cfg.cache.embed.dir == "/tmp/cache#不是注释（# 前没有空白）"


def test_real_env_vars_beat_dotenv(config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """**真实环境变量压过 `.env`**——`.env` 只是"这台机器的环境"的本地副本。

    顺序反了的话，`export AML_EMB_API_KEY=...` 会被 `.env` 里的旧值悄悄盖掉。
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AML_EMB_BASE_URL=https://from-dotenv/v1\nAML_EMB_API_KEY=sk-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AML_EMB_API_KEY", "sk-from-shell")

    cfg = load_config(config_dir=_REPO / "configs", env_file=env_file)

    assert cfg.embed_api_key == "sk-from-shell"  # shell 说了算
    assert cfg.embed_base_url == "https://from-dotenv/v1"  # 只有 .env 有的那个照旧生效


def test_missing_dotenv_is_fine_but_missing_keys_is_not(config_dir: Path, tmp_path: Path) -> None:
    """**没有 `.env` 不是错误**（CI / 容器里直接用真实环境变量）——但缺密钥仍然要响。"""
    env = {**_MIN_ENV}
    # 只有 base_url、没有 key ⇒ 仍然要响亮失败
    with pytest.raises(ConfigError, match="embed.api_key"):
        load_config(
            {"AML_EMB_BASE_URL": "https://gw/v1"},
            config_dir=config_dir,
            env_file=tmp_path / "no.env",
        )
    assert env[ENV_EMBED_API_KEY]  # _MIN_ENV 本身是完整的


def test_env_dict_never_reads_dotenv(config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """**传了 `env` dict 就绝不读磁盘上的 `.env`。**

    否则"测试不依赖真实环境变量"这条纪律就破了——机器上的一份 `.env`
    会悄悄改变测试结果。
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "AML_EMB_BASE_URL=https://should-not-be-read/v1\nAML_EMB_API_KEY=sk-should-not\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="embed.base_url"):
        load_config({}, config_dir=_REPO / "configs")  # 传了 env（空 dict）⇒ 不读 .env


def test_env_file_location_is_overridable(config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """`TIANXI_ENV_FILE` 能换 `.env` 的位置（预检靠它做到不依赖机器状态）。

    ⚠ 这条**必须走生产路径**（`env=None`）：传 `env` dict 时根本不读磁盘，
    在 dict 里放 `TIANXI_ENV_FILE` 是没有意义的（它管的是"去哪读环境"）。
    """
    other = tmp_path / "custom.env"
    other.write_text("AML_EMB_BASE_URL=https://custom/v1\nAML_EMB_API_KEY=k\n", encoding="utf-8")

    monkeypatch.setenv(ENV_FILE, str(other))
    monkeypatch.delenv("AML_EMB_BASE_URL", raising=False)
    monkeypatch.delenv("AML_EMB_API_KEY", raising=False)

    cfg = load_config(config_dir=_REPO / "configs")

    assert cfg.embed_base_url == "https://custom/v1"
