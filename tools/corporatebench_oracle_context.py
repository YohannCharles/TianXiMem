"""CorporateBench 的记忆上下文 oracle 诊断；不修改 Add/Search 或答案 pipeline。

prepare 构造原文/事实两组；scope、sentences、matches 追加独立记忆表示对照。
run 串行调用冻结的原始答案模板；report 离线判分。
ground-truth graph 只用于本地 oracle 的选源和事实参考，不是可部署的检索策略。
事实片段保留独立实体/关系/时间，不写题目答案、汇总数字或汇总名单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "harness"))

from eval.datasets.registry import benchmark_dir  # noqa: E402
from eval.harness import extra_pipeline as ep  # noqa: E402
from eval.harness.corporatebench_pipeline import score_corporatebench  # noqa: E402

IDS = (
    0,
    3,
    5,
    6,
    12,
    26,
    37,
    51,
    87,
    88,
    148,
    228,
    4,
    7,
    9,
    15,
    17,
    32,
    35,
    49,
    135,
    190,
    220,
    221,
)
ARMS = ("oracle_raw", "oracle_facts")
ENTITY = "http://storybeat.company/entity/"
CORE = "https://storybeat.company/ontology/core#"
COMPANY = "https://storybeat.company/ontology/company#"
MEET = "https://storybeat.company/ontology/meetings#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in map(json.loads, path.read_text().splitlines())}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def atom(raw: str) -> str:
    if raw.startswith('"'):
        token = re.match(r'"(?:\\.|[^"\\])*"', raw)
        assert token is not None
        return json.loads(token[0])
    if raw.startswith("<"):
        value = raw[1:-1]
        return value.removeprefix(ENTITY)
    return raw


def header(text: str, name: str) -> str:
    match = re.search(r"^" + re.escape(name) + r": (.*)$", text, re.MULTILINE)
    return match[1] if match else ""


class Corpus:
    def __init__(self, path: Path):
        self.data: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.starts: dict[tuple[str, str, str], str] = {}
        with zipfile.ZipFile(path) as archive:
            self.documents = {
                Path(name).name: archive.read(name).decode()
                for name in archive.namelist()
                if name.startswith("documents/") and name.endswith(".txt")
            }
            self.graph_text = archive.read("graph.nq").decode()
        for line in self.graph_text.splitlines():
            match = re.match(r"^(<[^>]+>|_:\S+) <([^>]+)> (.*?) \.$", line)
            if match:
                self.data[atom(match[1])][match[2]].append(atom(match[3]))
            temporal = re.match(
                r"^<<<([^>]+)> <([^>]+)> <([^>]+)>>> "
                r'<https://storybeat.company/ontology/core#beginning> "([0-9-]+)"',
                line,
            )
            if temporal:
                s, p, o, date = temporal.groups()
                self.starts[(s.removeprefix(ENTITY), p, o.removeprefix(ENTITY))] = date
        self.meetings = self.typed(MEET + "Meeting")
        self.employees = self.typed(COMPANY + "Employee")
        self.departments = self.typed(COMPANY + "Department")
        self.company = self.typed(COMPANY + "Company")[0]
        self.meeting_sources: dict[str, list[str]] = defaultdict(list)
        for name in self.documents:
            for meeting in self.values(name, CORE + "representsMeeting"):
                self.meeting_sources[meeting].append(name)
        self.attendees: dict[str, list[str]] = defaultdict(list)
        for employee in self.employees:
            for meeting in self.values(employee, MEET + "attend"):
                self.attendees[meeting].append(employee)

    def values(self, subject: str, predicate: str) -> list[str]:
        return self.data.get(subject, {}).get(predicate, [])

    def one(self, subject: str, predicate: str) -> str:
        values = self.values(subject, predicate)
        return values[0] if values else ""

    def typed(self, entity_type: str) -> list[str]:
        return sorted(s for s, props in self.data.items() if entity_type in props.get(RDF_TYPE, []))

    def name(self, subject: str) -> str:
        return (
            self.one(subject, CORE + "name")
            or self.one(subject, MEET + "title")
            or self.one(subject, COMPANY + "legalName")
        )

    def employee_sources(self, employee: str) -> list[str]:
        company_name = self.name(self.company)
        employment_start = self.starts.get((employee, COMPANY + "worksAt", self.company))
        name = self.name(employee)
        candidates = [
            filename
            for filename, body in self.documents.items()
            if header(body, "From").startswith(name + " <") and company_name in body
        ]
        assert candidates, name
        if employment_start:
            exact = [
                filename
                for filename in candidates
                if header(self.documents[filename], "Date") == employment_start
                and any(
                    word in self.documents[filename]
                    for word in ("commenced my role", "orientation schedule")
                )
            ]
            assert exact, name
            return exact[:1]
        preferred = [
            filename
            for filename in candidates
            if self.one(employee, COMPANY + "workEmail") in header(self.documents[filename], "From")
        ]
        # 原文中公司的任职/署名线索；优先最早的有效公司邮箱通信。
        pool = preferred or candidates
        return sorted(pool, key=lambda f: (header(self.documents[f], "Date"), f))[:1]

    def relevant(self, variables: dict) -> tuple[str, list[str], list[str]]:
        if "company" in variables or not variables:
            return "employees", self.employees, []
        if "meeting" in variables:
            meeting_name = variables["meeting"].casefold()
            scope = [m for m in self.meetings if self.name(m).casefold() == meeting_name]
        elif "project" in variables:
            projects = {
                p
                for p in self.data
                if p.startswith("project_")
                and self.name(p).casefold() == variables["project"].casefold()
            }
            scope = [
                m
                for m in self.meetings
                if projects.intersection(self.values(m, MEET + "mentionsProject"))
            ]
        else:
            topic = variables["meeting_topic"].casefold()
            scope = [m for m in self.meetings if self.one(m, CORE + "topic").casefold() == topic]
        scope.sort(key=lambda m: (self.one(m, CORE + "date"), m))
        filenames = [filename for m in scope for filename in self.meeting_sources[m]]
        assert scope and all(self.meeting_sources[m] for m in scope), variables
        return "meetings", scope, filenames

    def raw_fragment(self, filename: str, employee: bool) -> str:
        text = self.documents[filename]
        selected = [
            line
            for line in text.splitlines()
            if re.match(r"^(Message-ID|From|To|Date|Subject|Scheduled for):", line)
        ]
        body_parts = text.split("\n\n")[1:]
        if employee:
            excerpts = [
                part
                for part in body_parts
                if any(
                    word in part
                    for word in ("Zenith Labs", "commenced my role", "orientation schedule")
                )
            ]
            selected.extend(excerpts)
        elif "CALENDAR INVITE" in text:
            lines = text.splitlines()
            begin = lines.index("Attendees:") if "Attendees:" in lines else len(lines)
            attendees = []
            for line in lines[begin:]:
                if attendees and not line.strip():
                    break
                attendees.append(line)
            selected.extend(attendees)
        else:
            paragraphs = [
                part
                for part in body_parts
                if part.strip()
                and not part.startswith(("MIME-Version:", "Content-Type:"))
                and not (len(part) < 80 and part.startswith(("Hi ", "Hey ")))
            ]
            # 保留活动本身的开场说明；主题/参会/项目关系需要完整正文校验。
            selected.extend(paragraphs[:2])
            selected.extend(part for part in body_parts if "biweekly" in part or "weekly" in part)
        for snippet in selected:
            assert snippet in text, filename
        content = "\n\n".join(dict.fromkeys(selected))
        return f"[{header(text, 'Date')}] Source: {filename}\n{content}"

    def employee_fact(self, employee: str, sources: list[str]) -> tuple[str, list[str]]:
        fields = [
            "Entity type: Employee",
            f"Name: {self.name(employee)}",
            f"Employer: {self.name(self.company)}",
        ]
        start = self.starts.get((employee, COMPANY + "worksAt", self.company))
        if start:
            fields.append(f"Employment relationship began: {start}")
            uncertain = [
                "Employment relationship began: graph temporal annotation; prose is indirect"
            ]
        else:
            fields.append("Employment relationship: pre-existing in the initial KB state")
            fields.append("Exact hiring date: not recorded")
            uncertain = [
                "Initial-state employment status: oracle KB relation, not an explicit prose date"
            ]
        return "\n".join(fields + ["Source: " + ", ".join(sources)]), uncertain

    def meeting_fact(self, meeting: str, variables: dict) -> tuple[str, list[str]]:
        fields = [
            "Entity type: Meeting occurrence",
            f"Meeting title: {self.name(meeting)}",
            f"Occurrence date: {self.one(meeting, CORE + 'date')}",
        ]
        sources = self.meeting_sources[meeting]
        oracle_fields = []
        if not any("Scheduled for:" in self.documents[s] for s in sources):
            oracle_fields.append("Occurrence date: oracle mapping of document Date to meeting date")
        if "meeting_topic" in variables:
            fields.append("Topic: " + self.one(meeting, CORE + "topic"))
            oracle_fields.append("Topic: oracle-normalized topic label")
        if "meeting_frequency" in variables:
            fields.append("Frequency: " + self.one(meeting, MEET + "recurrence"))
            oracle_fields.append("Frequency: KB recurrence may not be explicit in each document")
        if "project" in variables:
            for project in self.values(meeting, MEET + "mentionsProject"):
                fields.append("Associated project: " + self.name(project))
            oracle_fields.append("Associated project: oracle KB relation")
        if "meeting" in variables:
            for employee in sorted(self.attendees[meeting], key=self.name):
                fields.append("Participant: " + self.name(employee))
            oracle_fields.append("Participant: KB attend relation, possibly implicit in prose")
        fields.append("Source: " + ", ".join(sources))
        return "\n".join(fields), oracle_fields


def prepare(output: Path) -> None:
    import tiktoken

    source_dir = ROOT / "eval/reports/runs/base-cb/corp-kb_qa"
    archive = ROOT / benchmark_dir() / "corporatebench/data/kb/zenith.kb"
    question_file = ROOT / benchmark_dir() / "corporatebench/data/kb_qa/zenith_questions.json"
    corpus = Corpus(archive)
    questions = json.loads(question_file.read_text())["questions"]
    inputs = read_rows(source_dir / "input.jsonl")
    baseline = read_rows(source_dir / "answers.jsonl")
    encoding = tiktoken.get_encoding("o200k_base")
    rows = {arm: [] for arm in ARMS}
    provenance = []
    for number in IDS:
        question = questions[number]
        ident = f"kb_qa-{number}"
        original = inputs[ident]
        variables = question["variables"]
        if number == 0:
            raw_names = ["agenda_Department_Strategy_and_Goals_Review_5897.txt"]
            raw = [corpus.raw_fragment(raw_names[0], employee=False)]
            facts = [
                "\n".join(
                    [
                        "Entity type: Department",
                        f"Name: {corpus.name(d)}",
                        f"Company: {corpus.name(corpus.company)}",
                        "Source: " + raw_names[0],
                    ]
                )
                for d in corpus.departments
            ]
            oracle_fields = [
                "Department type: oracle KB type, supported by prose department references"
            ]
        else:
            kind, scope, raw_names = corpus.relevant(variables)
            if kind == "employees":
                raw_names = [filename for e in scope for filename in corpus.employee_sources(e)]
                raw = [corpus.raw_fragment(name, employee=True) for name in raw_names]
                records = [corpus.employee_fact(e, corpus.employee_sources(e)) for e in scope]
            else:
                raw = [corpus.raw_fragment(name, employee=False) for name in raw_names]
                records = [corpus.meeting_fact(m, variables) for m in scope]
            facts = [record[0] for record in records]
            oracle_fields = sorted({field for record in records for field in record[1]})
        contexts = {"oracle_raw": "\n\n".join(raw), "oracle_facts": "\n\n".join(facts)}
        for arm, context in contexts.items():
            row = dict(original, retrieved_context=context)
            row["oracle_fragments"] = len(raw if arm == "oracle_raw" else facts)
            assert 0 < row["oracle_fragments"] <= 100
            rows[arm].append(row)
        provenance.append(
            {
                "id": ident,
                "question": original["question"],
                "answer_type": original["gold_answer"]["answer_type"],
                "scope_policy": (
                    "all records of relevant company/series/topic/project; "
                    "no date or frequency answer filtering"
                ),
                "raw_sources": raw_names,
                "oracle_only_or_normalized_fields": oracle_fields,
                "context_tokens": {
                    "baseline": len(encoding.encode(original["retrieved_context"])),
                    **{arm: len(encoding.encode(context)) for arm, context in contexts.items()},
                },
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        write_rows(output / arm / "input.jsonl", rows[arm])
        write_rows(
            output / arm / "prompts.jsonl",
            [
                {
                    "id": row["id"],
                    "prompt": ep.ANSWER_PROMPT.format(
                        memories=row["retrieved_context"], question=row["question"]
                    ),
                }
                for row in rows[arm]
            ],
        )
    write_rows(output / "baseline-answers.jsonl", [baseline[f"kb_qa-{n}"] for n in IDS])
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "manifest.json",
        {
            "scope": (
                "24 hand-picked diagnostic questions, 12 int and 12 List[str]; "
                "not representative accuracy"
            ),
            "ids": [f"kb_qa-{number}" for number in IDS],
            "source_run": "base-cb",
            "model": "Qwen/Qwen3.5-9B",
            "temperature": 0,
            "max_tokens": 1024,
            "answer_prompt": ep.ANSWER_PROMPT,
            "answer_prompt_sha256": hashlib.sha256(ep.ANSWER_PROMPT.encode()).hexdigest(),
            "oracle_warning": (
                "Graph/variables are oracle annotations for this diagnostic only. "
                "Oracle facts include information implicit or absent in prose. "
                "No final answers or precomputed aggregates enter prompts. "
                "Candidate scopes include nonmatching dates/frequencies for negative controls."
            ),
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    archive,
                    question_file,
                    source_dir / "input.jsonl",
                    source_dir / "answers.jsonl",
                    ROOT / "eval/harness/corporatebench_pipeline.py",
                    ROOT / "eval/harness/extra_pipeline.py",
                    Path(__file__),
                )
            },
        },
    )
    print(
        json.dumps({"prepared": str(output), "questions": len(IDS), "arms": list(ARMS)}), flush=True
    )


def run(output: Path) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    import api_config
    import httpx

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] == api_config.ANSWER_MODEL, (
        "Model changed; prepare a separate experiment"
    )
    active_arms = json.loads((output / "manifest.json").read_text()).get("arms", list(ARMS))
    for arm in active_arms:
        prompts = read_rows(output / arm / "prompts.jsonl")
        finished = read_rows(output / arm / "answers.jsonl")
        for ident in manifest["ids"]:
            if ident in finished:
                continue
            prompt = prompts[ident]["prompt"]
            began = time.monotonic()
            for attempt in range(2):
                try:
                    answer = ep._chat(
                        api_config.ANSWER_API_BASE,
                        api_config.ANSWER_API_KEY,
                        api_config.ANSWER_MODEL,
                        prompt,
                        max_tokens=manifest["max_tokens"],
                        timeout=120,
                    )
                    break
                except (httpx.HTTPError, KeyError) as exc:
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "id": ident,
                                "error_type": type(exc).__name__,
                                "attempt": attempt + 1,
                            }
                        ),
                        flush=True,
                    )
                    if attempt == 1:
                        raise
            row = {
                "id": ident,
                "generated_answer": answer,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "elapsed_seconds": round(time.monotonic() - began, 3),
            }
            with (output / arm / "answers.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "id": ident,
                        "completed": len(finished) + 1,
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            finished[ident] = row
    report(output)


def add_scope(output: Path) -> None:
    """独立第三组：只追加真实候选范围描述，不修改原有两组输入/输出。"""
    manifest = json.loads((output / "manifest.json").read_text())
    question_file = ROOT / benchmark_dir() / "corporatebench/data/kb_qa/zenith_questions.json"
    questions = json.loads(question_file.read_text())["questions"]
    corpus = Corpus(ROOT / benchmark_dir() / "corporatebench/data/kb/zenith.kb")
    first_date = min(header(body, "Date") for body in corpus.documents.values())
    source_rows = read_rows(output / "oracle_facts" / "input.jsonl")
    rows = []
    for ident, source in source_rows.items():
        number = int(ident.split("-")[-1])
        variables = questions[number]["variables"]
        if number == 0:
            scope = "all department entities recorded in this corpus"
        elif "company" in variables or not variables:
            scope = "all employee employment relationships for " + corpus.name(corpus.company)
        elif "meeting" in variables:
            scope = (
                "all occurrences and recorded participants of meeting series: "
                + variables["meeting"]
            )
        elif "project" in variables:
            scope = "all meeting occurrences associated with project: " + variables["project"]
        else:
            scope = "all meeting occurrences with topic: " + variables["meeting_topic"]
        context_header = (
            "Memory collection metadata\n"
            f"Collection scope: {scope}\n"
            "Coverage: complete for the stated scope in the source corpus snapshot.\n"
            "Date and frequency filtering have not been applied to these records.\n"
            "Source: full released KB snapshot and its linked source documents.\n"
        )
        if "employee employment" in scope:
            context_header += (
                f"Initial state timestamp: {first_date}\n"
                "Employment labeled pre-existing was already active at the initial state; "
                "the exact hiring date is not recorded.\n"
            )
        rows.append(
            dict(source, retrieved_context=context_header + "\n" + source["retrieved_context"])
        )
    arm = "oracle_scope_facts"
    write_rows(output / arm / "input.jsonl", rows)
    write_rows(
        output / arm / "prompts.jsonl",
        [
            {
                "id": row["id"],
                "prompt": manifest["answer_prompt"].format(
                    memories=row["retrieved_context"], question=row["question"]
                ),
            }
            for row in rows
        ],
    )
    manifest["arms"] = [*ARMS, arm]
    manifest["scope_arm_note"] = (
        "Hypothesis-driven third arm after original two runs: adds true exhaustive "
        "oracle candidate coverage metadata, with no totals, filtered answers or new "
        "answer instructions. The initial-state time anchor uses the earliest source "
        "document Date. Completeness is provided by the oracle snapshot and is not "
        "a claim about present Search capabilities."
    )
    manifest["scope_runner_sha256"] = digest(Path(__file__))
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"added_arm": arm, "n": len(rows)}), flush=True)


def add_sentences(output: Path) -> None:
    """第四组：同一独立事实与范围，转换为事实句，不加入聚合结果。"""
    manifest = json.loads((output / "manifest.json").read_text())
    source_rows = read_rows(output / "oracle_scope_facts" / "input.jsonl")
    rows = []
    for source in source_rows.values():
        metadata, records = source["retrieved_context"].split("\n\n", 1)
        fragments = []
        timestamp = re.search(r"Initial state timestamp: ([0-9-]+)", metadata)
        for record in records.split("\n\n"):
            fields: dict[str, list[str]] = defaultdict(list)
            for line in record.splitlines():
                key, value = line.split(": ", 1)
                fields[key].append(value)
            lines = []
            kind = fields["Entity type"][0]
            if kind == "Employee":
                name, employer = fields["Name"][0], fields["Employer"][0]
                lines.append(f"{name} is an employee of {employer}.")
                if "Employment relationship began" in fields:
                    date = fields["Employment relationship began"][0]
                    lines.append(f"{name} began employment at {employer} on {date}.")
                else:
                    assert timestamp is not None
                    lines.append(
                        f"{name} was already employed at {employer} at the initial "
                        f"snapshot on {timestamp[1]}; the exact hiring date is not recorded."
                    )
            elif kind == "Department":
                lines.append(f"{fields['Name'][0]} is a department of {fields['Company'][0]}.")
            else:
                title, date = fields["Meeting title"][0], fields["Occurrence date"][0]
                lines.append(f'A meeting occurrence titled "{title}" is recorded on {date}.')
                if "Topic" in fields:
                    lines.append(f'The topic of this occurrence was "{fields["Topic"][0]}".')
                if "Frequency" in fields:
                    lines.append(f"The meeting series has {fields['Frequency'][0]} frequency.")
                for project in fields.get("Associated project", []):
                    lines.append(f'This meeting occurrence is associated with project "{project}".')
                for participant in fields.get("Participant", []):
                    lines.append(f'{participant} attended the "{title}" occurrence on {date}.')
            lines.append("Source: " + fields["Source"][0])
            fragments.append("\n".join(lines))
        rows.append(dict(source, retrieved_context=metadata + "\n\n" + "\n\n".join(fragments)))
    arm = "oracle_sentence_facts"
    write_rows(output / arm / "input.jsonl", rows)
    write_rows(
        output / arm / "prompts.jsonl",
        [
            {
                "id": row["id"],
                "prompt": manifest["answer_prompt"].format(
                    memories=row["retrieved_context"], question=row["question"]
                ),
            }
            for row in rows
        ],
    )
    manifest["arms"].append(arm)
    manifest["sentence_arm_note"] = (
        "Changes key-value oracle facts into independent natural-language facts, "
        "with the same candidate records and complete-scope metadata. No computed "
        "counts or combined answer sets are present. Source KB attendee relations "
        "and occurrence dates remain oracle facts, not guaranteed prose extractions."
    )
    manifest["sentence_runner_sha256"] = digest(Path(__file__))
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"added_arm": arm, "n": len(rows)}), flush=True)


def add_matches(output: Path) -> None:
    """最后一组：oracle 按题目条件筛选独立事实；空集保留排除证据。"""
    manifest = json.loads((output / "manifest.json").read_text())
    source_rows = read_rows(output / "oracle_facts" / "input.jsonl")
    scope_rows = read_rows(output / "oracle_scope_facts" / "input.jsonl")
    questions = json.loads(
        (ROOT / benchmark_dir() / "corporatebench/data/kb_qa/zenith_questions.json").read_text()
    )["questions"]
    rows = []
    for ident, source in source_rows.items():
        variables = questions[int(ident.split("-")[-1])]["variables"]
        fragments = []
        initial = re.search(
            r"Initial state timestamp: ([0-9-]+)", scope_rows[ident]["retrieved_context"]
        )
        for record in source["retrieved_context"].split("\n\n"):
            fields: dict[str, list[str]] = defaultdict(list)
            for line in record.splitlines():
                key, value = line.split(": ", 1)
                fields[key].append(value)
            kind = fields["Entity type"][0]
            if kind == "Employee":
                start = fields.get("Employment relationship began", [""])[0]
                if "start_date_raw" in variables and (
                    not start
                    or not variables["start_date_raw"] <= start <= variables["end_date_raw"]
                ):
                    continue
                if "date_raw" in variables:
                    before = "before" in source["question"].lower()
                    if start:
                        if not (
                            start < variables["date_raw"]
                            if before
                            else start > variables["date_raw"]
                        ):
                            continue
                    elif not (before and initial and initial[1] < variables["date_raw"]):
                        continue
                name, employer = fields["Name"][0], fields["Employer"][0]
                if "date_raw" in variables or "start_date_raw" in variables:
                    if start:
                        fact = f"{name} began employment at {employer} on {start}."
                    else:
                        assert initial is not None
                        fact = f"{name} was already employed by {employer} on {initial[1]}."
                else:
                    fact = f"{name} is an employee of {employer}."
            elif kind == "Department":
                fact = (
                    f"The source corpus mentions the {fields['Name'][0]} department "
                    f"of {fields['Company'][0]}."
                )
            else:
                date = fields["Occurrence date"][0]
                if (
                    "start_date_raw" in variables
                    and not variables["start_date_raw"] <= date <= variables["end_date_raw"]
                ):
                    continue
                if (
                    "meeting_frequency" in variables
                    and variables["meeting_frequency"].casefold()
                    != fields["Frequency"][0].casefold()
                ):
                    continue
                title = fields["Meeting title"][0]
                if source["gold_answer"]["answer_type"] == "int":
                    fact = f'The "{title}" meeting occurred on {date}.'
                else:
                    keys = (
                        "Meeting title",
                        "Occurrence date",
                        "Topic",
                        "Frequency",
                        "Associated project",
                        "Participant",
                    )
                    fact = "\n".join(
                        f"{key}: {value}" for key in keys for value in fields.get(key, [])
                    )
            fragments.append(fact + "\nSource: " + fields["Source"][0])
        if fragments:
            context = (
                "Memory collection metadata\n"
                "Coverage: the source snapshot's independent records matching the "
                "requested entity, relation, time and frequency conditions.\n"
                "Source: full released KB snapshot and its linked source documents.\n\n"
                + "\n\n".join(fragments)
            )
            row = dict(source, retrieved_context=context, oracle_fragments=len(fragments))
        else:
            # 空上下文没有排除力，复用完整候选/范围组的原证据和输出，不给 [] / 0。
            row = scope_rows[ident]
        rows.append(row)
    arm = "oracle_matching_facts"
    write_rows(output / arm / "input.jsonl", rows)
    write_rows(
        output / arm / "prompts.jsonl",
        [
            {
                "id": row["id"],
                "prompt": manifest["answer_prompt"].format(
                    memories=row["retrieved_context"], question=row["question"]
                ),
            }
            for row in rows
        ],
    )
    scope_prompts = read_rows(output / "oracle_scope_facts" / "prompts.jsonl")
    scope_answers = read_rows(output / "oracle_scope_facts" / "answers.jsonl")
    reuse = []
    for row in rows:
        ident = row["id"]
        prompt = manifest["answer_prompt"].format(
            memories=row["retrieved_context"], question=row["question"]
        )
        if prompt == scope_prompts[ident]["prompt"]:
            reuse.append(dict(scope_answers[ident], reused_from="oracle_scope_facts"))
    write_rows(output / arm / "answers.jsonl", reuse)
    manifest["arms"].append(arm)
    manifest["matching_arm_note"] = (
        "Final hypothesis-driven arm: oracle filters independent records by question "
        "conditions, strips unused fields, and uses explicit occurrence facts for counts. "
        "No gold answer values, precomputed counts or combined answer lists are used. "
        "Empty candidate results retain the original complete-scope exclusion evidence. "
        "The occurrence verb is an oracle interpretation of KB event records, not "
        "proof that agenda/calendar prose confirms the event actually happened."
    )
    manifest["matching_runner_sha256"] = digest(Path(__file__))
    manifest["matching_reused_ids"] = [row["id"] for row in reuse]
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"added_arm": arm, "n": len(rows), "reused": len(reuse)}), flush=True)


def report(output: Path) -> None:
    provenance = {row["id"]: row for row in json.loads((output / "provenance.json").read_text())}
    inputs = read_rows(output / ARMS[0] / "input.jsonl")
    active_arms = json.loads((output / "manifest.json").read_text()).get("arms", list(ARMS))
    sources = {
        "baseline": read_rows(output / "baseline-answers.jsonl"),
        **{arm: read_rows(output / arm / "answers.jsonl") for arm in active_arms},
    }
    comparison = []
    summaries = {}
    for arm, answers in sources.items():
        labels = []
        for ident, answer in answers.items():
            value, reason = score_corporatebench(
                answer["generated_answer"], inputs[ident]["gold_answer"]
            )
            labels.append(
                {
                    "id": ident,
                    "is_correct": value == 1,
                    "partial": value,
                    "label": "CORRECT" if value == 1 else "WRONG",
                    "judge_response": reason,
                }
            )
        write_rows(output / arm / "labels.jsonl", labels)
        summaries[arm] = {}
        for kind in ("int", "List[str]", "all"):
            group = [
                r
                for r in labels
                if kind == "all" or inputs[r["id"]]["gold_answer"]["answer_type"] == kind
            ]
            summaries[arm][kind] = {
                "n": len(group),
                "correct": sum(r["is_correct"] for r in group),
                "qa_mean": sum(r["partial"] for r in group) / len(group) if group else None,
            }
    for ident, item in inputs.items():
        row = {
            "id": ident,
            "question": item["question"],
            "gold": item["gold_answer"],
            "context_tokens": provenance[ident]["context_tokens"],
        }
        for arm, answers in sources.items():
            if ident in answers:
                generated = answers[ident]["generated_answer"]
                value, reason = score_corporatebench(generated, item["gold_answer"])
                row[arm] = {"answer": generated, "score": value, "reason": reason}
        comparison.append(row)
    write_json(output / "comparison.json", comparison)
    write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("prepare", "scope", "sentences", "matches", "run", "report")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval/reports/runs/cb-oracle-context-20261004"
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "scope": add_scope,
        "sentences": add_sentences,
        "matches": add_matches,
        "run": run,
        "report": report,
    }[args.command](args.output)


if __name__ == "__main__":
    main()
