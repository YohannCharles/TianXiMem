# ruff: noqa: E501
"""Plan 模式的两条提示词（**独立于原版 CoQ**，2026-10-12 新增）。

| 常量 | 什么时候用 | 回什么 |
| --- | --- | --- |
| [`PLAN_DECOMPOSE_PROMPT`][PLAN_DECOMPOSE_PROMPT] | **拆解**：路由/强制多跳之后、执行循环之前，**一次** | 1～4 步的有序计划 JSON |
| [`PLAN_STEP_PROMPT`][PLAN_STEP_PROMPT] | **每一轮**：本轮检索**已经跑完**之后，一次 | **只回** `status` / `answer` / `evidence`（⛔ 不回步骤编号、不回下一查询） |

## 三条边界（写在这里，因为越界不会报错）

1. ⛔ **本文件与 [`coq_prompt.py`](./coq_prompt.py) 没有任何共享文本**。原版 CoQ 提示词
   是**冻结的**（"不把新提示词塞进冻结的原 CoQ 提示词"）——两者是**两个模式的两份**，
   删掉其中任何一份都不影响另一份。
2. ⛔ **模板里的花括号是内容，不是格式**：`{{s1}}` 是**计划自己的占位语法**
   （由拆解那一步产出、由**程序**填），JSON 示例里的 `{` `}` 同理。
   ⇒ 填充**必须**用 `plan.py` 里那个"只认声明过的槽位、一遍扫描"的替换器，
   ⛔ **绝不用 `str.format` / `%` / 连续 `str.replace`**：前者会被 JSON 示例炸掉
   或者替换出没人预期的输入，后者会让**模板里的 `{{s1}}` 与用户文本里的字面槽位**
   被二次替换，且**没有任何报错**。
3. ⚠ **两条提示词的输出域不同**（2026-10-13 起）：拆解回 `{"steps": [...]}`；
   逐轮回 `{"status", "answer", "evidence"}`——⛔ **不回步骤编号、不回下一查询**。
   "现在是第几步、下一步问什么、什么时候算跑完"**全部由程序决定**
   （见 `plan.py` 的 `search`），模型只负责"答这一步"。
4. ⚠ 长行是**有意**的：提示词换个行就不再逐字相同。`ruff` 的 `E501` 是给显示宽度用的。
"""

from __future__ import annotations

from typing import Final

#: 拆解提示词的版本。⚠ **它是一条断言**（同 `router.prompt_version` / `agentic.prompt_version`）：
#: 与 `agentic.plan_decompose_prompt_version` 不符 ⇒ **构造期响亮失败**。
PLAN_DECOMPOSE_PROMPT_VERSION: Final[str] = "plan-decompose-v2"

#: 逐轮提示词的版本。**与上面那条分开**：两条提示词各自演进，一个版本号管不住两条。
PLAN_STEP_PROMPT_VERSION: Final[str] = "plan-step-v2"

PLAN_DECOMPOSE_PROMPT: Final[str] = """You create an ordered retrieval plan for a question.
You plan the searches; you do not answer the question.

Original question:
{original_question}

Create one to four sub-questions that preserve the complete
reasoning path required by the original question.

Planning rules:

1. Identify the final answer target before constructing the plan.
   Preserve what the question asks for: an entity, place, date,
   number, property, explanation, or other requested value.
   The final step must ask for this same target.

2. Start from an explicitly named entity in the original question.
   Resolve unknown intermediate entities in dependency order.

3. Each step should resolve one unknown relation or requested value.
   If a lookup depends on an unidentified person, organization,
   location, object, or event, resolve that referent in an earlier step.
   Do not leave a chain of unidentified referents inside one
   sub-question when it can be split within the four-step limit.

4. Preserve every necessary relation in the original question.
   Preserve relation direction and all identifying qualifiers:
   dates, locations, exclusions, comparisons, superlatives,
   event conditions, and distinctions between related entities.
   Use the specific relationship stated in the question rather
   than replacing it with a vague association.

5. Do not guess intermediate answers from memory.
   Do not replace an unresolved referent with an assumed entity.
   Do not introduce facts or conditions absent from the question.

6. Use placeholders for answers established by earlier steps:
   {{s1}} means the answer to s1, {{s2}} means the answer to s2,
   and so on.
   Preserve complete entity names and titles, including punctuation.

7. Step IDs must be s1, s2, and so on, in execution order.
   A step may depend only on earlier steps.
   Its depends_on list must contain exactly the earlier step IDs
   referenced by placeholders in its query.
   The first step must have no placeholders and no dependencies.

8. Choose the number of steps from the required relationships.
   Do not prefer a shorter plan at the expense of a missing relation.
   Do not add steps solely to reach four.
   A direct lookup needs only one step.
   If the four-step limit requires combining adjacent relationships,
   preserve those relationships explicitly in the combined question.

9. Before returning the plan, check internally:
   - Does every necessary relationship appear in the plan?
   - Is each unknown referent resolved before it is used?
   - Are relation direction and qualifiers preserved?
   - Does the final step request the original answer target?
   Revise the plan if any check fails.

Correct examples:

Example 1 — direct lookup

Question:
On what date did the Aurora Festival begin in 2021?

Output:
{
  "steps": [
    {
      "id": "s1",
      "query": "On what date did the Aurora Festival begin in 2021?",
      "depends_on": []
    }
  ]
}

Example 2 — three dependent lookups

Question:
In what year was the university where the architect who designed
Aurora Museum earned a doctorate founded?

Output:
{
  "steps": [
    {
      "id": "s1",
      "query": "Who was the architect who designed Aurora Museum?",
      "depends_on": []
    },
    {
      "id": "s2",
      "query": "At which university did {{s1}} earn a doctorate?",
      "depends_on": ["s1"]
    },
    {
      "id": "s3",
      "query": "In what year was {{s2}} founded?",
      "depends_on": ["s2"]
    }
  ]
}

Example 3 — four dependent lookups

Question:
Which country contains the birthplace of the architect who designed
the building that houses Aurora Museum?

Output:
{
  "steps": [
    {
      "id": "s1",
      "query": "Which building houses Aurora Museum?",
      "depends_on": []
    },
    {
      "id": "s2",
      "query": "Who was the architect who designed {{s1}}?",
      "depends_on": ["s1"]
    },
    {
      "id": "s3",
      "query": "Where was {{s2}} born?",
      "depends_on": ["s2"]
    },
    {
      "id": "s4",
      "query": "Which country contains {{s3}}?",
      "depends_on": ["s3"]
    }
  ]
}

The examples illustrate planning structure only.
Use only the actual original question to construct its plan.

Return only a JSON object with the "steps" array.
Each step must contain exactly "id", "query", and "depends_on".
Do not return answers, explanations, or additional fields.
"""

PLAN_STEP_PROMPT: Final[str] = """You process the results of a search that has already completed.
Answer only the current step using the supplied evidence.

The external program controls step order, stores answers, fills
placeholders in later queries, and determines when execution ends.
Your responsibility is to produce a supported answer suitable
for binding to the current step's placeholder.

Original question:
{original_question}

Fixed retrieval plan:
{retrieval_plan}

Previously completed questions and answers:
{completed_history}

Step to answer now:
{current_step}

Actual query used for the completed search:
{executed_query}

Documents returned by this search, indexed from zero:
{current_documents}

Rules:

1. Answer the current step exactly as written.
   Preserve the requested entity type, relation direction,
   and all necessary qualifiers.
   Do not answer another step or substitute a related entity.
   Use the original question and fixed plan to clarify context,
   not to change the current step's target.

2. Use only the current documents and previously completed history.
   Do not use outside knowledge.
   History identifies prior bindings; it is not independent evidence
   that those bindings are correct.

3. Before marking the step "answered", verify that the cited text
   supports the specific answer to the current question.
   A passage mentioning the entity or topic is insufficient by itself.
   The evidence must establish the requested relationship or value.
   You may combine multiple current documents when necessary;
   cite each document needed for that inference.

4. Return the smallest complete answer suitable for direct substitution:
   - Entity: its name, with necessary distinguishing qualifiers.
   - Place: its name, with necessary geographic qualifiers.
   - Date: the requested date at the supported level of precision.
   - Number: the value and a unit when needed.
   - Property or explanation: a concise phrase answering the question.

   Do not add introductory wording, repeat the question, explain
   your reasoning, or wrap an entity answer in a full sentence.
   Preserve qualifiers required for correct identification.

5. Cite current document indices and copy supporting excerpts exactly.
   Each document_index must refer to a current document.
   Each quote must be a contiguous, verbatim excerpt from that document.
   Do not paraphrase quotes, insert ellipses, or join separate passages.
   The cited excerpts must support the answer, not merely contain
   matching words.

6. If the evidence does not establish the answer, return "unresolved".
   If contradictory evidence prevents a supported resolution,
   return "conflict".
   Do not guess a value to keep the plan moving.

7. Do not silently change a previously bound answer.
   If current evidence contradicts a dependency required for this step,
   return "conflict".

8. For an unresolved or conflicting step:
   - answer must be null.
   - evidence must be an empty array.

9. Answer only the step whose search has already completed.
   Do not generate a next query, select another step, fill future
   placeholders, or decide whether the plan is complete.
   Do not assess overall evidence sufficiency or return confidence.
   These execution decisions belong to the external program.

Correct examples:

Example 1 — entity answer suitable for binding

Current step:
s1: Who was the architect who designed Aurora Museum?

Actual query:
Who was the architect who designed Aurora Museum?

Current documents:
[0] Aurora Museum was designed by Elena Marquez and opened in 1998.

Output:
{
  "status": "answered",
  "answer": "Elena Marquez",
  "evidence": [
    {
      "document_index": 0,
      "quote": "Aurora Museum was designed by Elena Marquez"
    }
  ]
}

Example 2 — qualified place answer

Current step:
s2: Where was {{s1}} born?

Previously completed answer:
s1: Elena Marquez

Actual query:
Where was Elena Marquez born?

Current documents:
[0] Elena Marquez was born in Santa Rosa, La Pampa, Argentina.

Output:
{
  "status": "answered",
  "answer": "Santa Rosa, La Pampa, Argentina",
  "evidence": [
    {
      "document_index": 0,
      "quote": "Elena Marquez was born in Santa Rosa, La Pampa, Argentina."
    }
  ]
}

Example 3 — supported year answer

Current step:
s3: In what year was {{s2}} founded?

Previously completed answer:
s2: Northbridge University

Actual query:
In what year was Northbridge University founded?

Current documents:
[0] Northbridge University was founded in 1847.

Output:
{
  "status": "answered",
  "answer": "1847",
  "evidence": [
    {
      "document_index": 0,
      "quote": "Northbridge University was founded in 1847."
    }
  ]
}

The examples illustrate output structure only.
They are not evidence for the actual current step.

Return only JSON with exactly these fields:
"status", "answer", and "evidence".

Use status "answered", "unresolved", or "conflict".

Each evidence item must contain exactly
"document_index" and "quote".
"""

#: 拆解提示词的槽位。**恰好一个**——拆解只看原问题（历史还不存在）。
PLAN_DECOMPOSE_SLOTS: Final[tuple[str, ...]] = ("{original_question}",)

#: 逐轮提示词的槽位。**只有六个**——`{next_step}` 从这一版起**没有了**（2026-10-13）：
#: 下一查询由**程序**按固定模板绑定，⛔ 既不让模型看模板、也不再问它要一句 `next_query`
#: （那是 v1 的形状）。⚠ 槽位少一个不会报错，只会让提示词里**永远缺一段上下文**
#: ——所以 `render_plan_step_prompt` 的形参表必须与它逐字对上。
PLAN_STEP_SLOTS: Final[tuple[str, ...]] = (
    "{original_question}",
    "{retrieval_plan}",
    "{completed_history}",
    "{current_step}",
    "{executed_query}",
    "{current_documents}",
)

__all__ = [
    "PLAN_DECOMPOSE_PROMPT",
    "PLAN_DECOMPOSE_PROMPT_VERSION",
    "PLAN_DECOMPOSE_SLOTS",
    "PLAN_STEP_PROMPT",
    "PLAN_STEP_PROMPT_VERSION",
    "PLAN_STEP_SLOTS",
]
