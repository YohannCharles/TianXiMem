"""Two synthetic conflicting-history checks through the real PersonaMem MCQ answer/scorer."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from eval.harness.personamem_pipeline import (
    MEMORY_INPUT_VERSION,
    build_mcq_messages,
    cmd_answer,
    cmd_evaluate,
)


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for index, (preferred, distractor) in enumerate((("yoga", "swimming"), ("swimming", "yoga"))):
        rows.append(
            {
                "id": f"memory-source-{index}",
                "persona_id": 7,
                "question": "Which morning activity matches my explicitly stated preference?",
                "correct_answer": preferred.capitalize(),
                "incorrect_answers": [distractor.capitalize(), "Running"],
                "retrieved_context": f"User: My preferred morning activity is {preferred}.",
                "chat_history": [
                    {"role": "user", "content": f"I prefer {distractor}. HIDDEN-HISTORY"}
                ],
            }
        )
    for row in rows:
        messages, _, _ = build_mcq_messages(row)
        assert row["retrieved_context"] in str(messages)
        assert "HIDDEN-HISTORY" not in str(messages)
    source, answers, labels = (
        root / (stage + ".jsonl") for stage in ("input", "answers", "labels")
    )
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = SimpleNamespace(input=str(source), output=str(answers))
    cmd_answer(args)
    cmd_evaluate(SimpleNamespace(input=str(source), answers=str(answers), output=str(labels)))
    judged = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    answer_rows = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
    result = {
        "at": datetime.now(UTC).isoformat(),
        "scope": "Synthetic adapter smoke; not a dataset accuracy estimate or Add/Search benchmark",
        "memory_input_version": MEMORY_INPUT_VERSION,
        "model": answer_rows[0]["model"],
        "n": len(judged),
        "correct": sum(r["is_correct"] for r in judged),
        "raw_history_excluded": True,
        "adapter_sha256": hashlib.sha256(
            Path("eval/harness/personamem_pipeline.py").read_bytes()
        ).hexdigest(),
    }
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
