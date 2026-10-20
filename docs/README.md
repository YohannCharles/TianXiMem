# Documentation Index

The system specification is [`AML Agentic Memory 增强框架 PRD.md`](../AML%20Agentic%20Memory%20增强框架%20PRD.md). These documents provide current operational detail and navigation; they do not duplicate the full specification.

| Document | Purpose |
| --- | --- |
| [architecture.md](architecture.md) | Runtime flow, ownership boundaries, dependency direction, and invariants |
| [contract.md](contract.md) | Add/Search HTTP shapes and compliance checklist |
| [api-usage.md](api-usage.md) | Manual API examples and error handling |
| [config-reference.md](config-reference.md) | Keys, defaults, ownership, validation, and switch semantics |
| [benchmark-data.md](benchmark-data.md) | Evaluation data preparation and archive layout |
| [fixed-selection-suite.md](fixed-selection-suite.md) | Shared fixed-question evaluation workflow |
| [capture-evaluation-pipeline.md](capture-evaluation-pipeline.md) | Official request-capture replay workflow |
| [common-facts-gold-evaluation.md](common-facts-gold-evaluation.md) | Grounded-fact extraction evaluation |
| [active-spec/config-defaults.md](active-spec/config-defaults.md) | Active product profile values |
| [active-spec/development.md](active-spec/development.md) | Cross-platform command entry points |

## Documentation Rules

- Describe the current design only.
- Do not add dates, chronological change logs, superseded designs, or decision-number narratives.
- Keep one owner for each rule. Link to that owner instead of copying the full text.
- Generated measurements belong under `eval/reports/`; they are evidence, not specification.
- Update the owning document and its tests in the same change.
