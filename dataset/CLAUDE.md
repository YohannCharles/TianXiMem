# dataset

This directory stores downloaded evaluation material. Program code belongs in `eval/`.

```text
dataset/
├── <dataset>/     Downloaded source data
├── .upstream/     Upstream pipeline and judge source
├── .legacy/       Archived non-runnable material
└── .tmp/          Atomic download and patch workspace
```

Rules:

- Only this file is tracked.
- Downloaded data, upstream source, temporary files, and captures stay out of Git.
- The manifest in `eval/datasets/manifest.py` is the only source of URLs, revisions, and hashes.
- Preserve upstream bytes unless the manifest declares a local patch.
- Never use `.legacy/` as a live evaluation dataset.
- Do not place dataset-specific file names or network access in the service.
- Set the shared data root with `TIANXIMEM_BENCHMARK_DIR`.

Preparation, validation, supported input contracts, and licensing are documented in [`../docs/benchmark-data.md`](../docs/benchmark-data.md).
