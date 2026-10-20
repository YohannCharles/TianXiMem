# eval/baselines

This directory contains external reference implementations. Vendored code stays unchanged and is never imported by service code.

## Vendored Repositories

```text
refind/             AML-compatible external baseline
memmachine/         Source reading reference
invmem-candidate/   Unlicensed candidate mapping; reading only
```

Each repository is cloned from its upstream source and ignored by Git. Capture the upstream URL and pinned revision in the run or source record before using a clone.

## Evaluation Boundary

- Drive each baseline as an independent HTTP service.
- Do not import vendor code into the TianXiMem process.
- Do not modify vendor source to improve a baseline result.
- Use an isolated database and collection.
- Use the same answer/judge contract as the TianXiMem side of a comparison.
- Treat vendor-reported scores as external claims, not local results.

## Reference Use

Reference implementations may inform design review, but service code must remain an independent implementation. Code with an unclear license must not be copied, distributed, or treated as a dependency.

## Setup

The external repositories are not installed by the project. Follow their pinned requirements in a separate environment and keep generated databases outside the vendor directory.
