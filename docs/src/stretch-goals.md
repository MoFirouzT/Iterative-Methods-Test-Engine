# Stretch Goals

Capabilities the architecture is deliberately *designed to admit* but does not
currently ship. Each is reachable without breaking an existing abstraction — this
page records *how*, so the design intent is not lost and a future contributor has a
concrete starting point rather than a blank page.

## Quasi-Newton Hessian approximations (BFGS, SR1, L-BFGS)

Stateful Hessian approximations plug into the existing [`Hessian`](modules/problem-interface.md)
hierarchy without changes to any consumer: each is a `Hessian` subtype carrying an
internal `update!(H, s, y)` method that the algorithm calls after each step. L-BFGS,
in particular, never defines `materialize` — it stores the `(s, y)` history and
computes `apply` via the two-loop recursion. Methods consuming the Hessian are
unaware of the internal mechanism; they only ever call `apply` (and, when available,
`materialize` / `diagonal`).

## Other designed-for extensions

Each of these is reachable without breaking an existing abstraction; the detailed
analysis lives next to the code it touches, linked here so the design intent is
indexed in one place.

- **Barzilai–Borwein nonmonotone (GLL) safeguard** — the principled replacement for
  BB's `[α_min, α_max]` clamp, permitting BB's load-bearing excursions while catching
  genuine divergence. Full derivation and a ~30-line implementation sketch in
  [`step_sizes.md` §5.2](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/algorithms/components/step_sizes.md).
- **JLD2 struct-of-arrays schema migration** — a column-major on-disk layout that is
  expected to shrink `result.jld2` by ~5–10× on densely-typed payloads. Measured
  rationale and the proposed schema in
  [`persistence.md`](modules/persistence.md).

These are *architectural* stretch goals. The separate backlog of **not-yet-built
demonstrator experiments** (SGD / logistic regression, constrained / projection
problems, a live `FileProblem` path, and smaller swept-variant figures) is tracked in
[experiments/README.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/experiments/README.md#planned--not-yet-built),
since those are blocked on new content rather than on engine design.
