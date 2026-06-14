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
