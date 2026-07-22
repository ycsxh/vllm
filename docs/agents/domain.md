# Domain docs

This repository uses a single-context domain-document layout.

## Before exploring

Read these sources when they exist and are relevant:

- `CONTEXT.md` at the repository root;
- ADRs under `docs/adr/` that touch the area being changed.

If either source does not exist, proceed silently. Do not create empty domain
documents merely to satisfy this layout. `/domain-modeling` creates or updates
them when terminology or a durable architectural decision is actually
resolved.

## Layout

```text
/
├── CONTEXT.md
└── docs/
    └── adr/
        └── NNNN-short-decision-name.md
```

## Vocabulary

Use domain terms as defined in `CONTEXT.md` in issue titles, specifications,
tests, and implementation. Do not substitute synonyms that the glossary
explicitly avoids.

If a needed term is absent, first decide whether the codebase already uses a
more appropriate term. Record a genuine domain-language gap for
`/domain-modeling` rather than inventing competing vocabulary locally.

## ADR conflicts

If proposed work contradicts an applicable ADR, state the conflict explicitly
and identify the ADR. Do not silently override a recorded decision.
