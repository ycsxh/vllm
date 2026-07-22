# Issue tracker: GitHub

Issues and PRDs for this fork live in GitHub repository `ycsxh/vllm`. Use the
`gh` CLI and pass `--repo ycsxh/vllm` explicitly for every state-changing
operation.

`vllm-project/vllm` is a read-only reference. Never create, edit, comment on,
label, assign, close, or otherwise mutate an upstream issue or pull request.

## Conventions

- **Create an issue**:
  `gh issue create --repo ycsxh/vllm --title "..." --body "..."`.
- **Read an issue**:
  `gh issue view <number> --repo ycsxh/vllm --comments`.
- **List issues**:
  `gh issue list --repo ycsxh/vllm --state open --json number,title,body,labels,comments`.
- **Comment on an issue**:
  `gh issue comment <number> --repo ycsxh/vllm --body "..."`.
- **Apply or remove labels**:
  `gh issue edit <number> --repo ycsxh/vllm --add-label "..."` or
  `--remove-label "..."`.
- **Assign an issue**:
  `gh issue edit <number> --repo ycsxh/vllm --add-assignee @me`.
- **Close an issue**:
  `gh issue close <number> --repo ycsxh/vllm --comment "..."`.

Read-only upstream duplicate-work checks must also name the repository
explicitly, for example:

```bash
gh issue view <number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<keywords>"
```

## Pull requests as a triage surface

**PRs as a request surface: no.** Pull requests do not enter the issue-triage
queue.

## Skill meanings

- When a skill says **publish to the issue tracker**, create an issue in
  `ycsxh/vllm`.
- When a skill says **fetch the relevant ticket**, read the named issue and its
  comments from `ycsxh/vllm`.
- Tickets produced by `/to-tickets` are already agent-ready and do not pass
  through `/triage`.

## Dependencies and frontier

GitHub native issue dependencies are preferred only through a state-changing
interface that can explicitly scope the operation to `ycsxh/vllm` under the
repository authority policy. The installed `gh api` command has no `--repo`
flag, so do not use it for dependency mutations.

Until a compliant repo-scoped interface is available, add
`Blocked by: #<number>` at the top of the child issue body. A ticket is
unblocked only after every blocker closes. When selecting work, choose the
first open, unassigned ticket whose blockers are all closed.

## Wayfinding

When `/wayfinder` is used, its map and child tickets also live in `ycsxh/vllm`.
Use a `wayfinder:map` label for the map and `wayfinder:<type>` labels for child
tickets. Prefer GitHub sub-issues; if unavailable, use a task list in the map
and put `Part of #<map-number>` at the top of each child.
