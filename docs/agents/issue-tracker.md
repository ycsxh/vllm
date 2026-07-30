# Issue tracker: GitHub personal fork

Issues and PRDs for this working fork live in GitHub Issues under
`ycsxh/vllm`. Use the `gh` CLI for all operations.

Treat `vllm-project/vllm` as read-only. Never create, edit, comment on,
label, assign, or close upstream issues or pull requests.

## Conventions

- Create: `gh issue create --repo ycsxh/vllm ...`
- Read: `gh issue view <number> --repo ycsxh/vllm --comments`
- List: `gh issue list --repo ycsxh/vllm ...`
- Comment: `gh issue comment <number> --repo ycsxh/vllm ...`
- Label: `gh issue edit <number> --repo ycsxh/vllm --add-label "..."`
- Close: `gh issue close <number> --repo ycsxh/vllm ...`

All state-changing commands must explicitly specify `--repo ycsxh/vllm`.

## Pull requests as a triage surface

PRs as a request surface: no.

## Skill operations

When a skill says "publish to the issue tracker", create an issue in
`ycsxh/vllm`. When it says "fetch the relevant ticket", read the corresponding
issue and comments from `ycsxh/vllm`.

Any fork-only issue or PR metadata must be kept out of patches proposed to
upstream.
