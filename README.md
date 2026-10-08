# pr-triage-bench

Regression suite for PyTorch's [Auto PR Triage](https://github.com/pytorch/pytorch/blob/main/.github/workflows/auto-pr-triage.yml)
owner routing. Dashboard: https://soulitzer.github.io/pr-triage-bench/

Each case is a PR whose inputs were snapshotted, with the owner categories a
person says the bot should or should not assign. A run replays every snapshot
through the production pipeline at one pytorch commit, several times per case.
Cases are chosen to cover each part of a category's description (what it owns
and what it excludes), so the suite is not a random sample.

## Cases

`cases/<pr>/case.json` records:

- `expected`: owner category -> `assign` or `not_assign`
- `tests`: what this case tests, for example which clause of the description
- `source`: `bot-mislabeled`, `reviewed` (someone on the category's roster
  reviewed the PR), or `hand-picked`
- `labeled_by`, `labeled_at`, `reason`
- `snapshot`: the PR head it was taken from, and the pytorch commit whose
  pipeline took it
- context: the bot's runs on the PR with the categories each assigned, people's
  reviewer changes after the bot first acted, and roster reviews

`cases/<pr>/snapshot/` holds only the PR side of the worker's input: the
pipeline's intake result and the PR's changed files with patches. Runs never
read the live PR, so later pushes, reviewer changes, or labels do not change a
case. Intake facts are recorded as an open, unhandled PR so closed or triaged
PRs can be cases; the worker does not see these facts.

The config side (`worker.md`, `CODEOWNERS`, and `.github/auto-pr-triage/*.json`)
is not part of a case. Each run takes it, with the pipeline, from one pytorch
commit, and the dashboard links every file a run used at that commit.

Owner categories that get renamed are mapped in `cases/owner_aliases.json`.

## Adding cases

1. **Flag on GitHub.** When the bot routes a PR to the wrong category, add the
   `bot-mislabeled` label and fix the reviewers.
2. **Draft.** `python -m bench.harvest [options]` snapshots each PR with the
   pipeline on pytorch `main` and drafts `cases/pending/<pr>/`:
   - every new `bot-mislabeled` PR, with the categories the bot assigned as
     `not_assign` (skip with `--skip-mislabeled`);
   - with `--reviewed-owner <category>`, the most recently updated PRs a
     reviewer on that category's roster reviewed (up to `--reviewed-limit`,
     default 10, per reviewer), as `assign`. The bot need not have routed them,
     and reviewers also review work outside their category, so check each one;
   - with `--add <pr> --expect <category>=<assign|not_assign>`, a hand-picked PR.
3. **Confirm.** Fill in `tests` and `reason`, fix `expected`, and move the
   directory to `cases/`. Move drafts you don't want to `cases/skipped/` so
   harvest does not draft them again. Only directories in `cases/` count.

Prefer cases that cover a clause no other case covers, or that sit near a
boundary of the description, over more of the same kind.

## Running

Requires `gh` (authenticated), the `claude` CLI, and Python 3.10+.

A run is fully described by one pytorch commit: the pipeline and its config
both come from it. To try a change to a description, the worker prompt, or the
pipeline, push it as a pytorch PR (for example with ghstack) and run on the PR.

```
# Current behavior, at a main commit:
python -m bench.run --name main-<sha> --pytorch-sha <sha>

# A proposal, at the PR's head commit:
python -m bench.run --name pr-<number> --pr <number>

python -m bench.score runs/pr-<number>
```

`bench.run` fetches `scripts/auto_pr_triage`, `CODEOWNERS`, and
`.github/auto-pr-triage/*.json` at that commit. For each case it builds the
worker input from the snapshot with the pipeline's `build_ownership_input.py`,
then runs the tool-less worker (same flags as
`.github/actions/auto-pr-triage/action.yml`, `claude-sonnet-5` at effort
`low`, 5 reps by default) and validation. It never writes to GitHub. Planning
(triage vs. routed untriaged) is not run, since it depends on live reviewer
state. A PR run records the PR number and the head commit it used; if the PR
is updated, run it again under a new name.

If the pipeline changes the format of the intake result or file list, old
snapshots may need to be retaken.

## Scores

Per case and expected category, a run passes when it assigns a category the
case expects assigned, or leaves out one expected absent. Per category:

- **recall**: share of runs that assign the category on `assign` cases;
- **false positive rate**: share of runs that assign it on `not_assign` cases.

Neither depends on how many cases of each kind the suite has. Precision would,
and it can be computed from the two, so it is not reported. Because cases are
chosen to cover the description rather than sampled, use these to compare
configs, not as production rates.

Runs whose worker output fails validation are counted separately.

## Dashboard

```
python -m bench.publish runs/main runs/proposal
```

copies each run's sanitized `results.jsonl` and `meta.json` into
`results/<name>/` (committed) and rebuilds `docs/index.html`, which GitHub
Pages serves from `main`. Two pickers choose runs A and B: the category table
and its expandable samples show those two runs, and the comparison below shows
what differs (with a GitHub diff link between the two commits and a word diff
of changed descriptions), each metric's change, and samples whose result
changed. Run with no arguments to rebuild after changing cases.
