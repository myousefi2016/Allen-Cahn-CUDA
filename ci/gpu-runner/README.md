# Self-hosted GPU runner

The `Build & Test (CUDA)` job in `.github/workflows/ci.yml` runs on
`[self-hosted, gpu, linux]`: a host with an NVIDIA GPU, Docker and the NVIDIA
Container Toolkit (the job runs in `nvidia/cuda:*-devel` with `--gpus all`).

## Who can run jobs on it

The repository is public, so the question is what stops anyone else from
executing code on the runner host. GitHub's own guidance is that its
fork-approval settings are not enough for self-hosted runners, and any check
in the workflow file is not enough either, because a fork pull request runs
the workflow file from the fork.

The guarantee is enforced on the runner host, before any job step or job
container starts, by `job-started-guard.sh` (installed as the runner's
`ACTIONS_RUNNER_HOOK_JOB_STARTED`; a non-zero exit fails the job without
running it). A job runs only if **all** of these hold:

| Check | Source (not settable by a workflow) |
|---|---|
| repository is `myousefi2016/Allen-Cahn-CUDA` (id 155445643) | `GITHUB_REPOSITORY` and the event payload |
| actor and triggering actor are `myousefi2016` | `GITHUB_ACTOR`, `GITHUB_TRIGGERING_ACTOR` |
| event sender is user id 22246708 | event payload (`GITHUB_EVENT_PATH`) |
| event is `push` or `workflow_dispatch`, or a `pull_request` whose head is this repository and whose author is user 22246708 | event payload |

Workflows cannot overwrite `GITHUB_*` variables, and the event payload is
written by the runner from the job GitHub assigned. The guard's constants
are hard-coded (never read from the environment), and it runs with
`python3 -I -S` from absolute paths. The guard file is root-owned outside the
runner directory, and the runner's `.env` that registers it is `chattr +i`,
so no job can remove or edit it. Refused: fork pull requests (even when the
owner pushes to them or re-runs them), pull requests by anyone else, bots
such as Dependabot, re-runs started by anyone else, other events
(`pull_request_target`, `issue_comment`, `workflow_run`, ...), and anything
unparsable. `tests/test_gpu_runner_guard.py` covers each case and runs in CI.

What remains trusted: the owner's GitHub account and tokens, GitHub itself,
and root on the runner host.

## Install

On the GPU host, as root, with this directory checked out:

```bash
sudo ./install.sh --reg-token-file /root/regtoken   # token from Settings > Actions > Runners > New
# or
sudo ./install.sh --pat-file /root/pat               # PAT with repo admin rights
```

The secret file is read once and deleted. Guard decisions are logged to the
journal: `journalctl -t gh-runner-guard`.
