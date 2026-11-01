#!/usr/bin/env bash
# ============================================================================
# Pre-job guard for the self-hosted GPU runner (ACTIONS_RUNNER_HOOK_JOB_STARTED).
#
# The runner executes this script on the host after GitHub assigns it a job
# and before any step (or job container) starts; a non-zero exit fails the
# job without running it. It is installed root-owned outside the runner
# directory, so workflow code cannot change it, and it decides from values a
# workflow cannot set: the GITHUB_* variables (workflows cannot overwrite
# them) and the event payload the runner received from GitHub
# (GITHUB_EVENT_PATH).
#
# A job is allowed only if ALL hold:
#   * the repository is ALLOWED_REPO_ID / ALLOWED_REPO;
#   * the actor, the triggering actor (re-runs) and the event sender are the
#     single allowed account (checked by numeric user id, not only login);
#   * the event is push or workflow_dispatch, or a pull_request whose head
#     branch lives in this repository (never a fork) and whose author is the
#     allowed account.
# Everything else, including GitHub Apps/bots, fork pull requests, other
# users' pull requests, re-runs of such runs, and unknown events, is refused.
# Decisions go to the job log and to the system journal (tag gh-runner-guard,
# `journalctl -t gh-runner-guard`), which the runner user cannot rewrite.
# ============================================================================
set -u

# Constants on purpose, never read from the environment: whether a job's
# environment can reach this hook is not something to rely on.
readonly ALLOWED_REPO="myousefi2016/Allen-Cahn-CUDA"
readonly ALLOWED_REPO_ID=155445643   # GET /repos/myousefi2016/Allen-Cahn-CUDA -> id
readonly ALLOWED_LOGIN="myousefi2016"
readonly ALLOWED_USER_ID=22246708    # GET /users/myousefi2016 -> id

decision=$(
    /usr/bin/python3 -I -S - "$ALLOWED_REPO" "$ALLOWED_REPO_ID" "$ALLOWED_LOGIN" \
        "$ALLOWED_USER_ID" 2>&1 <<'PY'
import json, os, sys

def env(name):
    return os.environ.get(name, "")

allowed_repo = sys.argv[1]
allowed_repo_id = int(sys.argv[2])
allowed_login = sys.argv[3]
allowed_uid = int(sys.argv[4])

def verdict():
    if env("GITHUB_REPOSITORY").lower() != allowed_repo.lower():
        return f"DENY repository {env('GITHUB_REPOSITORY')!r}"
    for var in ("GITHUB_ACTOR", "GITHUB_TRIGGERING_ACTOR"):
        if env(var) != allowed_login:
            return f"DENY {var}={env(var)!r}"
    path = env("GITHUB_EVENT_PATH")
    try:
        with open(path, encoding="utf-8") as fh:
            ev = json.load(fh)
    except (OSError, ValueError) as exc:
        return f"DENY unreadable event payload {path!r}: {exc}"
    if not isinstance(ev, dict):
        return "DENY event payload is not an object"
    repo = ev.get("repository") or {}
    if repo.get("id") != allowed_repo_id:
        return f"DENY payload repository id {repo.get('id')!r}"
    sender = ev.get("sender") or {}
    if sender.get("id") != allowed_uid or sender.get("login") != allowed_login:
        return f"DENY sender {sender.get('login')!r} (id {sender.get('id')!r})"
    event = env("GITHUB_EVENT_NAME")
    if event in ("push", "workflow_dispatch"):
        return f"ALLOW {event}"
    if event == "pull_request":
        pr = ev.get("pull_request") or {}
        head_repo = (pr.get("head") or {}).get("repo") or {}
        if head_repo.get("id") != allowed_repo_id:
            return f"DENY pull_request from {head_repo.get('full_name')!r} (fork or deleted repository)"
        author = pr.get("user") or {}
        if author.get("id") != allowed_uid:
            return f"DENY pull_request authored by {author.get('login')!r}"
        return "ALLOW pull_request from this repository"
    return f"DENY event {event!r}"

print(verdict())
PY
)
# Anything but a single ALLOW line (e.g. a Python traceback) is a refusal.
case "$decision" in
    ALLOW*) [ "$(printf '%s\n' "$decision" | wc -l)" -eq 1 ] || decision="DENY guard error: $decision" ;;
    DENY*) ;;
    *) decision="DENY guard error: ${decision:-no output}" ;;
esac

line="$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ) run=${GITHUB_RUN_ID:-?} attempt=${GITHUB_RUN_ATTEMPT:-?} job=${GITHUB_JOB:-?} event=${GITHUB_EVENT_NAME:-?} actor=${GITHUB_ACTOR:-?} triggering_actor=${GITHUB_TRIGGERING_ACTOR:-?} ref=${GITHUB_REF:-?} sha=${GITHUB_SHA:-?} -> $decision"
/usr/bin/logger -t gh-runner-guard -- "$line" 2>/dev/null || true
echo "gpu-runner guard: $decision"

case "$decision" in
    ALLOW*) exit 0 ;;
    *) echo "::error::This self-hosted GPU runner only runs jobs triggered by $ALLOWED_LOGIN from $ALLOWED_REPO itself."; exit 1 ;;
esac
