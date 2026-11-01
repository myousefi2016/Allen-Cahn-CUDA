#!/usr/bin/env python3
"""Tests for ci/gpu-runner/job-started-guard.sh (stdlib only).

Each case feeds the guard the GITHUB_* variables and the event payload a real
job would carry and checks that it allows exactly the jobs triggered by the
repository owner from the repository's own code.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "ci" / "gpu-runner" / "job-started-guard.sh"

REPO = {"id": 155445643, "full_name": "myousefi2016/Allen-Cahn-CUDA"}
OWNER = {"login": "myousefi2016", "id": 22246708}
OTHER = {"login": "mallory", "id": 999001}
BOT = {"login": "dependabot[bot]", "id": 49699333}
FORK = {"id": 777002, "full_name": "mallory/Allen-Cahn-CUDA"}


def pr_payload(sender, author, head_repo):
    return {
        "repository": REPO,
        "sender": sender,
        "pull_request": {"user": author, "head": {"repo": head_repo}, "base": {"repo": REPO}},
    }


class GuardTest(unittest.TestCase):
    def run_guard(self, event, payload, actor="myousefi2016", triggering=None,
                  repository="myousefi2016/Allen-Cahn-CUDA", raw=None):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "event.json"
            path.write_text(raw if raw is not None else json.dumps(payload))
            env = {
                "PATH": "/usr/bin:/bin",
                "GITHUB_REPOSITORY": repository,
                "GITHUB_EVENT_NAME": event,
                "GITHUB_EVENT_PATH": str(path),
                "GITHUB_ACTOR": actor,
                "GITHUB_TRIGGERING_ACTOR": triggering if triggering is not None else actor,
                "GITHUB_RUN_ID": "1", "GITHUB_JOB": "build-and-test",
            }
            out = subprocess.run(["bash", str(GUARD)], env=env, capture_output=True, text=True,
                                 timeout=60)
        return out.returncode, out.stdout

    def assertAllowed(self, rc_out):
        rc, out = rc_out
        self.assertEqual(rc, 0, out)
        self.assertIn("ALLOW", out)

    def assertDenied(self, rc_out, reason):
        rc, out = rc_out
        self.assertNotEqual(rc, 0, out)
        self.assertIn("DENY", out)
        self.assertIn(reason, out)

    # ── allowed ──────────────────────────────────────────────────────────
    def test_push_by_owner(self):
        self.assertAllowed(self.run_guard("push", {"repository": REPO, "sender": OWNER}))

    def test_workflow_dispatch_by_owner(self):
        self.assertAllowed(self.run_guard("workflow_dispatch", {"repository": REPO, "sender": OWNER}))

    def test_same_repo_pull_request_by_owner(self):
        self.assertAllowed(self.run_guard("pull_request", pr_payload(OWNER, OWNER, REPO)))

    # ── refused ──────────────────────────────────────────────────────────
    def test_fork_pull_request_by_other_user(self):
        self.assertDenied(self.run_guard("pull_request", pr_payload(OTHER, OTHER, FORK),
                                         actor="mallory"), "GITHUB_ACTOR")

    def test_fork_pull_request_even_when_owner_pushes_to_it(self):
        # Owner pushes a commit onto a fork PR branch (or approves the run):
        # the code still comes from the fork.
        self.assertDenied(self.run_guard("pull_request", pr_payload(OWNER, OTHER, FORK)),
                          "fork")

    def test_owner_rerun_of_other_users_run(self):
        self.assertDenied(self.run_guard("pull_request", pr_payload(OTHER, OTHER, FORK),
                                         actor="mallory", triggering="myousefi2016"),
                          "GITHUB_ACTOR")

    def test_rerun_by_other_user_of_owner_run(self):
        self.assertDenied(self.run_guard("push", {"repository": REPO, "sender": OWNER},
                                         triggering="mallory"), "GITHUB_TRIGGERING_ACTOR")

    def test_bot_push(self):
        self.assertDenied(self.run_guard("push", {"repository": REPO, "sender": BOT},
                                         actor="dependabot[bot]"), "GITHUB_ACTOR")

    def test_sender_with_owner_login_but_other_id(self):
        spoof = {"login": "myousefi2016", "id": 123}
        self.assertDenied(self.run_guard("push", {"repository": REPO, "sender": spoof}), "sender")

    def test_same_repo_pull_request_by_other_author(self):
        self.assertDenied(self.run_guard("pull_request", pr_payload(OWNER, OTHER, REPO)),
                          "authored by")

    def test_other_repository(self):
        self.assertDenied(self.run_guard("push", {"repository": FORK, "sender": OWNER},
                                         repository="mallory/Allen-Cahn-CUDA"), "repository")

    def test_payload_repository_id_mismatch(self):
        self.assertDenied(self.run_guard("push", {"repository": FORK, "sender": OWNER}),
                          "repository id")

    def test_unlisted_events(self):
        for event in ("pull_request_target", "issue_comment", "workflow_run", "schedule",
                      "repository_dispatch"):
            with self.subTest(event=event):
                self.assertDenied(self.run_guard(event, {"repository": REPO, "sender": OWNER}),
                                  "event")

    def test_unreadable_or_malformed_payload(self):
        self.assertDenied(self.run_guard("push", None, raw="not json"), "unreadable")
        self.assertDenied(self.run_guard("push", None, raw="[1, 2]"), "not an object")

    def test_pull_request_from_deleted_head_repository(self):
        self.assertDenied(self.run_guard("pull_request", pr_payload(OWNER, OWNER, None)), "fork")

    def test_environment_cannot_override_the_allowed_account(self):
        # Even if a job's environment reached the hook, variables named like
        # configuration must not change the decision.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "event.json"
            path.write_text(json.dumps({"repository": FORK, "sender": OTHER}))
            env = {"PATH": "/usr/bin:/bin", "GITHUB_REPOSITORY": "mallory/Allen-Cahn-CUDA",
                   "GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": str(path),
                   "GITHUB_ACTOR": "mallory", "GITHUB_TRIGGERING_ACTOR": "mallory",
                   "ALLOWED_LOGIN": "mallory", "ALLOWED_USER_ID": "999001",
                   "ALLOWED_REPO": "mallory/Allen-Cahn-CUDA", "ALLOWED_REPO_ID": "777002",
                   "GUARD_ALLOWED_LOGIN": "mallory", "PYTHONPATH": d, "PYTHONSTARTUP": str(path)}
            out = subprocess.run(["bash", str(GUARD)], env=env, capture_output=True, text=True,
                                 timeout=60)
        self.assertNotEqual(out.returncode, 0, out.stdout)


if __name__ == "__main__":
    sys.exit(unittest.main())
