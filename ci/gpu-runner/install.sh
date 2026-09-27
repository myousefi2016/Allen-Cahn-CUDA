#!/usr/bin/env bash
# ============================================================================
# Install the repository's self-hosted GPU runner on this host (run as root).
#
#   sudo ./install.sh --reg-token-file FILE   # 1-hour token from repo Settings
#   sudo ./install.sh --pat-file FILE         # mint the token with a PAT
#
# FILE is read once and deleted. Result:
#   * user gh-runner (system account, no login shell), member of `docker`
#     (the CI job runs in a container with --gpus all);
#   * actions-runner v2.337.0 (SHA-256 verified) in /opt/gh-runner,
#     registered to myousefi2016/Allen-Cahn-CUDA only, labels
#     self-hosted,Linux,X64,gpu, running as a systemd service;
#   * job-started-guard.sh installed root-owned in /usr/local/lib/gh-runner-guard
#     and set as ACTIONS_RUNNER_HOOK_JOB_STARTED in the runner's .env, which is
#     made immutable (chattr +i) so no job can unset the guard.
# ============================================================================
set -euo pipefail

readonly REPO="myousefi2016/Allen-Cahn-CUDA"
readonly RUNNER_VERSION="2.337.0"
readonly RUNNER_SHA256="70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"
readonly RUNNER_USER="gh-runner"
readonly RUNNER_HOME="/opt/gh-runner"
readonly GUARD_DIR="/usr/local/lib/gh-runner-guard"
readonly HERE="$(cd "$(dirname "$0")" && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" -eq 0 ] || die "run as root"
[ $# -eq 2 ] || die "usage: $0 --reg-token-file FILE | --pat-file FILE"
mode="$1"; secret_file="$2"
[ -f "$secret_file" ] || die "$secret_file not found"
secret="$(tr -d '[:space:]' < "$secret_file")"
rm -f -- "$secret_file"
[ -n "$secret" ] || die "empty secret"

for cmd in docker nvidia-ctk curl sha256sum chattr systemctl python3 logger; do
    command -v "$cmd" >/dev/null || die "$cmd not installed"
done

case "$mode" in
    --reg-token-file) reg_token="$secret" ;;
    --pat-file)
        # The PAT goes to curl through a 0600 header file, never argv.
        hdr="$(mktemp)"; chmod 600 "$hdr"
        printf 'Authorization: Bearer %s\n' "$secret" > "$hdr"
        reg_token="$(curl -fsS -X POST -H @"$hdr" -H 'Accept: application/vnd.github+json' \
            "https://api.github.com/repos/$REPO/actions/runners/registration-token" |
            python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
        rm -f -- "$hdr" ;;
    *) die "unknown option $mode" ;;
esac
unset secret

# Guard: root-owned, outside the runner directory, not writable by the runner.
install -d -o root -g root -m 0755 "$GUARD_DIR"
install -o root -g root -m 0755 "$HERE/job-started-guard.sh" "$GUARD_DIR/job-started-guard.sh"

# Runner user.
id "$RUNNER_USER" >/dev/null 2>&1 ||
    useradd --system --create-home --home-dir "$RUNNER_HOME" --shell /usr/sbin/nologin "$RUNNER_USER"
usermod -aG docker "$RUNNER_USER"

# Runner, verified.
tarball="actions-runner-linux-x64-$RUNNER_VERSION.tar.gz"
cd "$RUNNER_HOME"
curl -fsSLo "$tarball" "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/$tarball"
echo "$RUNNER_SHA256  $tarball" | sha256sum -c -
install -d -o "$RUNNER_USER" -g "$RUNNER_USER" "$RUNNER_HOME/actions-runner"
tar -xzf "$tarball" -C "$RUNNER_HOME/actions-runner"
rm -f "$tarball"
chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_HOME/actions-runner"
cd "$RUNNER_HOME/actions-runner"

runuser -u "$RUNNER_USER" -- ./config.sh --unattended --replace \
    --url "https://github.com/$REPO" --token "$reg_token" \
    --name "gpu-$(hostname -s)" --labels gpu --work _work
unset reg_token

# Hook: .env is read by the service at start; immutable so no job can drop it.
[ -f .env ] && chattr -i .env 2>/dev/null || true
grep -v '^ACTIONS_RUNNER_HOOK_JOB_STARTED=' .env 2>/dev/null > .env.new || true
echo "ACTIONS_RUNNER_HOOK_JOB_STARTED=$GUARD_DIR/job-started-guard.sh" >> .env.new
mv .env.new .env
chown root:root .env; chmod 0644 .env
chattr +i .env

./svc.sh install "$RUNNER_USER"
./svc.sh start
systemctl --no-pager status "$(./svc.sh status 2>/dev/null | grep -o 'actions\.runner\.[^ ]*\.service' | head -1)" | head -5 || true
echo "Installed. Guard decisions: journalctl -t gh-runner-guard"
