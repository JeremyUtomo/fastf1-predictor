#!/usr/bin/env bash
# Deploys origin/main once its GitHub Actions checks have passed.
# Runs every 2 minutes from fastf1-deploy.timer (see setup.sh); pull-based,
# so GitHub needs no SSH keys or AWS credentials.
set -euo pipefail

REPO=JeremyUtomo/fastf1-predictor
DIR=/opt/fastf1-predictor
SKIP=/var/lib/fastf1-deploy-skip  # last commit whose CI failed

main() {
    cd "$DIR"
    git fetch -q origin main
    local target
    target=$(git rev-parse origin/main)
    [ "$target" = "$(git rev-parse HEAD)" ] && return 0
    [ "$target" = "$(cat "$SKIP" 2>/dev/null)" ] && return 0

    local runs total pending failed
    runs=$(curl -fsS -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$REPO/commits/$target/check-runs")
    total=$(jq '.total_count' <<<"$runs")
    pending=$(jq '[.check_runs[] | select(.status != "completed")] | length' <<<"$runs")
    failed=$(jq '[.check_runs[] | select(.status == "completed"
        and (.conclusion | IN("success", "skipped", "neutral") | not))] | length' <<<"$runs")

    if [ "$total" -eq 0 ] || [ "$pending" -gt 0 ]; then
        echo "waiting for CI on ${target:0:7}"
        return 0
    fi
    if [ "$failed" -gt 0 ]; then
        echo "CI failed on ${target:0:7}, not deploying"
        echo "$target" > "$SKIP"
        return 0
    fi

    echo "deploying ${target:0:7}"
    git reset -q --hard "$target"
    docker compose up -d --build --remove-orphans
    docker image prune -f >/dev/null
}

main "$@"
