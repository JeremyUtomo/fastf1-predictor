#!/usr/bin/env bash
# Deploys origin/main once its GitHub Actions checks have passed.
# Runs every 2 minutes from fastf1-deploy.timer (see setup.sh); pull-based,
# so GitHub needs no SSH keys or AWS credentials.
#
# Also deploys the Worlds 2026 predictor (worlds-* services, built from a local
# clone of JeremyUtomo/worlds-2026-predictor) when its main branch has a new
# commit with green CI.
set -euo pipefail

REPO=JeremyUtomo/fastf1-predictor
DIR=/opt/fastf1-predictor
SKIP=/var/lib/fastf1-deploy-skip           # last commit whose CI failed
DEPLOYED=/var/lib/fastf1-deployed          # last commit that came up successfully

WORLDS_REPO=JeremyUtomo/worlds-2026-predictor
WORLDS_DIR=/opt/worlds-2026-predictor
WORLDS_DEPLOYED=/var/lib/fastf1-deploy-worlds       # Worlds commit currently deployed
WORLDS_SKIP=/var/lib/fastf1-deploy-worlds-skip      # last Worlds commit whose CI failed
export WORLDS_DIR

# prints "pending", "failed" or "ok" for a commit's GitHub checks
ci_state() {
    local runs total pending failed
    runs=$(curl -fsS -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$1/commits/$2/check-runs")
    total=$(jq '.total_count' <<<"$runs")
    pending=$(jq '[.check_runs[] | select(.status != "completed")] | length' <<<"$runs")
    failed=$(jq '[.check_runs[] | select(.status == "completed"
        and (.conclusion | IN("success", "skipped", "neutral") | not))] | length' <<<"$runs")
    if [ "$total" -eq 0 ] || [ "$pending" -gt 0 ]; then
        echo pending
    elif [ "$failed" -gt 0 ]; then
        echo failed
    else
        echo ok
    fi
}

# puts the Worlds clone at a commit (cloning it the first time)
worlds_checkout() {
    [ -d "$WORLDS_DIR/.git" ] || git clone -q "https://github.com/$WORLDS_REPO.git" "$WORLDS_DIR"
    git -C "$WORLDS_DIR" fetch -q origin main
    git -C "$WORLDS_DIR" reset -q --hard "$1"
}

main() {
    cd "$DIR"
    git fetch -q origin main
    local target
    target=$(git rev-parse origin/main)
    # compare with what last came up, not HEAD: a failed start is retried
    [ "$target" = "$(cat "$DEPLOYED" 2>/dev/null)" ] && return 0
    [ "$target" = "$(cat "$SKIP" 2>/dev/null)" ] && return 0

    case "$(ci_state "$REPO" "$target")" in
        pending)
            echo "waiting for CI on ${target:0:7}"
            return 0 ;;
        failed)
            echo "CI failed on ${target:0:7}, not deploying"
            echo "$target" > "$SKIP"
            return 0 ;;
    esac

    echo "deploying ${target:0:7}"
    git reset -q --hard "$target"
    worlds_checkout "$(cat "$WORLDS_DEPLOYED" 2>/dev/null || echo origin/main)"
    docker compose up -d --build --remove-orphans
    echo "$target" > "$DEPLOYED"
    docker image prune -f >/dev/null
}

worlds() {
    cd "$DIR"
    local target
    target=$(git ls-remote "https://github.com/$WORLDS_REPO.git" refs/heads/main | cut -f1)
    [ -n "$target" ] || return 0
    [ "$target" = "$(cat "$WORLDS_DEPLOYED" 2>/dev/null)" ] && return 0
    [ "$target" = "$(cat "$WORLDS_SKIP" 2>/dev/null)" ] && return 0

    case "$(ci_state "$WORLDS_REPO" "$target")" in
        pending)
            echo "waiting for Worlds CI on ${target:0:7}"
            return 0 ;;
        failed)
            echo "Worlds CI failed on ${target:0:7}, not deploying"
            echo "$target" > "$WORLDS_SKIP"
            return 0 ;;
    esac

    echo "deploying Worlds ${target:0:7}"
    worlds_checkout "$target"
    docker compose up -d --build worlds-app worlds-updater
    echo "$target" > "$WORLDS_DEPLOYED"
    docker image prune -f >/dev/null
}

main "$@"
worlds
