#!/usr/bin/env bash
# Daily data refresh: download fresh GP data from CelesTrak, run the tests on
# it and open (or update) a pull request with the new data.
#
# Run by .github/workflows/refresh.yml. Locally, DRY_RUN=1 downloads and tests
# as usual, prints the PR title and body, and then puts data/inputs back as it
# was; it never commits, pushes or opens a PR. (outputs/ is left showing the
# downloaded day; re-run the scripts to go back.)
#
# Needs the repository setting "Allow GitHub Actions to create and approve
# pull requests" (Settings > Actions > General) to open the PR.
set -euo pipefail

BRANCH=data/daily-refresh
BOT_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
MIN_HOURS_BETWEEN_DOWNLOADS=2 # CelesTrak: same data at most once per 2 hours
DATA=(data/inputs/orbital_elements.json data/inputs/data_manifest.json)

WORK=$(mktemp -d)
cleanup() {
  if [ "${DRY_RUN:-}" = 1 ] && [ -f "$WORK/orbital_elements.json" ]; then
    cp "$WORK/orbital_elements.json" "$WORK/data_manifest.json" data/inputs/
    echo "DRY RUN: data/inputs restored."
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

# 0. Checks before downloading anything.
HOURS=$(python - <<'PY'
import json
from datetime import UTC, datetime

manifest = json.load(open("data/inputs/data_manifest.json"))
last = datetime.fromisoformat(manifest["retrieved_utc"].replace("Z", "+00:00"))
print(f"{(datetime.now(UTC) - last).total_seconds() / 3600:.1f}")
PY
)
if python -c "import sys; sys.exit(not $HOURS < $MIN_HOURS_BETWEEN_DOWNLOADS)"; then
  echo "Last download was $HOURS h ago; CelesTrak asks for at most one every" \
    "$MIN_HOURS_BETWEEN_DOWNLOADS h. Nothing to do."
  exit 0
fi

if [ "${DRY_RUN:-}" != 1 ]; then
  # The setting can only be read with admin rights; when the token can't read
  # it, the PR step below reports the same problem.
  if ALLOWED=$(gh api "repos/$GITHUB_REPOSITORY/actions/permissions/workflow" \
    --jq .can_approve_pull_request_reviews 2>/dev/null) && [ "$ALLOWED" = false ]; then
    echo "::error::Turn on 'Allow GitHub Actions to create and approve pull" \
      "requests' in Settings > Actions > General, or no PR can be opened."
    exit 1
  fi

  # Never overwrite someone's own commits on the open refresh PR.
  if git fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null; then
    OTHERS=$(git log "origin/main..origin/$BRANCH" --format=%ae | grep -vcF "$BOT_EMAIL" || true)
    if [ "$OTHERS" -gt 0 ]; then
      PR=$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number // empty')
      MSG="Skipped the $(date -u +%F) refresh: \`$BRANCH\` has $OTHERS commit(s) not made by the refresh workflow, and refreshing would force-push over them. Merge or close this PR to resume the daily refresh."
      echo "$MSG"
      if [ -n "$PR" ]; then gh pr comment "$PR" --body "$MSG"; fi
      exit 0
    fi
  fi
fi

if [ "${DRY_RUN:-}" = 1 ]; then cp "${DATA[@]}" "$WORK/"; fi

# 1. Download. fetch.py writes nothing unless every satellite validates, so a
#    failed download fails this step and leaves the data untouched.
python -m orbit_explorer.fetch | tee "$WORK/fetch.txt"

# 2. Test the new data. A failure stops here: no commit, no PR. (CI runs the
#    tests again on every Python version; this run keeps bad data out of PRs.)
python -m pytest -q

# 3. What the new day looks like, for the PR description.
export MPLBACKEND=Agg
python -m orbit_explorer.sgp4_compare > /dev/null
python -m orbit_explorer.visibility > /dev/null
DAY=$(python -c "from orbit_explorer.elements import load_reference_start as s; print(f'{s():%Y-%m-%d}')")
TITLE="Refresh GP data from CelesTrak ($DAY)"

STALE=$(grep '^Warning:' "$WORK/fetch.txt" | sed 's/^Warning: //' || true)
WARNING=""
if [ -n "$STALE" ]; then
  TITLE="$TITLE: stale elements"
  WARNING=$(printf '> [!WARNING]\n> %s\n> Check CelesTrak before merging: the viewer would be built on old elements.\n' "$STALE")
fi

cat > "$WORK/body.md" <<EOF
$WARNING

Daily refresh of the GP elements for the four satellites. The propagation
window moves to **$DAY 00:00 UTC**. All tests pass on the new data.

Merging publishes the 3D viewer for the new day. The README figures and
numbers stay at their documented date.

\`\`\`
$(cat "$WORK/fetch.txt")
\`\`\`

<details><summary>Kepler vs SGP4</summary>

\`\`\`
$(cat outputs/sgp4_comparison.txt)
\`\`\`
</details>

<details><summary>Passes over Belgrade</summary>

\`\`\`
$(cat outputs/visibility.txt)
\`\`\`
</details>

_Opened by the daily refresh workflow. If a newer refresh runs before this is
merged, it updates this PR instead of opening another._
EOF

if [ "${DRY_RUN:-}" = 1 ]; then
  echo "--- DRY RUN: would commit data/inputs and open or update a PR ---"
  echo "Title: $TITLE"
  cat "$WORK/body.md"
  exit 0
fi

# 4. One branch, force-pushed each day, so open refreshes never pile up.
git config user.name "github-actions[bot]"
git config user.email "$BOT_EMAIL"
git switch -C "$BRANCH"
git add "${DATA[@]}"
git commit -m "$TITLE"
git push --force origin "$BRANCH"

# Pull requests opened with GITHUB_TOKEN do not trigger other workflows, so
# start CI on the branch explicitly, right after the push, so the new commit
# gets checks even if the PR step below fails.
gh workflow run ci.yml --ref "$BRANCH"

PR=$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number // empty')
if [ -n "$PR" ]; then
  gh pr edit "$PR" --title "$TITLE" --body-file "$WORK/body.md"
elif ! gh pr create --base main --head "$BRANCH" --title "$TITLE" --body-file "$WORK/body.md"; then
  echo "::error::Could not open the PR. If the error above says GitHub Actions" \
    "is not permitted to create pull requests, turn on 'Allow GitHub Actions to" \
    "create and approve pull requests' in Settings > Actions > General."
  exit 1
fi
