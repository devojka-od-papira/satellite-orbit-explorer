#!/usr/bin/env bash
# Daily data refresh: download fresh GP data from CelesTrak, run the tests on
# it and open (or update) a pull request with the new data.
#
# Every new element set is also added to the archive in data/archive/
# (orbit_explorer.archive), which keeps the history the daily data replaces.
#
# Run by .github/workflows/refresh.yml. Locally, DRY_RUN=1 downloads and tests
# as usual, prints the PR title and body, and then puts data/inputs back as it
# was; it never commits, pushes, comments or opens a PR, and only says what it
# would add to the archive. (outputs/ is left
# showing the downloaded day; re-run the scripts to go back.)
#
# Opening the PR needs the repository setting "Allow GitHub Actions to create
# and approve pull requests" (Settings > Actions > General).
set -euo pipefail

BRANCH=data/daily-refresh
BOT_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
DATA=(data/inputs/orbital_elements.json data/inputs/data_manifest.json)
# Dry runs put the old manifest back, so remember their download time here,
# or repeated dry runs would ignore CelesTrak's rate limit.
LAST_DRY_RUN="$(git rev-parse --git-dir)/daily-refresh-last-download.json"

WORK=$(mktemp -d)
cleanup() {
  if [ "${DRY_RUN:-}" = 1 ] && [ -f "$WORK/orbital_elements.json" ]; then
    if ! cmp -s data/inputs/data_manifest.json "$WORK/data_manifest.json"; then
      cp data/inputs/data_manifest.json "$LAST_DRY_RUN"
    fi
    cp "$WORK/orbital_elements.json" "$WORK/data_manifest.json" data/inputs/
    echo "DRY RUN: data/inputs restored."
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

pr_number() {
  gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number // empty'
}

# 0. The refresh branch, if any. ls-remote fails loudly on a network or auth
#    error; an empty answer means the branch doesn't exist.
REMOTE_SHA=$(git ls-remote --heads origin "$BRANCH" | cut -f1)
SEEN=(--seen-manifest "$LAST_DRY_RUN")
if [ -n "$REMOTE_SHA" ]; then
  git fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  # Its manifest records a download that main doesn't know about yet.
  git show "origin/$BRANCH:data/inputs/data_manifest.json" > "$WORK/branch_manifest.json"
  SEEN+=(--seen-manifest "$WORK/branch_manifest.json")
  # Its archive holds the element sets of every download since the last merge;
  # the branch is replaced below, so they are carried over.
  if git cat-file -e "origin/$BRANCH:data/archive" 2>/dev/null; then
    mkdir "$WORK/branch"
    git archive "origin/$BRANCH" data/archive | tar -x -C "$WORK/branch"
  fi

  # Never overwrite someone's own commits on an open refresh PR. Merge commits
  # (e.g. GitHub's "Update branch") don't count. A branch whose PR was closed
  # is simply replaced.
  PR=$(pr_number)
  if [ -n "$PR" ]; then
    OTHERS=$(git log --no-merges "origin/main..origin/$BRANCH" --format=%ae \
      | grep -vcF "$BOT_EMAIL" || true)
    if [ "$OTHERS" -gt 0 ]; then
      MSG="Skipped the $(date -u +%F) refresh: this PR has $OTHERS commit(s) not made by the refresh workflow, and refreshing would force-push over them. Merge or close the PR to resume the daily refresh."
      echo "$MSG"
      if [ "${DRY_RUN:-}" != 1 ]; then gh pr comment "$PR" --body "$MSG"; fi
      exit 0
    fi
  fi
fi

if [ "${DRY_RUN:-}" = 1 ]; then cp "${DATA[@]}" "$WORK/"; fi

# 1. Download. fetch.py refuses within 2 hours of the last download (exit 3)
#    and writes nothing unless every satellite validates.
set +e
python -m orbit_explorer.fetch "${SEEN[@]}" | tee "$WORK/fetch.txt"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -eq 3 ]; then exit 0; fi
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

# 2. Archive the new element sets (and an unmerged refresh's), before the
#    tests, which check the archive too.
if [ "${DRY_RUN:-}" = 1 ]; then
  python -m orbit_explorer.archive add --dry-run | tee "$WORK/archive.txt"
else
  if [ -d "$WORK/branch/data/archive" ]; then
    python -m orbit_explorer.archive merge "$WORK/branch/data/archive"
  fi
  python -m orbit_explorer.archive add | tee "$WORK/archive.txt"
fi

# 3. Test the new data. A failure stops here: no commit, no PR. (CI runs the
#    tests again on every Python version; this run keeps bad data out of PRs.)
python -m pytest -q

# 4. What the new day looks like, for the PR description.
export MPLBACKEND=Agg
python -m orbit_explorer.sgp4_compare > /dev/null
python -m orbit_explorer.j2 > /dev/null
python -m orbit_explorer.visibility > /dev/null
DAY=$(python -c "from orbit_explorer.elements import load_reference_start as s; print(f'{s():%Y-%m-%d}')")
TITLE="Refresh GP data from CelesTrak ($DAY)"

STALE=$(python - <<'PY'
import json

from orbit_explorer.elements import MANIFEST_PATH
from orbit_explorer.fetch import STALE_DAYS, stale_records

stale = stale_records(json.loads(MANIFEST_PATH.read_text()))
if stale:
    print(f"Epoch more than {STALE_DAYS} days from the window start: {', '.join(stale)}")
PY
)
WARNING=""
if [ -n "$STALE" ]; then
  TITLE="$TITLE: stale elements"
  WARNING=$(printf '> [!WARNING]\n> %s.\n> Check CelesTrak before merging: the viewer would be built on old elements.\n' "$STALE")
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

$(cat "$WORK/archive.txt")

<details><summary>Element set archive</summary>

\`\`\`
$(python -m orbit_explorer.archive summary)
\`\`\`
</details>

<details><summary>Kepler vs SGP4</summary>

\`\`\`
$(cat outputs/sgp4_comparison.txt)
\`\`\`
</details>

<details><summary>Two-body vs J2</summary>

\`\`\`
$(cat outputs/j2_comparison.txt)
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

# 5. One branch, replaced each day, so open refreshes never pile up. The lease
#    makes the push fail if anyone pushed to the branch after the check above
#    (an empty SHA means "the branch must still not exist").
git config user.name "github-actions[bot]"
git config user.email "$BOT_EMAIL"
git switch -C "$BRANCH"
git add "${DATA[@]}" data/archive
git commit -m "$TITLE"
git push --force-with-lease="$BRANCH:$REMOTE_SHA" origin "$BRANCH"

# Pull requests opened with GITHUB_TOKEN do not trigger other workflows, so
# start CI on the branch explicitly, right after the push, so the new commit
# gets checks even if the PR step below fails.
gh workflow run ci.yml --ref "$BRANCH"

PR=$(pr_number)
if [ -n "$PR" ]; then
  gh pr edit "$PR" --title "$TITLE" --body-file "$WORK/body.md"
elif ! gh pr create --base main --head "$BRANCH" --title "$TITLE" --body-file "$WORK/body.md"; then
  echo "::error::Could not open the PR. If the error above says GitHub Actions" \
    "is not permitted to create pull requests, turn on 'Allow GitHub Actions to" \
    "create and approve pull requests' in Settings > Actions > General."
  exit 1
fi
