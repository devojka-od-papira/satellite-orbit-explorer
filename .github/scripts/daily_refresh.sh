#!/usr/bin/env bash
# Daily data refresh: download fresh GP data from CelesTrak, run the tests on
# it and open (or update) a pull request with the new data.
#
# Run by .github/workflows/refresh.yml. Locally, DRY_RUN=1 stops before any
# commit, push or PR and prints the PR body instead.
set -euo pipefail

BRANCH=data/daily-refresh
WORK=$(mktemp -d)

# 1. Download. fetch.py writes nothing unless every satellite validates, so a
#    failed download fails this step and leaves the data untouched.
python -m orbit_explorer.fetch | tee "$WORK/fetch.txt"

# 2. Test the new data. A failure stops here: no commit, no PR.
python -m pytest -q

# 3. What the new day looks like, for the PR description.
export MPLBACKEND=Agg
python -m orbit_explorer.sgp4_compare > /dev/null
python -m orbit_explorer.visibility > /dev/null
DAY=$(python -c "from orbit_explorer.elements import load_reference_start as s; print(f'{s():%Y-%m-%d}')")
TITLE="Refresh GP data from CelesTrak ($DAY)"

{
  echo "Daily refresh of the GP elements for the four satellites. The propagation"
  echo "window moves to **$DAY 00:00 UTC**. All tests pass on the new data."
  echo
  echo "Merging publishes the 3D viewer for the new day. The README figures and"
  echo "numbers stay at the documented date."
  echo
  echo '```'
  cat "$WORK/fetch.txt"
  echo '```'
  echo
  echo "<details><summary>Kepler vs SGP4</summary>"
  echo
  echo '```'
  cat outputs/sgp4_comparison.txt
  echo '```'
  echo "</details>"
  echo
  echo "<details><summary>Passes over Belgrade</summary>"
  echo
  echo '```'
  cat outputs/visibility.txt
  echo '```'
  echo "</details>"
  echo
  echo "_Opened by the daily refresh workflow. If a newer refresh runs before this"
  echo "is merged, it updates this PR instead of opening another._"
} > "$WORK/body.md"

if [ "${DRY_RUN:-}" = 1 ]; then
  echo "--- DRY RUN: would commit data/inputs and open or update a PR ---"
  echo "Title: $TITLE"
  cat "$WORK/body.md"
  exit 0
fi

# 4. One branch, force-pushed each day, so open refreshes never pile up.
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git switch -C "$BRANCH"
git add data/inputs
git commit -m "$TITLE"
git push --force origin "$BRANCH"

PR=$(gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number // empty')
if [ -n "$PR" ]; then
  gh pr edit "$PR" --title "$TITLE" --body-file "$WORK/body.md"
else
  gh pr create --base main --head "$BRANCH" --title "$TITLE" --body-file "$WORK/body.md"
fi

# Pull requests opened with GITHUB_TOKEN do not trigger other workflows, so
# start CI on the branch explicitly; its checks then show on the PR.
gh workflow run ci.yml --ref "$BRANCH"
