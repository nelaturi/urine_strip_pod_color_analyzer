#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$ROOT_DIR")"
OUTPUT_DIR="${1:-$ROOT_DIR/download}"

mkdir -p "$OUTPUT_DIR"

BUNDLE_PATH="$OUTPUT_DIR/${REPO_NAME}.bundle"
ZIP_PATH="$OUTPUT_DIR/${REPO_NAME}-source.zip"
README_PATH="$OUTPUT_DIR/README_MANUAL_PUSH.md"

# Full repository history for migration to a new remote.
git -C "$ROOT_DIR" bundle create "$BUNDLE_PATH" --all

# Downloadable source snapshot for opening in VS Code.
git -C "$ROOT_DIR" archive --format=zip --output "$ZIP_PATH" HEAD

cat > "$README_PATH" <<README
# Manual GitHub push guide

Generated files in this download folder:
- **$(basename "$BUNDLE_PATH")**: full git history (all branches + tags)
- **$(basename "$ZIP_PATH")**: source snapshot for VS Code
- **$(basename "$README_PATH")**: this manual push guide

## Push full project history to a new GitHub repository

\`\`\`bash
git clone "$(basename "$BUNDLE_PATH")" "$REPO_NAME"
cd "$REPO_NAME"
git remote add origin <NEW_GITHUB_REPO_URL>
git push -u origin --all
git push origin --tags
\`\`\`

## Open source snapshot in VS Code

\`\`\`bash
unzip "$(basename "$ZIP_PATH")" -d "$REPO_NAME-source"
code "$REPO_NAME-source"
\`\`\`
README

printf 'Created:\n- %s\n- %s\n- %s\n' "$BUNDLE_PATH" "$ZIP_PATH" "$README_PATH"
