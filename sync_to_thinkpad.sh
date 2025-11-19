#!/usr/bin/env bash
# Quick helper to add, commit and push current changes.

set -e

cd "$(dirname "$0")"

echo "=== Git status BEFORE sync ==="
git status

echo
read -p "Commit message (leave empty for 'quick sync'): " MSG
if [ -z "$MSG" ]; then
  MSG="quick sync"
fi

echo
echo "Adding all changes..."
git add -A

echo "Committing with message: $MSG"
git commit -m "$MSG" || {
  echo "Nothing to commit (maybe no changes?)."
  exit 0
}

echo "Pushing to origin/main..."
git push origin main

echo "=== Sync done. You can now restart assistant on ThinkPad. ==="
