#!/bin/bash

# Auto git commit and push script
# Usage: ./git-commit.sh [optional commit message]

cd "$(dirname "$0")"

MESSAGE="${1:-fix: update code $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A
git commit -m "$MESSAGE"
git push

echo "Done: committed and pushed with message: '$MESSAGE'"
