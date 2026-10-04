#!/usr/bin/env bash
# Crée le dépôt GitHub public et pousse main.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO_NAME="${REPO_NAME:-maths-appliquees-l3-m1}"
VISIBILITY="${VISIBILITY:-public}"

gh auth status
OWNER="$(gh api user --jq .login)"
echo "Compte GitHub: $OWNER"
echo "Repo: $OWNER/$REPO_NAME ($VISIBILITY)"

if gh repo view "$OWNER/$REPO_NAME" >/dev/null 2>&1; then
  echo "Le dépôt existe déjà — mise à jour du remote et push."
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$OWNER/$REPO_NAME.git"
else
  gh repo create "$REPO_NAME" \
    --"$VISIBILITY" \
    --description "9 projets de maths appliquées L3/M1 : finance, physique, informatique" \
    --source=. \
    --remote=origin \
    --push
  gh repo edit "$OWNER/$REPO_NAME" \
    --add-topic mathematics \
    --add-topic finance \
    --add-topic physics \
    --add-topic python \
    --add-topic numerical-methods \
    --add-topic portfolio || true
  echo "Publié: https://github.com/$OWNER/$REPO_NAME"
  exit 0
fi

git push -u origin main
gh repo edit "$OWNER/$REPO_NAME" \
  --add-topic mathematics \
  --add-topic finance \
  --add-topic physics \
  --add-topic python \
  --add-topic numerical-methods \
  --add-topic portfolio || true
echo "Publié: https://github.com/$OWNER/$REPO_NAME"
