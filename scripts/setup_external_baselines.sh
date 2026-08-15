#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="$ROOT/baseline"

clone_pinned() {
  local name="$1" url="$2" revision="$3" target="$BASELINE_DIR/$name"
  if [[ ! -d "$target/.git" ]]; then
    git clone "$url" "$target"
  fi
  git -C "$target" fetch --tags origin
  git -C "$target" checkout --detach "$revision"
}

mkdir -p "$BASELINE_DIR"
clone_pinned "STID-official" "https://github.com/GestaltCogTeam/STID.git" "e8b313bc591bdd0101a1619962c9b503e75127c0"
clone_pinned "STAEformer-official" "https://github.com/XDZhelheim/STAEformer.git" "fc49d39b2f1a8e3cf37b6289d7240680e1690f3f"
clone_pinned "EAC-official" "https://github.com/Onedean/EAC.git" "0a99297e01e484d56b2dfc845eacbbcf733efd1b"
clone_pinned "PDFormer-official" "https://github.com/BUAABIGSCity/PDFormer.git" "f8c8f6ad007a04fad3baee958b89504711852ce9"

echo "External baselines are available under $BASELINE_DIR"
