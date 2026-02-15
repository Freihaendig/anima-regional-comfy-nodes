#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Freihaendig/anima-regional-comfy-nodes.git}"
TARGET_DIR="${TARGET_DIR:-anima_regional}"

if [[ -z "${COMFYUI_DIR:-}" ]]; then
  if [[ -d "custom_nodes" ]]; then
    COMFYUI_DIR="$(pwd)"
  elif [[ -d "/workspace/ComfyUI/custom_nodes" ]]; then
    COMFYUI_DIR="/workspace/ComfyUI"
  else
    echo "Set COMFYUI_DIR to your ComfyUI path, then rerun."
    echo "Example: COMFYUI_DIR=/opt/ComfyUI bash install.sh"
    exit 1
  fi
fi

CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
DEST="$CUSTOM_NODES_DIR/$TARGET_DIR"

mkdir -p "$CUSTOM_NODES_DIR"

if [[ -d "$DEST/.git" ]]; then
  echo "Updating existing install at $DEST"
  git -C "$DEST" pull --ff-only
else
  if [[ -e "$DEST" ]]; then
    echo "Path exists and is not a git repo: $DEST"
    exit 1
  fi
  echo "Cloning to $DEST"
  git clone "$REPO_URL" "$DEST"
fi

if [[ -f "$DEST/requirements.txt" ]] && grep -Eq '^[[:space:]]*[^#[:space:]]' "$DEST/requirements.txt"; then
  echo "Installing Python dependencies"
  python -m pip install -r "$DEST/requirements.txt"
else
  echo "No external Python dependencies to install"
fi

echo "Install complete. Restart ComfyUI."
