#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/PQ_project}"
HF_HOME_DEFAULT="${HF_HOME:-/workspace/hf_cache}"
HUGGINGFACE_HUB_CACHE_DEFAULT="${HUGGINGFACE_HUB_CACHE:-${HF_HOME_DEFAULT}/hub}"
INSTALL_OS_TOOLS=1

usage() {
  cat <<'USAGE'
Usage: tools/runpod_bootstrap.sh [--no-install]

Checks a RunPod benchmark worker after Pod replacement/migration.

By default, when run as root with apt-get available, it installs missing
system tools used by the project workflow. Use --no-install for read-only mode.

Environment overrides:
  REPO_DIR                 default: /workspace/PQ_project
  HF_HOME                  default: /workspace/hf_cache
  HUGGINGFACE_HUB_CACHE    default: /workspace/hf_cache/hub
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-install)
      INSTALL_OS_TOOLS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

section() {
  printf '\n== %s ==\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

install_missing_os_tools() {
  local missing=()
  local tool
  for tool in git tmux rsync curl; do
    if ! have "$tool"; then
      missing+=("$tool")
    fi
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    echo "System tools present: git tmux rsync curl"
    return
  fi

  echo "Missing system tools: ${missing[*]}"
  if [ "$INSTALL_OS_TOOLS" -eq 0 ]; then
    echo "Read-only mode; not installing. Run as root without --no-install to install them."
    return
  fi
  if [ "$(id -u)" -ne 0 ]; then
    echo "Not root; install manually: apt-get update && apt-get install -y ${missing[*]}"
    return
  fi
  if ! have apt-get; then
    echo "apt-get not available; install missing tools through the active RunPod image/template."
    return
  fi

  apt-get update
  apt-get install -y "${missing[@]}"
}

section "RunPod Bootstrap"
date -u +"UTC time: %Y-%m-%dT%H:%M:%SZ"
echo "Host: $(hostname)"
echo "User: $(id -un) (uid $(id -u))"

section "System Tools"
install_missing_os_tools

section "GPU"
if have nvidia-smi; then
  if ! nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; then
    echo "nvidia-smi is present but could not query the GPU"
  fi
else
  echo "nvidia-smi not found"
fi

section "Persistent Paths"
export HF_HOME="$HF_HOME_DEFAULT"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE_DEFAULT"
echo "REPO_DIR=$REPO_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
if ! mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"; then
  echo "Could not create cache directories; check filesystem permissions"
fi

if [ -d "$HF_HOME" ]; then
  du -sh "$HF_HOME" 2>/dev/null || true
fi

section "Repository"
if [ ! -d "$REPO_DIR" ]; then
  echo "Repo missing at $REPO_DIR"
else
  cd "$REPO_DIR"
  pwd
  if have git && [ -d .git ]; then
    git status --short
    git rev-parse --short HEAD
  fi
fi

section "Python Environment"
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  "$REPO_DIR/.venv/bin/python" --version
  "$REPO_DIR/.venv/bin/python" -m pip show torch transformers bitsandbytes accelerate psutil \
    | awk '/^(Name|Version):/ {print}'
else
  echo "Project venv missing at $REPO_DIR/.venv"
fi

section "tmux"
if have tmux; then
  tmux -V
  tmux ls 2>/dev/null || echo "No active tmux sessions"
else
  echo "tmux unavailable"
fi

section "Done"
echo "Bootstrap check complete. Exported HF cache paths apply to this process only."
