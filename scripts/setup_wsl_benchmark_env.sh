#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required inside WSL2." >&2
  exit 1
fi

NODE_INSTALL_ROOT="${HOME}/.local/opt"
NODE_LINK="${NODE_INSTALL_ROOT}/node-v22"
export PATH="${NODE_LINK}/bin:${HOME}/.local/bin:${PATH}"
if ! command -v node >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install a local node runtime inside WSL2." >&2
    exit 1
  fi
  mkdir -p "${NODE_INSTALL_ROOT}"
  mkdir -p "${HOME}/.local/bin"
  NODE_BASE_URL="${PTXBENCH_NODE_BASE_URL:-https://nodejs.org/dist/latest-v22.x}"
  NODE_TARBALL="$(curl -fsSL "${NODE_BASE_URL}/SHASUMS256.txt" | awk '/linux-x64.tar.xz$/ {print $2; exit}')"
  if [[ -z "${NODE_TARBALL}" ]]; then
    echo "Failed to resolve a Linux node tarball from ${NODE_BASE_URL}." >&2
    exit 1
  fi
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "${TMPDIR}"' EXIT
  curl -fsSL "${NODE_BASE_URL}/${NODE_TARBALL}" -o "${TMPDIR}/${NODE_TARBALL}"
  tar -xJf "${TMPDIR}/${NODE_TARBALL}" -C "${NODE_INSTALL_ROOT}"
  NODE_EXTRACTED_DIR="${NODE_INSTALL_ROOT}/${NODE_TARBALL%.tar.xz}"
  ln -sfn "${NODE_EXTRACTED_DIR}" "${NODE_LINK}"
  ln -sfn "${NODE_LINK}/bin/node" "${HOME}/.local/bin/node"
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to install the Linux Codex CLI inside WSL2." >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1 || ! codex --version >/dev/null 2>&1; then
  npm install --global --prefix "${HOME}/.local" @openai/codex@latest >/dev/null
  hash -r
fi

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv inside WSL2." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv python install 3.13
rm -rf .venv
uv venv --python 3.13 .venv

source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade pip setuptools wheel >/dev/null
uv pip install --python .venv/bin/python --editable ".[dev]" >/dev/null
uv pip install --python .venv/bin/python --upgrade ninja >/dev/null
uv pip install --python .venv/bin/python --upgrade "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128 >/dev/null
uv pip install --python .venv/bin/python --upgrade "transformers>=4.51" >/dev/null

python - <<'PY'
import shutil
import subprocess

import torch

print("torch_version=", torch.__version__)
print("torch_cuda_available=", torch.cuda.is_available())
print("torch_cuda_version=", torch.version.cuda)
print("gpu_name=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for tool in ("ptxas", "nvcc"):
    path = shutil.which(tool)
    print(f"{tool}_path=", path)
    if path:
        result = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
        output = result.stdout.strip() or result.stderr.strip()
        print(f"{tool}_version=", output.splitlines()[-1] if output else None)
PY

echo "node_version=$(node --version)"
if command -v codex >/dev/null 2>&1; then
  echo "codex_version=$(codex --version 2>/dev/null || true)"
fi
