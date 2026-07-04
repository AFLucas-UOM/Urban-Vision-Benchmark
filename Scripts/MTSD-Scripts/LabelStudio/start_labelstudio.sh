#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GROUP="${1:-${GROUP:-GRP-1}}"
PORT="${2:-${PORT:-8080}}"
HOST_URL="http://localhost:${PORT}"
DATASET_DIR="Datasets/${GROUP}"
DATA_DIR="Datasets/${GROUP}/labelstudio_output/LabelStudioData"

if [ ! -d "$DATASET_DIR" ]; then
  echo "Could not find dataset directory for ${GROUP} at ${DATASET_DIR}" >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

if command -v netstat >/dev/null 2>&1; then
  PIDS="$(netstat -ano | awk -v port=":${PORT}" '$0 ~ port && $0 ~ /LISTENING/ { print $NF }' | sort -u || true)"
  for pid in $PIDS; do
    if [ "$pid" != "0" ]; then
      echo "Stopping existing process on port ${PORT}: PID ${pid}"
      taskkill //PID "$pid" //F >/dev/null 2>&1 || true
    fi
  done
  [ -z "$PIDS" ] || sleep 2
fi

export DEBUG=false
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT="$PWD"
export LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD"

ENV_FILE="$DATA_DIR/.env"
touch "$ENV_FILE"
grep -v -E '^(LOCAL_FILES_SERVING_ENABLED|LOCAL_FILES_DOCUMENT_ROOT|LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED|LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT)=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
{
  cat "$ENV_FILE.tmp"
  echo "LOCAL_FILES_SERVING_ENABLED=true"
  echo "LOCAL_FILES_DOCUMENT_ROOT=$PWD"
  echo "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
  echo "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD"
} > "$ENV_FILE"
rm -f "$ENV_FILE.tmp"

echo "Starting Label Studio for ${GROUP}..."
echo "URL: $HOST_URL"
echo "Username: sample@example.com"
echo "Password: SampleAnnotations123!"
echo
echo "Keep this terminal open while using Label Studio. Press Ctrl+C to stop it."
echo

if command -v label-studio >/dev/null 2>&1; then
  label-studio start \
    --no-browser \
    --data-dir "$DATA_DIR" \
    --host "$HOST_URL" \
    --port "$PORT" \
    --username sample@example.com \
    --password 'SampleAnnotations123!' \
    --user-token sample-annotations-token \
    --enable-legacy-api-token
else
  uvx --python 3.12 label-studio start \
    --no-browser \
    --data-dir "$DATA_DIR" \
    --host "$HOST_URL" \
    --port "$PORT" \
    --username sample@example.com \
    --password 'SampleAnnotations123!' \
    --user-token sample-annotations-token \
    --enable-legacy-api-token
fi
