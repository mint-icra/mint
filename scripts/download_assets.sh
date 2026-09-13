#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
MINT_CHECKPOINT_URL="${MINT_CHECKPOINT_URL:-}"
HF_MIRROR_ENDPOINT="${MINT_HF_MIRROR:-${HF_ENDPOINT:-https://hf-mirror.com}}"
DOWNLOAD_CONNECT_TIMEOUT="${MINT_DOWNLOAD_CONNECT_TIMEOUT:-15}"
DOWNLOAD_SPEED_LIMIT="${MINT_DOWNLOAD_SPEED_LIMIT:-1024}"
DOWNLOAD_SPEED_TIME="${MINT_DOWNLOAD_SPEED_TIME:-30}"
MINT_SHA256="7b2f0aa5dfd00c271bb2f12c841ccfcc70e81e4052d413eacb5fb42a1bcc36c8"

mkdir -p "$CHECKPOINT_DIR"

verify() {
  local destination="$1"
  local sha256="$2"
  printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check --status
}

download_once() {
  local url="$1"
  local partial="$2"
  local retries="$3"
  shift 3

  curl \
    --fail \
    --location \
    --retry "$retries" \
    --retry-all-errors \
    --retry-delay 2 \
    --continue-at - \
    --connect-timeout "$DOWNLOAD_CONNECT_TIMEOUT" \
    --speed-limit "$DOWNLOAD_SPEED_LIMIT" \
    --speed-time "$DOWNLOAD_SPEED_TIME" \
    --progress-bar \
    --show-error \
    "$@" \
    --output "$partial" \
    "$url"
}

download_verified() {
  local url="$1"
  local destination="$2"
  local sha256="$3"
  shift 3
  local partial="${destination}.part"

  if [[ -s "$destination" ]] && verify "$destination" "$sha256"; then
    echo "Verified existing asset: ${destination#"$PROJECT_DIR/"}"
    return
  fi

  echo "Downloading ${destination#"$PROJECT_DIR/"}"
  if [[ -s "$partial" ]]; then
    echo "Resuming partial download ($(stat -c '%s' "$partial") bytes already present)."
  fi

  if [[ "$url" == https://huggingface.co/* ]]; then
    if ! download_once "$url" "$partial" 0 "$@"; then
      local mirror_url="${HF_MIRROR_ENDPOINT%/}/${url#https://huggingface.co/}"
      if [[ "$mirror_url" == "$url" ]]; then
        echo "Hugging Face download failed; partial data remains at ${partial#"$PROJECT_DIR/"}." >&2
        return 1
      fi
      echo "Official Hugging Face endpoint stalled or failed; switching to $HF_MIRROR_ENDPOINT"
      # These release assets are public; do not forward private HF tokens to a mirror.
      if ! download_once "$mirror_url" "$partial" 5; then
        echo "Mirror download failed; partial data remains at ${partial#"$PROJECT_DIR/"}." >&2
        echo "Run this script again to resume it." >&2
        return 1
      fi
    fi
  elif ! download_once "$url" "$partial" 5 "$@"; then
    echo "Download failed; partial data remains at ${partial#"$PROJECT_DIR/"}." >&2
    echo "Run this script again to resume it." >&2
    return 1
  fi

  if ! verify "$partial" "$sha256"; then
    echo "Checksum verification failed: ${partial#"$PROJECT_DIR/"}" >&2
    rm -f "$partial"
    return 1
  fi
  mv -f "$partial" "$destination"
}

download_mint_model() {
  local destination="$CHECKPOINT_DIR/model.safetensors"
  local hf_header=()
  local hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

  if [[ -s "$destination" ]] && verify "$destination" "$MINT_SHA256"; then
    echo "Verified existing asset: ${destination#"$PROJECT_DIR/"}"
    return
  fi

  if [[ -z "$MINT_CHECKPOINT_URL" ]]; then
    echo "No checkpoint URL is embedded in the anonymous review repository." >&2
    echo "Set MINT_CHECKPOINT_URL to the anonymous reviewer artifact URL," >&2
    echo "or place the verified file at checkpoints/model.safetensors." >&2
    return 1
  fi

  if [[ -n "$hf_token" ]]; then
    hf_header=(--header "Authorization: Bearer $hf_token")
  fi

  if download_verified "$MINT_CHECKPOINT_URL" "$destination" "$MINT_SHA256" "${hf_header[@]}"; then
    return
  fi

  return 1
}

echo "[1/1] MINT model checkpoint"
download_mint_model

echo "MINT model checkpoint is ready."
echo "MANO is not downloaded. Follow docs/installation.md and accept its license."
