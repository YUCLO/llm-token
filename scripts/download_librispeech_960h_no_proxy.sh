#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_dir="${workspace_root}/data/LibriSpeech/archives"
extract_dir="${workspace_root}/data/LibriSpeech/extracted"
base_url="https://www.openslr.org/resources/12"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy

mkdir -p "${download_dir}" "${extract_dir}"

archives=(
  train-clean-100.tar.gz
  train-clean-360.tar.gz
  train-other-500.tar.gz
  dev-clean.tar.gz
  dev-other.tar.gz
  test-clean.tar.gz
  test-other.tar.gz
)

wget --no-proxy --continue --directory-prefix="${download_dir}" \
  "${base_url}/md5sum.txt"

for archive in "${archives[@]}"; do
  wget --no-proxy --continue --directory-prefix="${download_dir}" \
    "${base_url}/${archive}"
done

(
  cd "${download_dir}"
  for archive in "${archives[@]}"; do
    expected_line="$(grep "  ${archive}$" md5sum.txt)"
    printf '%s\n' "${expected_line}" | md5sum --check --status -
    printf 'checksum OK: %s\n' "${archive}"
  done
)

for archive in "${archives[@]}"; do
  tar -xzf "${download_dir}/${archive}" -C "${extract_dir}"
done

printf 'LIBRISPEECH_960H_COMPLETE\n'
