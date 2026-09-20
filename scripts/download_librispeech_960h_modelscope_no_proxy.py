#!/usr/bin/env python3

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "LibriSpeech" / "modelscope_parquet"
INDEX_ROOT = OUTPUT_ROOT / ".indexes"
API_URL = "https://www.modelscope.cn/api/v1/datasets/openslr/librispeech_asr/repo/tree"
FILE_URL = "https://www.modelscope.cn/datasets/openslr/librispeech_asr/resolve/master"
SUBSETS = (
    "all/train.clean.100",
    "all/train.clean.360",
    "all/train.other.500",
    "all/validation.clean",
    "all/validation.other",
    "all/test.clean",
    "all/test.other",
)
WORKERS = 4


def direct_env():
    env = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        env.pop(name, None)
    return env


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_index(subset):
    index_path = INDEX_ROOT / (subset.replace("/", "__") + ".json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API_URL}?Revision=master&Root={quote(subset, safe='/')}"
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--output",
        str(index_path),
        url,
    ]
    subprocess.run(command, check=True, env=direct_env())
    payload = json.loads(index_path.read_text())
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope index failed for {subset}: {payload}")
    return [item for item in payload["Data"]["Files"] if item["Type"] == "blob"]


def download_one(item):
    relative_path = item["Path"]
    expected_size = int(item["Size"])
    expected_sha256 = item["Sha256"]
    destination = OUTPUT_ROOT / relative_path
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_size:
        if not expected_sha256 or sha256(destination) == expected_sha256:
            return f"already complete: {relative_path}"

    url = f"{FILE_URL}/{quote(relative_path, safe='/')}"
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "30",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True, env=direct_env())

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {relative_path}: {actual_size} != {expected_size}"
        )
    if expected_sha256:
        actual_sha256 = sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"sha256 mismatch for {relative_path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    os.replace(partial, destination)
    return f"downloaded and verified: {relative_path}"


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_items = []
    for subset in SUBSETS:
        items = fetch_index(subset)
        subset_bytes = sum(int(item["Size"]) for item in items)
        print(
            f"indexed {subset}: {len(items)} files, "
            f"{subset_bytes / 1_000_000_000:.2f} GB",
            flush=True,
        )
        all_items.extend(items)

    total_bytes = sum(int(item["Size"]) for item in all_items)
    print(
        f"download plan: {len(all_items)} files, "
        f"{total_bytes / 1_000_000_000:.2f} GB, workers={WORKERS}",
        flush=True,
    )

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, item): item for item in all_items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append((item["Path"], str(error)))
                print(f"FAILED: {item['Path']}: {error}", file=sys.stderr, flush=True)

    if failures:
        print(f"download failed for {len(failures)} files", file=sys.stderr)
        for path, error in failures:
            print(f"  {path}: {error}", file=sys.stderr)
        return 1

    marker = OUTPUT_ROOT / "LIBRISPEECH_960H_PARQUET_COMPLETE"
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))
    print("LIBRISPEECH_960H_PARQUET_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
