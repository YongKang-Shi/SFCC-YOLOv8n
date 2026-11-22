from __future__ import annotations

import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def fetch_range(url: str, start: int, end: int, target: Path) -> None:
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return
    last_error = None
    for _ in range(4):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Mozilla/5.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if len(data) != expected:
                raise RuntimeError(
                    f"Short range {start}-{end}: {len(data)} != {expected}"
                )
            target.write_bytes(data)
            return
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Failed range {start}-{end}: {last_error}")


def main() -> None:
    url = os.environ.get("FIELDPLANT_ARCHIVE_URL")
    if not url:
        raise SystemExit("FIELDPLANT_ARCHIVE_URL is not set")
    root = Path(os.environ["FIELDPLANT_ROOT"])
    audit = json.loads(
        (root / "selection_balanced_audit.json").read_text(encoding="utf-8")
    )
    cache = root / "range-cache-balanced"
    cache.mkdir(parents=True, exist_ok=True)

    tasks = []
    for group_index, group in enumerate(audit["download_ranges"]):
        group_dir = cache / f"group-{group_index:02d}"
        group_dir.mkdir(exist_ok=True)
        start = group["start"]
        end = group["end"] - 1
        chunk_index = 0
        while start <= end:
            chunk_end = min(end, start + CHUNK_SIZE - 1)
            target = group_dir / f"chunk-{chunk_index:03d}.bin"
            tasks.append((start, chunk_end, target))
            start = chunk_end + 1
            chunk_index += 1

    print(f"RANGES={len(audit['download_ranges'])} CHUNKS={len(tasks)}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_range, url, start, end, target): target
            for start, end, target in tasks
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 5 == 0 or completed == len(futures):
                print(f"COMPLETED={completed}/{len(futures)}", flush=True)

    for group_index, group in enumerate(audit["download_ranges"]):
        group_dir = cache / f"group-{group_index:02d}"
        chunks = sorted(group_dir.glob("chunk-*.bin"))
        group_path = cache / f"group-{group_index:02d}.bin"
        with group_path.open("wb") as output:
            for chunk in chunks:
                with chunk.open("rb") as source:
                    while data := source.read(1024 * 1024):
                        output.write(data)
        if group_path.stat().st_size != group["bytes"]:
            raise RuntimeError(
                f"Group {group_index} size mismatch: "
                f"{group_path.stat().st_size} != {group['bytes']}"
            )
    print(f"DOWNLOADED={sum(group['bytes'] for group in audit['download_ranges'])}")


if __name__ == "__main__":
    main()
