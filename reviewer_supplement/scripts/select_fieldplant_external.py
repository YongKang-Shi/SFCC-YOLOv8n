from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


ARCHIVE_SIZE = 5_913_835_395
TAIL_START = 5_911_280_000
CLASS_MAP = {
    8: 7,    # Corn Gray leaf spot
    19: 8,   # Corn leaf blight
    20: 9,   # Corn rust leaf
    24: 25,  # Tomato healthy -> Tomato leaf
    25: 23,  # Tomato leaf mosaic virus
    26: 24,  # Tomato leaf yellow virus
}
MAX_IMAGES_PER_CLASS = 200


class CachedTailRangeFile(io.RawIOBase):
    def __init__(self, url: str, tail_path: Path) -> None:
        self.url = url
        self.position = 0
        self.tail_path = tail_path
        self.tail_start = TAIL_START
        self.tail = tail_path.open("rb")
        self.tail_size = tail_path.stat().st_size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = ARCHIVE_SIZE + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= ARCHIVE_SIZE:
            return b""
        if size < 0:
            size = ARCHIVE_SIZE - self.position
        size = min(size, ARCHIVE_SIZE - self.position)
        end = self.position + size - 1
        if self.position >= self.tail_start and end < self.tail_start + self.tail_size:
            self.tail.seek(self.position - self.tail_start)
            data = self.tail.read(size)
        else:
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={self.position}-{end}",
                    "User-Agent": "Mozilla/5.0",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        self.position += len(data)
        return data

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.tail.close()
        super().close()


def merged_ranges(rows: list[dict], max_gap: int = 8 * 1024 * 1024) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["header_offset"])
    groups = []
    for row in ordered:
        start = row["header_offset"]
        end = start + row["compress_size"] + 1024
        if not groups or start - groups[-1]["end"] > max_gap:
            groups.append({"start": start, "end": end, "entries": 1})
        else:
            groups[-1]["end"] = max(groups[-1]["end"], end)
            groups[-1]["entries"] += 1
    for group in groups:
        group["bytes"] = group["end"] - group["start"]
    return groups


def main() -> None:
    url = os.environ.get("FIELDPLANT_ARCHIVE_URL")
    if not url:
        raise SystemExit("FIELDPLANT_ARCHIVE_URL is not set")
    root = Path(os.environ["FIELDPLANT_ROOT"])
    remote = CachedTailRangeFile(url, root / "tail.bin")
    image_by_stem = {}
    label_by_stem = {}

    with zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.filename.startswith("train/images/"):
                image_by_stem[Path(info.filename).stem] = info
            elif info.filename.startswith("train/labels/"):
                label_by_stem[Path(info.filename).stem] = info

        selected = []
        excluded_multiclass = 0
        for stem, label_info in sorted(label_by_stem.items()):
            ids = []
            lines = []
            for line in archive.read(label_info).decode("utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                class_id = int(float(fields[0]))
                ids.append(class_id)
                if class_id in CLASS_MAP:
                    lines.append(
                        f"{CLASS_MAP[class_id]} " + " ".join(fields[1:])
                    )
            if not ids or not set(ids).issubset(CLASS_MAP):
                if set(ids) & set(CLASS_MAP):
                    excluded_multiclass += 1
                continue
            image_info = image_by_stem.get(stem)
            if image_info is None:
                raise RuntimeError(f"Missing image for label stem: {stem}")
            selected.append(
                {
                    "stem": stem,
                    "source_ids": " ".join(str(value) for value in sorted(set(ids))),
                    "mapped_labels": "\n".join(lines),
                    "header_offset": image_info.header_offset,
                    "compress_size": image_info.compress_size,
                    "file_size": image_info.file_size,
                    "image_name": Path(image_info.filename).name,
                }
            )

    selected_by_class: dict[str, list[dict]] = {}
    for row in selected:
        selected_by_class.setdefault(row["source_ids"], []).append(row)
    balanced = []
    for class_key, group in sorted(selected_by_class.items()):
        balanced.extend(
            sorted(
                group,
                key=lambda row: hashlib.sha256(
                    row["stem"].encode("utf-8")
                ).hexdigest(),
            )[:MAX_IMAGES_PER_CLASS]
        )

    ranges = merged_ranges(balanced)
    class_counts = Counter()
    box_counts = Counter()
    for row in balanced:
        ids = [int(value) for value in row["source_ids"].split()]
        class_counts.update(ids)
        box_counts.update(
            int(line.split()[0]) for line in row["mapped_labels"].splitlines()
        )

    out = root / "selection_balanced.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balanced[0]))
        writer.writeheader()
        writer.writerows(balanced)
    audit = {
        "selection_policy": (
            "All class-exact images, with a deterministic SHA-256 filename "
            f"cap of {MAX_IMAGES_PER_CLASS} images per source class."
        ),
        "available_class_exact_images": len(selected),
        "selected_images": len(balanced),
        "selected_boxes": sum(box_counts.values()),
        "source_class_image_counts": {
            str(key): class_counts[key] for key in sorted(class_counts)
        },
        "mapped_box_counts": {
            str(key): box_counts[key] for key in sorted(box_counts)
        },
        "all_class_ids": {
            str(key): CLASS_MAP[key] for key in sorted(CLASS_MAP)
        },
        "excluded_images_with_mapped_and_unmapped_classes": excluded_multiclass,
        "download_ranges": ranges,
        "download_range_bytes": sum(group["bytes"] for group in ranges),
    }
    (root / "selection_balanced_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
