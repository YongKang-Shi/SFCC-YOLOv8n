from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ARCHIVE_SIZE = 5_913_835_395
TAIL_START = 5_911_280_000
PLANTDOC_NAMES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato two spotted spider mites leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "grape leaf black rot",
    "grape leaf",
]


class CachedRangesFile(io.RawIOBase):
    def __init__(
        self,
        url: str,
        cached_ranges: list[tuple[int, int, Path]],
        tail_start: int,
        tail_path: Path,
    ) -> None:
        self.url = url
        self.position = 0
        self.cached_ranges = cached_ranges
        self.tail_start = tail_start
        self.tail_size = tail_path.stat().st_size
        self.cached_ranges.append(
            (tail_start, tail_start + self.tail_size - 1, tail_path)
        )

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
        for start, range_end, path in self.cached_ranges:
            if self.position >= start and end <= range_end:
                with path.open("rb") as handle:
                    handle.seek(self.position - start)
                    data = handle.read(size)
                self.position += len(data)
                return data
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


def main() -> None:
    url = os.environ.get("FIELDPLANT_ARCHIVE_URL")
    if not url:
        raise SystemExit("FIELDPLANT_ARCHIVE_URL is not set")
    root = Path(os.environ["FIELDPLANT_ROOT"])
    out = root / "external-v1"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    images_out = out / "images" / "test"
    labels_out = out / "labels" / "test"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    with (root / "selection_balanced.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    audit = json.loads(
        (root / "selection_balanced_audit.json").read_text(encoding="utf-8")
    )
    cache = root / "range-cache-balanced"
    cached_ranges = []
    for index, group in enumerate(audit["download_ranges"]):
        path = cache / f"group-{index:02d}.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        cached_ranges.append(
            (group["start"], group["end"] - 1, path)
        )

    remote = CachedRangesFile(
        url,
        cached_ranges,
        TAIL_START,
        root / "tail.bin",
    )
    class_counts = Counter()
    box_counts = Counter()
    dimensions = Counter()
    with zipfile.ZipFile(remote) as archive:
        for row in rows:
            entry_name = f"train/images/{row['image_name']}"
            info = archive.getinfo(entry_name)
            image_bytes = archive.read(info)
            image_path = images_out / row["image_name"]
            image_path.write_bytes(image_bytes)
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1

            label_path = labels_out / f"{Path(row['image_name']).stem}.txt"
            label_path.write_text(
                row["mapped_labels"].strip() + "\n",
                encoding="utf-8",
            )
            class_counts[row["source_ids"]] += 1
            box_counts.update(
                int(line.split()[0])
                for line in row["mapped_labels"].splitlines()
            )

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "path: "
        + str(out).replace("\\", "/")
        + "\n"
        + "train: images/test\n"
        + "val: images/test\n"
        + "test: images/test\n"
        + "names:\n"
        + "\n".join(
            f"  {index}: {name}" for index, name in enumerate(PLANTDOC_NAMES)
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "source": "FieldPlant Kaggle dataset, Roboflow version 11",
        "source_paper_doi": "10.1109/ACCESS.2023.3263042",
        "source_license": "CC BY 4.0",
        "selection_policy": audit["selection_policy"],
        "selected_images": len(rows),
        "selected_boxes": sum(box_counts.values()),
        "images_by_source_class": {
            key: class_counts[key] for key in sorted(class_counts)
        },
        "boxes_by_mapped_class": {
            str(key): box_counts[key] for key in sorted(box_counts)
        },
        "image_dimensions": dict(sorted(dimensions.items())),
    }
    (out / "filter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
