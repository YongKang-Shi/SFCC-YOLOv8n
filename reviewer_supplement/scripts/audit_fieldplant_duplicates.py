from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


WORK = Path(r"C:\Users\SYK\Documents\Codex\2026-10-30\bi-a\work")
PLANTDOC_IMAGES = Path(
    r"C:\Users\SYK\plantdoc-project\plantdoc-github-official-yolo\images"
)
SOURCE = Path(
    os.environ.get(
        "AUDIT_SOURCE",
        str(WORK / "datasets" / "FieldPlant" / "external-v1"),
    )
)
OUT = Path(
    os.environ.get(
        "AUDIT_OUT",
        str(WORK / "fieldplant-external-v2"),
    )
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BYTE_POPCOUNT = np.array(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(path: Path) -> tuple[int, int]:
    with Image.open(path) as source:
        image = source.convert("L")
        phash_image = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS))
        dhash_image = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS))
    coefficients = dct(
        dct(phash_image.astype(float), axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    low_frequency = coefficients[:8, :8].flatten()
    phash_bits = (
        low_frequency >= np.median(low_frequency[1:])
    ).astype(np.uint8)
    dhash_bits = (
        dhash_image[:, 1:] >= dhash_image[:, :-1]
    ).astype(np.uint8).flatten()
    phash = 0
    dhash = 0
    for bit in phash_bits:
        phash = (phash << 1) | int(bit)
    for bit in dhash_bits:
        dhash = (dhash << 1) | int(bit)
    return phash, dhash


def hamming_to_all(value: int, hashes: np.ndarray) -> np.ndarray:
    xor_values = np.bitwise_xor(hashes, np.uint64(value))
    return BYTE_POPCOUNT[
        xor_values.view(np.uint8).reshape(-1, 8)
    ].sum(axis=1)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    source_images = sorted(
        path
        for path in (SOURCE / "images" / "test").glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    plantdoc_paths = sorted(
        path
        for path in [
            *PLANTDOC_IMAGES.glob("train/*"),
            *PLANTDOC_IMAGES.glob("test/*"),
        ]
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    plantdoc_sha = {sha256(path): path for path in plantdoc_paths}
    plantdoc_phash = np.empty(len(plantdoc_paths), dtype=np.uint64)
    plantdoc_dhash = np.empty(len(plantdoc_paths), dtype=np.uint64)
    for index, path in enumerate(plantdoc_paths):
        plantdoc_phash[index], plantdoc_dhash[index] = perceptual_hashes(path)

    rows = []
    retained = []
    retained_sha = {}
    retained_phash: list[int] = []
    retained_dhash: list[int] = []
    retained_paths: list[Path] = []

    for image_path in source_images:
        exact_hash = sha256(image_path)
        phash, dhash = perceptual_hashes(image_path)
        plantdoc_exact = plantdoc_sha.get(exact_hash)
        plantdoc_phash_distance = int(hamming_to_all(phash, plantdoc_phash).min())
        plantdoc_dhash_distance = int(hamming_to_all(dhash, plantdoc_dhash).min())
        internal_exact = retained_sha.get(exact_hash)
        if retained_phash:
            internal_phash_distance = int(
                hamming_to_all(phash, np.asarray(retained_phash, dtype=np.uint64)).min()
            )
            internal_dhash_distance = int(
                hamming_to_all(dhash, np.asarray(retained_dhash, dtype=np.uint64)).min()
            )
        else:
            internal_phash_distance = 64
            internal_dhash_distance = 64

        cross_duplicate = (
            plantdoc_exact is not None
            or plantdoc_phash_distance <= 8
            or plantdoc_dhash_distance <= 8
        )
        internal_duplicate = (
            internal_exact is not None
            or internal_phash_distance <= 8
            or internal_dhash_distance <= 8
        )
        excluded = cross_duplicate or internal_duplicate
        rows.append(
            {
                "image": image_path.name,
                "plantdoc_exact": bool(plantdoc_exact),
                "plantdoc_exact_match": (
                    str(plantdoc_exact) if plantdoc_exact is not None else ""
                ),
                "plantdoc_phash_min": plantdoc_phash_distance,
                "plantdoc_dhash_min": plantdoc_dhash_distance,
                "internal_exact": bool(internal_exact),
                "internal_phash_min": internal_phash_distance,
                "internal_dhash_min": internal_dhash_distance,
                "excluded": excluded,
                "exclusion_reason": (
                    "plantdoc_duplicate"
                    if cross_duplicate
                    else "internal_duplicate" if internal_duplicate else ""
                ),
            }
        )
        if excluded:
            continue
        retained.append(image_path)
        retained_sha[exact_hash] = image_path
        retained_phash.append(phash)
        retained_dhash.append(dhash)
        retained_paths.append(image_path)

    images_out = OUT / "images" / "test"
    labels_out = OUT / "labels" / "test"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    class_counts = Counter()
    for image_path in retained:
        label_path = SOURCE / "labels" / "test" / f"{image_path.stem}.txt"
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_counts[int(line.split()[0])] += 1

    (OUT / "duplicate_audit.csv").write_text(
        "\n".join(
            [
                ",".join(rows[0].keys()),
                *[
                    ",".join(
                        str(value)
                        .replace(",", ";")
                        .replace("\n", " ")
                        for value in row.values()
                    )
                    for row in rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_yaml = (SOURCE / "data.yaml").read_text(encoding="utf-8")
    data_yaml = data_yaml.replace(
        str(SOURCE).replace("\\", "/"), str(OUT).replace("\\", "/")
    )
    (OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report = {
        "source": os.environ.get(
            "AUDIT_SOURCE_NAME",
            "FieldPlant Kaggle dataset, Roboflow version 11",
        ),
        "plantdoc_files_compared": len(plantdoc_paths),
        "source_mapped_images": len(source_images),
        "excluded_plantdoc_duplicates": sum(
            row["exclusion_reason"] == "plantdoc_duplicate" for row in rows
        ),
        "excluded_internal_duplicates": sum(
            row["exclusion_reason"] == "internal_duplicate" for row in rows
        ),
        "retained_images": len(retained),
        "retained_boxes": sum(class_counts.values()),
        "retained_box_counts": {
            str(key): class_counts[key] for key in sorted(class_counts)
        },
        "duplicate_thresholds": {
            "exact": "SHA-256",
            "phash_hamming_max": 8,
            "dhash_hamming_max": 8,
        },
    }
    (OUT / "filter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
