from __future__ import annotations

import pipeline

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from optimized_anno_img import CLASSES, IMAGE_SUFFIXES, STATUSES

# Every path below starts from the repo root.
ROOT = pipeline.REPO_ROOT
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels_obb"
DATASET_DIR = DATA_DIR / "dataset"

# Share of each folder kept back for validation.
VAL_SHARE = 0.15
BLOCK_SIZE = 4
SEED = 42

# Images where a hand covers the shape; never used.
EXCLUDED_IMAGES = (
    "circle/circle_0063",
    "hexagon/hexagon_0023",
    "hexagon/hexagon_0044",
)
EXPECTED_TOTAL = 334
EXPECTED_CLASSES = {0: 166, 1: 162, 2: 195, 3: 259, 4: 121}

# Load one JSON sidecar and check its fields.
def read_sidecar(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read sidecar ({exc})") from exc
    if data.get("status") not in STATUSES:
        raise ValueError(f"{path}: bad status {data.get('status')!r}")
    if not isinstance(data.get("objects"), list):
        raise ValueError(f"{path}: 'objects' is not a list")
    return data

# Each label has a class and four normalized corners.
def check_labels(text, count=None, where=""):
    lines = [line for line in text.splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"{where}: line {number} has {len(fields)} fields, expected 9")
        try:
            class_id, coords = int(fields[0]), [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{where}: line {number} is not numeric") from exc
        if class_id not in range(len(CLASSES)):
            raise ValueError(f"{where}: line {number} has class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"{where}: line {number} has a coordinate outside [0, 1]")
    if count is not None and len(lines) != count:
        raise ValueError(f"{where}: {len(lines)} labels for {count} objects")
    return lines

# Keep approved sidecars and find their image and label text.
def collect_samples(raw_dir, labels_dir, excluded_images=()):
    samples, excluded, approved = [], [], set()
    for sidecar in sorted(labels_dir.glob("*/*.json")):
        folder, stem = sidecar.parent.name, sidecar.stem
        name = f"{folder}/{stem}"
        data = read_sidecar(sidecar)
        if data["status"] != "approved":
            excluded.append((name, data["status"]))
            continue
        approved.add(name)
        if name in excluded_images:
            excluded.append((name, "hand blocks the shape"))
            continue

        candidates = [raw_dir / folder / f"{stem}{suffix}" for suffix in sorted(IMAGE_SUFFIXES)]
        image = next((path for path in candidates if path.is_file()), None)
        if image is None:
            raise ValueError(f"no image for sidecar: {raw_dir / folder / stem}")

        label_text = None
        if data["objects"]:
            label_path = sidecar.with_suffix(".txt")
            if not label_path.is_file():
                raise ValueError(f"{name}: {len(data['objects'])} objects but no .txt")
            label_text = label_path.read_text(encoding="utf-8")
            check_labels(label_text, len(data["objects"]), where=str(label_path))
        samples.append({"folder": folder, "stem": stem, "image": image,
                        "label_text": label_text})

    unknown = set(excluded_images) - approved
    if unknown:
        raise ValueError(f"EXCLUDED_IMAGES are not approved sidecars: {sorted(unknown)}")
    return samples, excluded

# Split inside each folder in blocks, so both sets look alike.
def split_samples(samples, val_share=VAL_SHARE, block_size=BLOCK_SIZE, seed=SEED):
    rng = random.Random(seed)
    by_folder = defaultdict(list)
    for sample in samples:
        by_folder[sample["folder"]].append(sample)

    train, val = [], []
    for folder in sorted(by_folder):
        group = sorted(by_folder[folder], key=lambda sample: sample["stem"])
        blocks = [group[start : start + block_size] for start in range(0, len(group), block_size)]
        rng.shuffle(blocks)
        chosen, target = [], round(len(group) * val_share)
        for block in blocks:
            if len(chosen) >= target:
                break
            chosen.extend(block)
        val_stems = {sample["stem"] for sample in chosen}
        train.extend(sample for sample in group if sample["stem"] not in val_stems)
        val.extend(sample for sample in group if sample["stem"] in val_stems)
    return train, val

# Copy images and labels into the YOLO folder layout.
def write_dataset(train, val, dataset_dir):
    for name in ("images", "labels"):
        shutil.rmtree(dataset_dir / name, ignore_errors=True)
    for split, samples in (("train", train), ("val", val)):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            shutil.copy2(sample["image"], image_dir / sample["image"].name)
            if sample["label_text"] is not None:
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{sample['stem']}.txt").write_text(
                    sample["label_text"], encoding="utf-8")
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}",
        encoding="utf-8",
    )

def check_dataset(dataset_dir, train, val, excluded_names,
                  expected_classes: dict[int, int] | None = EXPECTED_CLASSES):
    # Read the written dataset again and collect any mismatch.
    failures, per_class = [], Counter()
    for split, samples in (("train", train), ("val", val)):
        images = {path.stem for path in (dataset_dir / "images" / split).glob("*")}
        labels = {path.stem for path in (dataset_dir / "labels" / split).glob("*.txt")}
        if images != {sample["stem"] for sample in samples}:
            failures.append(f"{split}: {len(images)} images written, {len(samples)} planned")
        labelled = {sample["stem"] for sample in samples if sample["label_text"] is not None}
        if labels != labelled:
            failures.append(f"{split}: {len(labels)} labels written, {len(labelled)} expected")
        for path in sorted((dataset_dir / "labels" / split).glob("*.txt")):
            lines = check_labels(path.read_text(encoding="utf-8"), where=str(path))
            per_class.update(int(line.split()[0]) for line in lines)

    if expected_classes is not None and dict(per_class) != expected_classes:
        failures.append(
            f"per-class totals {dict(sorted(per_class.items()))} != expected {expected_classes}")
    stray = {path.stem for path in dataset_dir.rglob("*")
             if path.is_file() and path.stem in excluded_names}
    if stray:
        failures.append(f"excluded images written: {sorted(stray)}")

    config = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    if config.get("names") != dict(enumerate(CLASSES)):
        failures.append(f"data.yaml names are {config.get('names')!r}")
    if (config.get("path"), config.get("train"), config.get("val")) != (
            dataset_dir.as_posix(), "images/train", "images/val"):
        failures.append("data.yaml path/train/val mismatch")
    return failures

def build_dataset(raw_dir=RAW_DIR, labels_dir=LABELS_DIR, dataset_dir=DATASET_DIR,
                  val_share=VAL_SHARE, block_size=BLOCK_SIZE, seed=SEED,
                  excluded_images=EXCLUDED_IMAGES, expected_total: int | None = EXPECTED_TOTAL,
                  expected_classes: dict[int, int] | None = EXPECTED_CLASSES):
    # Collect, split, write, then check the new dataset.
    samples, excluded = collect_samples(raw_dir, labels_dir, excluded_images)
    if expected_total is not None and len(samples) != expected_total:
        raise ValueError(f"selected {len(samples)} images, expected {expected_total}")

    train, val = split_samples(samples, val_share, block_size, seed)
    write_dataset(train, val, dataset_dir)

    excluded_names = {name.split("/")[1] for name, _ in excluded}
    failures = check_dataset(dataset_dir, train, val, excluded_names, expected_classes)
    if failures:
        raise ValueError("dataset check failed:\n  " + "\n  ".join(failures))

    per_class = Counter(
        int(line.split()[0])
        for sample in samples if sample["label_text"]
        for line in sample["label_text"].splitlines() if line.strip()
    )
    return {"samples": samples, "train": train, "val": val, "excluded": excluded,
            "per_class": per_class,
            "backgrounds": sum(1 for sample in samples if sample["label_text"] is None)}

# Print counts per folder, class and split.
def print_report(summary, dataset_dir=DATASET_DIR):
    used = Counter(sample["folder"] for sample in summary["samples"])
    dropped = Counter(name.split("/")[0] for name, _ in summary["excluded"])
    print(f"selected {len(summary['samples'])} images, excluded {len(summary['excluded'])}:")
    for folder in sorted(set(used) | set(dropped)):
        print(f"  {folder:<10} used {used[folder]:>3}  excluded {dropped[folder]:>3}")
    for name, reason in summary["excluded"]:
        print(f"  {name:<24} {reason}")
    written = len(summary["train"]) + len(summary["val"])
    print(f"background samples: {summary['backgrounds']}")
    print(f"objects per class: {dict(sorted(summary['per_class'].items()))} "
          f"(sum {sum(summary['per_class'].values())})")
    print(f"split: train {len(summary['train'])}  val {len(summary['val'])}")
    print(f"images written: {written}  label files written: {written - summary['backgrounds']}")
    print(f"data.yaml: {(dataset_dir / 'data.yaml').read_text(encoding='utf-8')}")

# Build the dataset and stop if any check fails.
def main():
    try:
        summary = build_dataset()
    except ValueError as exc:
        raise SystemExit(f"prepare_dataset failed: {exc}") from None
    print_report(summary)
    print(f"dataset ready: {DATASET_DIR}")

if __name__ == "__main__":
    main()
