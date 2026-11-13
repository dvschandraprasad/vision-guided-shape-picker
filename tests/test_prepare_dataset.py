"""Headless tests for prepare_dataset.py.

Every test builds a synthetic corpus in tmp_path (a handful of tiny PNGs plus
sidecars), so the real data/raw/ and data/labels_obb/ are never read or written.

Run:
    .venv/Scripts/python.exe -m pytest test_prepare_dataset.py -q
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import prepare_dataset as prep

# Maps each class name to its number for tests.
CLASS_IDS = {name: index for index, name in enumerate(prep.CLASSES)}


def make_sample(raw_dir, labels_dir, folder, stem, status="approved", objects=None, label=True):
    """Write one image, one sidecar and (optionally) one .txt into the sandbox."""
    image_path = raw_dir / folder / f"{stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a tiny blank image so tests run fast.
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))

    # Default object: one circle with pixel corners.
    if objects is None:
        objects = [{"class_id": CLASS_IDS["circle"],
                    "corners": [[4, 4], [20, 4], [20, 20], [4, 20]]}]
    sidecar = {"status": status, "objects": objects, "pending_points": []}
    sidecar_path = labels_dir / folder / f"{stem}.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    if label and objects:
        (sidecar_path.with_suffix(".txt")).write_text(
            "\n".join(
                f"{obj['class_id']} 0.100000 0.100000 0.200000 0.100000 "
                "0.200000 0.200000 0.100000 0.200000"
                for obj in objects
            )
            + "\n",
            encoding="utf-8",
        )
    return sidecar_path


@pytest.fixture
def corpus(tmp_path):
    """An empty data/raw/ + data/labels_obb/ pair rooted in tmp_path."""
    raw = tmp_path / "raw"
    labels = tmp_path / "labels_obb"
    raw.mkdir()
    labels.mkdir()
    return raw, labels


def build(corpus, dataset, **settings):
    """Build a dataset from the sandbox corpus with the frozen audit checks off."""
    raw, labels = corpus
    settings.setdefault("excluded_images", ())   # the real corpus' names do not exist here
    return prep.build_dataset(
        raw_dir=raw,
        labels_dir=labels,
        dataset_dir=dataset,
        expected_total=None,
        expected_classes=None,
        **settings,
    )


def test_block_of_consecutive_stems_never_splits(corpus):
    raw, labels = corpus
    for index in range(1, 21):
        make_sample(raw, labels, "circle", f"circle_{index:04d}")
    samples, _ = prep.collect_samples(raw, labels, ())
    # Blocks of four neighbouring stems stay together.
    train, val = prep.split_samples(samples, val_share=0.2, block_size=4, seed=7)

    train_stems = {sample["stem"] for sample in train}
    val_stems = {sample["stem"] for sample in val}
    for index in range(0, 20, 4):
        block = {f"circle_{number:04d}" for number in range(index + 1, index + 5)}
        assert block <= train_stems or block <= val_stems
    assert not (train_stems & val_stems)


def test_val_fraction_stays_within_tolerance(corpus):
    raw, labels = corpus
    for index in range(1, 61):
        make_sample(raw, labels, "circle", f"circle_{index:04d}")
    samples, _ = prep.collect_samples(raw, labels, ())
    train, val = prep.split_samples(samples, val_share=0.15, block_size=4, seed=42)
    share = len(val) / (len(train) + len(val))
    assert 0.12 <= share <= 0.20


def test_same_seed_same_split_different_seed_differs(corpus):
    raw, labels = corpus
    for index in range(1, 41):
        make_sample(raw, labels, "circle", f"circle_{index:04d}")
    samples, _ = prep.collect_samples(raw, labels, ())
    first = prep.split_samples(samples, block_size=4, seed=1)
    again = prep.split_samples(samples, block_size=4, seed=1)
    other = prep.split_samples(samples, block_size=4, seed=2)
    assert {s["stem"] for s in first[1]} == {s["stem"] for s in again[1]}
    assert {s["stem"] for s in first[1]} != {s["stem"] for s in other[1]}


# Label line: class id then eight corner numbers.
def test_label_validation_accepts_nine_fields_and_rejects_the_rest():
    good = "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000 0.100000 0.200000"
    assert len(prep.check_labels(good + "\n")) == 1

    with pytest.raises(ValueError):
        prep.check_labels("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1")
    with pytest.raises(ValueError):
        prep.check_labels(good + " 0.3")
    with pytest.raises(ValueError):
        prep.check_labels("5 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2")
    with pytest.raises(ValueError):
        prep.check_labels("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1.2")
    with pytest.raises(ValueError):
        prep.check_labels(good, count=2)


def test_approved_with_objects_but_no_label_raises(corpus):
    raw, labels = corpus
    make_sample(raw, labels, "circle", "circle_0001", label=False)
    with pytest.raises(ValueError, match="no .txt"):
        prep.collect_samples(raw, labels, ())


def test_background_gets_image_and_no_label(corpus, tmp_path):
    raw, labels = corpus
    make_sample(raw, labels, "circle", "circle_0001", objects=[])
    summary = build(corpus, tmp_path / "dataset")

    assert summary["backgrounds"] == 1
    written = list((tmp_path / "dataset").rglob("circle_0001.png"))
    assert len(written) == 1
    assert not list((tmp_path / "dataset" / "labels").rglob("circle_0001.txt"))


def test_data_yaml_matches_the_contract(corpus, tmp_path):
    raw, labels = corpus
    make_sample(raw, labels, "circle", "circle_0001")
    dataset = tmp_path / "dataset"
    build(corpus, dataset)

    expected = (
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: circle\n"
        "  1: square\n"
        "  2: hexagon\n"
        "  3: star\n"
        "  4: triangle\n"
    )
    assert (dataset / "data.yaml").read_text(encoding="utf-8") == expected


def test_flagged_and_excluded_stems_never_appear(corpus, tmp_path):
    raw, labels = corpus
    make_sample(raw, labels, "circle", "circle_0001")
    make_sample(raw, labels, "circle", "circle_0002", status="flagged")
    make_sample(raw, labels, "hexagon", "hexagon_0001", objects=[])

    # Leave this image out by folder and stem.
    excluded = ("hexagon/hexagon_0001",)
    summary = build(corpus, tmp_path / "dataset", excluded_images=excluded)
    dataset = tmp_path / "dataset"

    assert {sample["stem"] for sample in summary["samples"]} == {"circle_0001"}
    stems = {path.stem for path in dataset.rglob("*") if path.is_file()}
    assert "circle_0002" not in stems
    assert "hexagon_0001" not in stems
