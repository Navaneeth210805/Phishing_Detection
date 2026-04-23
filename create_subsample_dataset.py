#!/usr/bin/env python3
"""Build a combined, stratified, and balanced dataset for multiclass + binary training.

Workflow:
1) Merge existing train/test feature chunks.
2) Remap multiclass labels to fixed known targets + one unknown class.
3) Downsample unknown to a strict cap.
4) Stratified 80/20 split per target class.
5) Enforce exact benign/phishing balance inside each split.
6) Write new train/test chunks and y11 sidecars.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_LABEL_SUFFIX = "_y11.npz"
DEFAULT_KNOWN_TARGETS = [
    "facebook",
    "meta",
    "usps",
    "at&t",
    "robinhood",
    "whatsapp",
    "booking",
    "instagram",
    "naver",
]

logging.basicConfig(level=logging.INFO, format="[create_subsample] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_target_name(value: str) -> str:
    return str(value).strip().lower()


def _find_chunks(source_dir: Path, split: str) -> List[Path]:
    return sorted(
        p
        for p in source_dir.glob(f"{split}_domfeat_chunk_*.npz")
        if p.suffix == ".npz" and not p.name.endswith(DEFAULT_LABEL_SUFFIX)
    )


def _load_y11(sidecar_path: Path) -> np.ndarray:
    with np.load(sidecar_path, allow_pickle=False) as data:
        if "y11" in data:
            return np.asarray(data["y11"], dtype=np.int16)
        if "y" in data:
            return np.asarray(data["y"], dtype=np.int16)
    raise ValueError(f"Missing y11/y in sidecar {sidecar_path}")


def _load_mapping(mapping_path: Path) -> Dict[str, object]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def _build_known_id_map(mapping: Dict[str, object], known_targets: Sequence[str]) -> Tuple[Dict[int, int], List[str], int]:
    top_targets = [str(v) for v in mapping.get("top_targets", [])]
    if not top_targets:
        raise ValueError("Mapping has no top_targets")

    norm_to_orig_id: Dict[str, int] = {}
    for i, name in enumerate(top_targets):
        norm_to_orig_id[_normalize_target_name(name)] = int(i)

    known_id_map: Dict[int, int] = {}
    class_names: List[str] = []
    missing: List[str] = []
    for new_id, raw_name in enumerate(known_targets):
        name = _normalize_target_name(raw_name)
        if name not in norm_to_orig_id:
            missing.append(raw_name)
            continue
        known_id_map[int(norm_to_orig_id[name])] = int(new_id)
        class_names.append(name)

    if missing:
        raise ValueError(f"Known targets missing in mapping top_targets: {missing}")

    unknown_new_id = len(class_names)
    class_names.append("unknown_agg")
    return known_id_map, class_names, unknown_new_id


def _collect_rows(
    chunk_files: Sequence[Path],
    label_suffix: str,
    known_id_map: Dict[int, int],
    unknown_new_id: int,
    binary_from_mapping: bool,
) -> Tuple[Dict[int, List[Tuple[Path, int, int, int]]], Counter, Counter]:
    """Collect rows as tuples: (chunk_path, row_idx, mapped_class_id, y_bin)."""
    rows_by_class: Dict[int, List[Tuple[Path, int, int, int]]] = defaultdict(list)
    class_dist: Counter = Counter()
    bin_dist: Counter = Counter()

    for chunk_path in tqdm(chunk_files, desc="Scanning combined chunks"):
        sidecar = chunk_path.with_name(chunk_path.stem + label_suffix)
        if not sidecar.exists():
            logger.warning(f"Missing sidecar: {sidecar}")
            continue

        with np.load(chunk_path, allow_pickle=False) as data:
            y_bin = None
            if not binary_from_mapping:
                if "y" not in data.files:
                    raise ValueError(f"Missing binary y in {chunk_path}")
                y_bin = np.asarray(data["y"], dtype=np.int16)

        y11 = _load_y11(sidecar)
        if y_bin is not None and len(y11) != len(y_bin):
            raise ValueError(f"Length mismatch in {chunk_path}: y11={len(y11)} y={len(y_bin)}")

        for row_idx, orig_cls in enumerate(y11.tolist()):
            orig_cls_i = int(orig_cls)
            mapped = int(known_id_map.get(orig_cls_i, unknown_new_id))
            # Stage-1 binary target: known brand => phishing(1), unknown => benign(0).
            if binary_from_mapping:
                yb = int(0 if mapped == unknown_new_id else 1)
            else:
                yb = int(y_bin[row_idx])
            rows_by_class[mapped].append((chunk_path, int(row_idx), mapped, yb))
            class_dist[mapped] += 1
            bin_dist[yb] += 1

    return rows_by_class, class_dist, bin_dist


def _choose_class_targets(
    class_dist: Counter,
    unknown_id: int,
    unknown_cap: int,
    min_per_known: int,
    known_balance_mode: str,
    per_class_caps: Dict[int, int],
) -> Dict[int, int]:
    known_counts = [int(v) for k, v in class_dist.items() if int(k) != unknown_id and int(v) > 0]
    if not known_counts:
        raise ValueError("No known-class rows found")

    if known_balance_mode == "min":
        known_target = int(min(known_counts))
    elif known_balance_mode == "max":
        known_target = int(max(known_counts))
    elif known_balance_mode == "none":
        known_target = -1
    else:
        known_target = int(np.median(np.asarray(known_counts, dtype=np.int64)))

    known_target = max(min_per_known, known_target) if known_target > 0 else known_target

    targets: Dict[int, int] = {}
    for cls, count in class_dist.items():
        cls_i = int(cls)
        count_i = int(count)
        if cls_i == unknown_id:
            targets[cls_i] = min(count_i, int(unknown_cap))
        else:
            if known_target <= 0:
                targets[cls_i] = count_i
            else:
                targets[cls_i] = min(count_i, known_target)
            if cls_i in per_class_caps and int(per_class_caps[cls_i]) > 0:
                targets[cls_i] = min(targets[cls_i], int(per_class_caps[cls_i]))
    return targets


def _parse_target_caps(spec: str, class_names: Sequence[str]) -> Dict[int, int]:
    """Parse per-target caps from CSV like 'facebook:20000,meta:8000'."""
    if not spec.strip():
        return {}

    name_to_id = {str(name).strip().lower(): i for i, name in enumerate(class_names)}
    out: Dict[int, int] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid target cap '{part}'. Use target:cap format.")
        name, cap_s = part.split(":", 1)
        name_n = name.strip().lower()
        if name_n not in name_to_id:
            raise ValueError(f"Unknown target in --target-caps: {name}")
        try:
            cap_v = int(cap_s.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid cap value for {name}: {cap_s}") from exc
        if cap_v <= 0:
            continue
        out[int(name_to_id[name_n])] = int(cap_v)
    return out


def _sample_by_class(
    rows_by_class: Dict[int, List[Tuple[Path, int, int, int]]],
    class_targets: Dict[int, int],
    seed: int,
) -> Tuple[List[Tuple[Path, int, int, int]], Counter]:
    rng = np.random.default_rng(seed)
    sampled: List[Tuple[Path, int, int, int]] = []
    actual: Counter = Counter()

    for cls, rows in rows_by_class.items():
        take_n = int(class_targets.get(int(cls), 0))
        arr = list(rows)
        rng.shuffle(arr)
        chosen = arr[:take_n]
        sampled.extend(chosen)
        actual[int(cls)] = len(chosen)

    rng.shuffle(sampled)
    return sampled, actual


def _stratified_split_rows(
    sampled_rows: Sequence[Tuple[Path, int, int, int]],
    train_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int, int, int]], List[Tuple[Path, int, int, int]], Counter, Counter]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[Tuple[Path, int, int, int]]] = defaultdict(list)
    for row in sampled_rows:
        by_class[int(row[2])].append(row)

    train_rows: List[Tuple[Path, int, int, int]] = []
    test_rows: List[Tuple[Path, int, int, int]] = []
    train_dist: Counter = Counter()
    test_dist: Counter = Counter()

    for cls, rows in by_class.items():
        arr = list(rows)
        rng.shuffle(arr)
        n = len(arr)
        if n == 1:
            train_rows.append(arr[0])
            train_dist[int(cls)] += 1
            continue
        n_train = int(round(n * train_fraction))
        n_train = max(1, min(n - 1, n_train))
        left = arr[:n_train]
        right = arr[n_train:]
        train_rows.extend(left)
        test_rows.extend(right)
        train_dist[int(cls)] += len(left)
        test_dist[int(cls)] += len(right)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows, train_dist, test_dist


def _limit_binary_imbalance(
    rows: Sequence[Tuple[Path, int, int, int]],
    unknown_id: int,
    max_binary_imbalance_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int, int, int]], Counter]:
    """Downsample only enough to keep majority/minority <= max_binary_imbalance_ratio.

    Prefers dropping unknown-class rows from the majority side first.
    """
    rng = np.random.default_rng(seed)

    if max_binary_imbalance_ratio <= 0:
        out = list(rows)
        rng.shuffle(out)
        return out, Counter(int(r[3]) for r in out)

    def _pick(rows_in: List[Tuple[Path, int, int, int]], want: int) -> List[Tuple[Path, int, int, int]]:
        if len(rows_in) <= want:
            return list(rows_in)
        unknown_rows = [r for r in rows_in if int(r[2]) == unknown_id]
        known_rows = [r for r in rows_in if int(r[2]) != unknown_id]
        rng.shuffle(unknown_rows)
        rng.shuffle(known_rows)
        out: List[Tuple[Path, int, int, int]] = []
        for part in (known_rows, unknown_rows):
            if len(out) >= want:
                break
            need = want - len(out)
            out.extend(part[:need])
        if len(out) < want:
            extra = rows_in.copy()
            rng.shuffle(extra)
            out.extend(extra[: want - len(out)])
        return out[:want]

    rows_by_class: Dict[int, List[Tuple[Path, int, int, int]]] = defaultdict(list)
    for row in rows:
        rows_by_class[int(row[2])].append(row)

    out: List[Tuple[Path, int, int, int]] = []
    for cls, cls_rows in rows_by_class.items():
        benign = [r for r in cls_rows if int(r[3]) == 0]
        phishing = [r for r in cls_rows if int(r[3]) == 1]

        if not benign or not phishing:
            # If a class only has one binary side, keep it; dropping would erase class signal.
            out.extend(cls_rows)
            continue

        if len(benign) >= len(phishing):
            majority_rows = benign
            minority_rows = phishing
            majority_label = 0
        else:
            majority_rows = phishing
            minority_rows = benign
            majority_label = 1

        minority_n = len(minority_rows)
        max_majority_keep = int(np.floor(float(minority_n) * float(max_binary_imbalance_ratio)))
        keep_majority = min(len(majority_rows), max_majority_keep)

        majority_keep = _pick(majority_rows, keep_majority)
        if majority_label == 0:
            out.extend(majority_keep + minority_rows)
        else:
            out.extend(minority_rows + majority_keep)

    rng.shuffle(out)

    dist = Counter(int(r[3]) for r in out)
    return out, dist


def _write_rows(
    rows: Sequence[Tuple[Path, int, int, int]],
    out_dir: Path,
    split_name: str,
    label_suffix: str,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_chunk: Dict[Path, List[Tuple[int, int, int]]] = defaultdict(list)
    for chunk_path, row_idx, mapped_cls, y_bin in rows:
        by_chunk[chunk_path].append((int(row_idx), int(mapped_cls), int(y_bin)))

    written_chunks = 0
    written_rows = 0
    for idx, (chunk_path, items) in enumerate(tqdm(sorted(by_chunk.items(), key=lambda kv: kv[0].name), desc=f"Writing {split_name}")):
        row_indices = np.asarray([it[0] for it in items], dtype=np.int64)
        mapped_y11 = np.asarray([it[1] for it in items], dtype=np.int16)
        y_bin = np.asarray([it[2] for it in items], dtype=np.int16)

        with np.load(chunk_path, allow_pickle=False) as data:
            X = np.asarray(data["X"], dtype=np.float32)
        X_sub = X[row_indices]

        out_chunk = out_dir / f"{split_name}_domfeat_chunk_{idx:05d}.npz"
        out_sidecar = out_dir / f"{split_name}_domfeat_chunk_{idx:05d}{label_suffix}"
        np.savez(out_chunk, X=X_sub, y=y_bin)
        np.savez(out_sidecar, y11=mapped_y11)

        written_chunks += 1
        written_rows += int(len(mapped_y11))

    return written_chunks, written_rows


def _class_dist(rows: Sequence[Tuple[Path, int, int, int]]) -> Counter:
    return Counter(int(r[2]) for r in rows)


def _binary_dist(rows: Sequence[Tuple[Path, int, int, int]]) -> Counter:
    return Counter(int(r[3]) for r in rows)


def build_dataset(
    source_dir: Path,
    output_dir: Path,
    label_suffix: str,
    mapping_path: Path,
    known_targets: Sequence[str],
    unknown_cap: int,
    train_fraction: float,
    min_per_known: int,
    known_balance_mode: str,
    max_binary_imbalance_ratio: float,
    target_caps: str,
    binary_from_mapping: bool,
    seed: int,
) -> Dict[str, object]:
    train_chunks = _find_chunks(source_dir, "train")
    test_chunks = _find_chunks(source_dir, "test")
    if not train_chunks or not test_chunks:
        raise ValueError(f"Missing train/test chunks under {source_dir}")

    combined_chunks = train_chunks + test_chunks
    logger.info(f"Combined chunks: train={len(train_chunks)} test={len(test_chunks)} total={len(combined_chunks)}")

    mapping = _load_mapping(mapping_path)
    known_id_map, class_names, unknown_id = _build_known_id_map(mapping, known_targets)
    per_class_caps = _parse_target_caps(target_caps, class_names)

    rows_by_class, combined_class_dist, combined_bin_dist = _collect_rows(
        chunk_files=combined_chunks,
        label_suffix=label_suffix,
        known_id_map=known_id_map,
        unknown_new_id=unknown_id,
        binary_from_mapping=binary_from_mapping,
    )

    class_targets = _choose_class_targets(
        class_dist=combined_class_dist,
        unknown_id=unknown_id,
        unknown_cap=unknown_cap,
        min_per_known=min_per_known,
        known_balance_mode=known_balance_mode,
        per_class_caps=per_class_caps,
    )

    sampled_rows, sampled_class_dist = _sample_by_class(rows_by_class, class_targets, seed=seed)
    train_rows, test_rows, train_class_dist_pre, test_class_dist_pre = _stratified_split_rows(
        sampled_rows,
        train_fraction=train_fraction,
        seed=seed + 1,
    )

    train_rows_bal, train_bin_dist = _limit_binary_imbalance(
        train_rows,
        unknown_id=unknown_id,
        max_binary_imbalance_ratio=max_binary_imbalance_ratio,
        seed=seed + 2,
    )
    test_rows_bal, test_bin_dist = _limit_binary_imbalance(
        test_rows,
        unknown_id=unknown_id,
        max_binary_imbalance_ratio=max_binary_imbalance_ratio,
        seed=seed + 3,
    )

    train_chunks_written, train_rows_written = _write_rows(
        rows=train_rows_bal,
        out_dir=output_dir / "train",
        split_name="train",
        label_suffix=label_suffix,
    )
    test_chunks_written, test_rows_written = _write_rows(
        rows=test_rows_bal,
        out_dir=output_dir / "test",
        split_name="test",
        label_suffix=label_suffix,
    )

    out_map = {
        "source_split": "combined_train_test",
        "top_k": int(len(class_names) - 1),
        "top_targets": list(class_names[:-1]),
        "known_targets": list(class_names[:-1]),
        "unknown_class_name": "unknown_agg",
        "unknown_class_id": int(unknown_id),
        "num_classes": int(len(class_names)),
    }
    (output_dir / "top10_target_map.json").write_text(json.dumps(out_map, indent=2), encoding="utf-8")

    meta = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "mapping_path": str(mapping_path),
        "known_targets": list(class_names[:-1]),
        "class_names": class_names,
        "unknown_class_id": int(unknown_id),
        "train_fraction": float(train_fraction),
        "unknown_cap": int(unknown_cap),
        "max_binary_imbalance_ratio": float(max_binary_imbalance_ratio),
        "target_caps": target_caps,
        "binary_from_mapping": bool(binary_from_mapping),
        "per_class_caps": {str(k): int(v) for k, v in sorted(per_class_caps.items())},
        "min_per_known": int(min_per_known),
        "known_balance_mode": known_balance_mode,
        "seed": int(seed),
        "label_suffix": label_suffix,
        "combined_rows_total": int(sum(combined_class_dist.values())),
        "combined_class_dist": {str(k): int(v) for k, v in sorted(combined_class_dist.items())},
        "combined_binary_dist": {str(k): int(v) for k, v in sorted(combined_bin_dist.items())},
        "class_targets": {str(k): int(v) for k, v in sorted(class_targets.items())},
        "sampled_class_dist": {str(k): int(v) for k, v in sorted(sampled_class_dist.items())},
        "train_class_dist_pre_binary_balance": {str(k): int(v) for k, v in sorted(train_class_dist_pre.items())},
        "test_class_dist_pre_binary_balance": {str(k): int(v) for k, v in sorted(test_class_dist_pre.items())},
        "train_class_dist": {str(k): int(v) for k, v in sorted(_class_dist(train_rows_bal).items())},
        "test_class_dist": {str(k): int(v) for k, v in sorted(_class_dist(test_rows_bal).items())},
        "train_binary_dist": {str(k): int(v) for k, v in sorted(train_bin_dist.items())},
        "test_binary_dist": {str(k): int(v) for k, v in sorted(test_bin_dist.items())},
        "train_rows": int(train_rows_written),
        "test_rows": int(test_rows_written),
        "train_chunks_written": int(train_chunks_written),
        "test_chunks_written": int(test_chunks_written),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create combined stratified balanced dataset")
    p.add_argument("--source-dir", default="dom_unsup_features")
    p.add_argument("--output-dir", default="dom_unsup_subsample_combined")
    p.add_argument("--mapping-path", default="", help="Path to top10_target_map.json; defaults to <source-dir>/top10_target_map.json")
    p.add_argument("--known-targets", default=",".join(DEFAULT_KNOWN_TARGETS), help="Comma-separated known targets in desired class order")
    p.add_argument("--unknown-cap", type=int, default=5000)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument(
        "--max-binary-imbalance-ratio",
        type=float,
        default=1.8,
        help="Maximum majority/minority ratio for benign vs phishing in each split; <=0 disables balancing",
    )
    p.add_argument("--min-per-known", type=int, default=1000)
    p.add_argument("--known-balance-mode", choices=["median", "min", "max", "none"], default="none")
    p.add_argument(
        "--target-caps",
        default="",
        help="Optional per-target caps, CSV target:cap (e.g. 'facebook:20000,meta:8000')",
    )
    p.add_argument(
        "--binary-from-mapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, set binary labels as known-brand=1(phishing), unknown=0(benign)",
    )
    p.add_argument("--label-suffix", default=DEFAULT_LABEL_SUFFIX)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping_path) if args.mapping_path else (source_dir / "top10_target_map.json")
    known_targets = [t.strip() for t in str(args.known_targets).split(",") if t.strip()]

    meta = build_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        label_suffix=str(args.label_suffix),
        mapping_path=mapping_path,
        known_targets=known_targets,
        unknown_cap=int(args.unknown_cap),
        train_fraction=float(args.train_fraction),
        min_per_known=int(args.min_per_known),
        known_balance_mode=str(args.known_balance_mode),
        max_binary_imbalance_ratio=float(args.max_binary_imbalance_ratio),
        target_caps=str(args.target_caps),
        binary_from_mapping=bool(args.binary_from_mapping),
        seed=int(args.seed),
    )

    logger.info("Combined balanced dataset created")
    logger.info(f"train rows: {meta['train_rows']:,} | test rows: {meta['test_rows']:,}")
    logger.info(f"train binary: {meta['train_binary_dist']} | test binary: {meta['test_binary_dist']}")
    logger.info(f"metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
