#!/usr/bin/env python
"""Evaluate EMBER raw-feature JSONL with a LightGBM model.

Examples:
    # Score with the pretrained EMBER2018 model
    python scripts/evaluate_features.py features.jsonl \
        --model /data/ember2018/ember_model_2018.txt \
        --target-fpr 0.01 0.001

    # Train on a separate feature file and evaluate on features.jsonl
    python scripts/evaluate_features.py eval.jsonl \
        --train-features train.jsonl --save-model model.txt

    # Train/evaluate with a stratified split of one file
    python scripts/evaluate_features.py features.jsonl \
        --train --test-size 0.3 --seed 42

The input must contain both label=0 and label=1 records. Unlabeled records
(label=-1) are ignored. Metrics are printed to stdout; LightGBM logs go to
stderr.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import ember
from ember.features import PEFeatureExtractor

EMBER_TRAIN_PARAMS = {
    "boosting": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 2048,
    "max_depth": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.5,
    "verbosity": -1,
}
EMBER_NUM_BOOST_ROUND = 1000


def configure_lightgbm_logging(verbose: bool) -> None:
    logger = logging.getLogger("ember_eval_lightgbm")
    logger.handlers = [logging.StreamHandler(sys.stderr)]
    logger.propagate = False
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
    lgb.register_logger(logger)


def count_lines(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        try:
            with open(path, "rb") as handle:
                for _ in handle:
                    total += 1
        except OSError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
    return total


def vectorize(
    paths: Sequence[str],
    workdir: str,
    prefix: str,
    feature_version: int,
) -> Tuple[np.memmap, np.memmap, PEFeatureExtractor]:
    nrows = count_lines(paths)
    if nrows == 0:
        raise SystemExit(f"no records in {', '.join(paths)}")
    extractor = PEFeatureExtractor(feature_version, print_feature_warning=False)
    X_path = os.path.join(workdir, f"{prefix}_X.dat")
    y_path = os.path.join(workdir, f"{prefix}_y.dat")
    try:
        ember.vectorize_subset(X_path, y_path, list(paths), extractor, nrows)
    except Exception as exc:  # noqa: BLE001 - make the error actionable
        raise SystemExit(
            "vectorization failed: "
            f"{type(exc).__name__}: {exc}\n"
            "Check that --feature-version matches the JSONL and that every "
            "record contains all feature keys."
        ) from exc
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="r")
    return X, y, extractor


def predict_scores(model: lgb.Booster, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int(np.count_nonzero(mask))
    scores = np.empty(n, dtype=np.float64)
    out = 0
    chunk_size = 100_000
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        chunk_mask = mask[start:end]
        if not chunk_mask.any():
            continue
        chunk = np.asarray(X[start:end])
        predictions = model.predict(chunk[chunk_mask])
        scores[out : out + len(predictions)] = predictions
        out += len(predictions)
    if out != n:
        raise RuntimeError(f"predicted {out} scores, expected {n}")
    return scores


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_pred & (y == 1)))
    fp = int(np.sum(y_pred & (y == 0)))
    tn = int(np.sum(~y_pred & (y == 0)))
    fn = int(np.sum(~y_pred & (y == 1)))
    tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * tpr, precision + tpr)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "accuracy": accuracy,
        "threshold_used": None,
    }


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    result = classification_metrics(y, scores >= threshold)
    result["threshold_used"] = float(threshold)
    return result


def operating_point(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> Dict[str, Any]:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    target = min(max(float(target_fpr), 0.0), 1.0)
    idx = int(np.searchsorted(fpr, target, side="right") - 1)
    idx = min(max(idx, 0), len(fpr) - 1)

    achieved_fpr = float(fpr[idx])
    achieved_tpr = float(tpr[idx])
    tp = int(round(achieved_tpr * n_pos))
    fp = int(round(achieved_fpr * n_neg))
    tn = n_neg - fp
    fn = n_pos - tp
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * achieved_tpr, precision + achieved_tpr)
    threshold = float(thresholds[idx])
    if not math.isfinite(threshold):
        threshold_out: Optional[float] = None
    else:
        threshold_out = threshold
    return {
        "target_fpr": target,
        "achieved_fpr": achieved_fpr,
        "tpr": achieved_tpr,
        "fnr": 1.0 - achieved_tpr,
        "threshold": threshold_out,
        "precision": precision,
        "recall": achieved_tpr,
        "f1": f1,
        "accuracy": safe_div(tp + tn, n_pos + n_neg),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "min_fpr_one_fp": safe_div(1, n_neg),
    }


def best_f1_point(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.zeros_like(precision)
    denom = precision + recall
    nonzero = denom > 0
    f1[nonzero] = 2 * precision[nonzero] * recall[nonzero] / denom[nonzero]
    idx = int(np.argmax(f1))
    if idx < len(thresholds):
        threshold = float(thresholds[idx])
        result = threshold_metrics(y, scores, threshold)
        result["f1"] = float(f1[idx])
        result["precision"] = float(precision[idx])
        result["recall"] = float(recall[idx])
        return result
    # Last PR point has no corresponding threshold; fall back to 0.0.
    result = threshold_metrics(y, scores, 0.0)
    result["f1"] = float(f1[idx])
    result["precision"] = float(precision[idx])
    result["recall"] = float(recall[idx])
    return result


def fpr_at_tpr(y: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    idx = int(np.searchsorted(tpr, target_tpr, side="left"))
    idx = min(max(idx, 0), len(tpr) - 1)
    threshold = float(thresholds[idx])
    return {
        "target_tpr": float(target_tpr),
        "achieved_tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "threshold": threshold if math.isfinite(threshold) else None,
    }


def score_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(percentiles[2]),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    target_fprs: Sequence[float],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    aucs: List[float] = []
    aps: List[float] = []
    tprs: Dict[str, List[float]] = {f"{t:g}": [] for t in target_fprs}
    for _ in range(n_boot):
        take = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        yb = y[take]
        sb = scores[take]
        aucs.append(float(roc_auc_score(yb, sb)))
        aps.append(float(average_precision_score(yb, sb)))
        fpr, tpr, _ = roc_curve(yb, sb, drop_intermediate=False)
        for target in target_fprs:
            idx = int(np.searchsorted(fpr, target, side="right") - 1)
            idx = min(max(idx, 0), len(tpr) - 1)
            tprs[f"{target:g}"].append(float(tpr[idx]))

    def interval(values: Sequence[float]) -> Optional[List[float]]:
        if not values:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "n_boot": int(n_boot),
        "roc_auc": interval(aucs),
        "average_precision": interval(aps),
        "tpr_at_fpr": {key: interval(values) for key, values in tprs.items()},
    }


def iter_records(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    raise SystemExit(f"blank line in {path}:{line_no}")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                source = obj.get("source") or {}
                yield {
                    "sha256": obj.get("sha256", ""),
                    "label": obj.get("label"),
                    "archive": source.get("archive", ""),
                    "member": source.get("member", ""),
                }


def write_scores_csv(
    path: str,
    records: Iterable[Dict[str, Any]],
    scores: np.ndarray,
    indices: np.ndarray,
    split: str,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    sorted_scores = scores[order]
    ptr = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "label", "score", "split", "archive", "member"])
        for record_index, record in enumerate(records):
            if ptr >= len(sorted_indices):
                break
            if record_index == int(sorted_indices[ptr]):
                writer.writerow(
                    [
                        record.get("sha256", ""),
                        record.get("label", ""),
                        f"{float(sorted_scores[ptr]):.10g}",
                        split,
                        record.get("archive", ""),
                        record.get("member", ""),
                    ]
                )
                ptr += 1


def load_train_params(path: Optional[str]) -> Tuple[Dict[str, Any], int]:
    params = dict(EMBER_TRAIN_PARAMS)
    num_boost_round = EMBER_NUM_BOOST_ROUND
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                user_params = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        if not isinstance(user_params, dict):
            raise SystemExit("--train-params-json must contain a JSON object")
        if "num_iterations" in user_params:
            num_boost_round = int(user_params.pop("num_iterations"))
        params.update(user_params)
    params.pop("num_iterations", None)
    return params, num_boost_round


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "inf"
    return f"{value:.{digits}f}"


def print_operating_points(points: List[Tuple[str, Dict[str, Any]]]) -> None:
    header = (
        f"{'point':<18}{'target':>8}{'FPR':>9}{'threshold':>11}"
        f"{'TPR':>9}{'FNR':>9}{'precision':>11}{'F1':>9}"
        f"{'TP':>7}{'FP':>7}{'TN':>7}{'FN':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, point in points:
        if name in ("threshold 0.5", "best F1"):
            target_text = "-"
        else:
            target = point.get("target_fpr", point.get("target_tpr"))
            if isinstance(target, float):
                target_text = f"{target:.4f}"
            elif target is None:
                target_text = "-"
            else:
                target_text = str(target)
        threshold = point.get("threshold", point.get("threshold_used"))
        if threshold is None and "threshold" in point:
            threshold_text = "inf"
        else:
            threshold_text = format_float(threshold, 4)
        print(
            f"{name:<18}{target_text:>8}"
            f"{point.get('fpr', point.get('achieved_fpr', 0.0)):>9.4f}"
            f"{threshold_text:>11}"
            f"{point['tpr']:>9.4f}"
            f"{point['fnr']:>9.4f}"
            f"{point.get('precision', 0.0):>11.4f}"
            f"{point.get('f1', 0.0):>9.4f}"
            f"{point.get('tp', 0):>7d}"
            f"{point.get('fp', 0):>7d}"
            f"{point.get('tn', 0):>7d}"
            f"{point.get('fn', 0):>7d}"
        )


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    target_fprs: Sequence[float],
    partial_auc_max_fpr: float,
) -> Dict[str, Any]:
    if len(y) != len(scores):
        raise SystemExit(f"label/score length mismatch: {len(y)} vs {len(scores)}")
    if not np.isfinite(scores).all():
        raise SystemExit("model produced non-finite scores")
    if len(np.unique(y)) < 2:
        raise SystemExit("need both label=0 and label=1 records to evaluate")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    eps = 1e-15
    clipped = np.clip(scores.astype(np.float64), eps, 1 - eps)
    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "prevalence": safe_div(n_pos, n_pos + n_neg),
    }
    if partial_auc_max_fpr is not None and 0 < partial_auc_max_fpr < 1:
        try:
            partial_auc = float(roc_auc_score(y, scores, max_fpr=partial_auc_max_fpr))
            metrics["partial_auc"] = partial_auc if math.isfinite(partial_auc) else None
            metrics["partial_auc_max_fpr"] = float(partial_auc_max_fpr)
        except ValueError:
            metrics["partial_auc"] = None
            metrics["partial_auc_max_fpr"] = float(partial_auc_max_fpr)
    metrics["fpr_at_tpr_0.95"] = fpr_at_tpr(y, scores, 0.95)

    points: List[Tuple[str, Dict[str, Any]]] = []
    for target in target_fprs:
        point = operating_point(y, scores, target)
        point["name"] = f"target_fpr_{target:g}"
        points.append((f"target FPR {target:g}", point))
    point = threshold_metrics(y, scores, 0.5)
    point["name"] = "threshold_0.5"
    points.append(("threshold 0.5", point))
    point = best_f1_point(y, scores)
    point["name"] = "best_f1"
    points.append(("best F1", point))
    metrics["operating_points"] = [point for _, point in points]
    return {
        "metrics": metrics,
        "operating_points_named": points,
        "counts": {"positive": n_pos, "negative": n_neg},
        "score_stats": {
            "positive": score_stats(scores[y == 1]),
            "negative": score_stats(scores[y == 0]),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EMBER feature JSONL with a LightGBM model."
    )
    parser.add_argument("features", nargs="+", help="feature JSONL file(s) to evaluate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="pretrained LightGBM model (.txt)")
    source.add_argument(
        "--train",
        action="store_true",
        help="train on a stratified split of the input and evaluate on the rest",
    )
    source.add_argument(
        "--train-features",
        action="append",
        help="train on this JSONL file (repeatable), evaluate on the positional features",
    )
    parser.add_argument(
        "--target-fpr",
        nargs="+",
        type=float,
        default=[0.01, 0.001],
        help="target false-positive rates (default: 0.01 0.001)",
    )
    parser.add_argument(
        "--partial-auc-max-fpr",
        type=float,
        default=0.005,
        help="partial ROC AUC restricted to FPR <= this value (default: 0.005; 0 disables)",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="test fraction for --train")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--train-params-json",
        help="JSON object overriding LightGBM training parameters",
    )
    parser.add_argument("--save-model", help="save a model trained by --train/--train-features")
    parser.add_argument("--json-out", help="write all metrics as JSON")
    parser.add_argument("--scores-out", help="write per-sample scores as CSV")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for 95%% CI")
    parser.add_argument("--workdir", help="directory for vectorized memmaps (default: temporary)")
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the vectorized memmaps after evaluation",
    )
    parser.add_argument("--verbose", action="store_true", help="show LightGBM logs")
    return parser


def run(args: argparse.Namespace) -> int:
    if any(target < 0 or target > 1 for target in args.target_fpr):
        raise SystemExit("--target-fpr values must be in [0, 1]")
    if not (0 < args.test_size < 1):
        raise SystemExit("--test-size must be between 0 and 1")
    if args.save_model and not (args.train or args.train_features):
        raise SystemExit("--save-model requires --train or --train-features")

    configure_lightgbm_logging(args.verbose)

    if args.workdir:
        os.makedirs(args.workdir, mode=0o700, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="ember_eval_", dir=args.workdir)
    else:
        workdir = tempfile.mkdtemp(prefix="ember_eval_")
    remove_workdir = not args.keep_workdir
    try:
        if args.model:
            if not os.path.exists(args.model):
                raise SystemExit(f"model does not exist: {args.model}")
            X, y, extractor = vectorize(
                args.features, workdir, "eval", args.feature_version
            )
            mask = np.asarray(y) != -1
            eval_indices = np.flatnonzero(mask)
            labels = np.asarray(y[eval_indices], dtype=np.int64)
            if labels.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the input")
            model = lgb.Booster(model_file=args.model)
            if model.num_feature() != X.shape[1]:
                raise SystemExit(
                    f"model expects {model.num_feature()} features but the JSONL "
                    f"produced {X.shape[1]}; check --feature-version"
                )
            scores = predict_scores(model, X, mask)
            train_info = None
            eval_files = args.features
            eval_split = "eval"
        elif args.train:
            X_all, y_all, extractor = vectorize(
                args.features, workdir, "all", args.feature_version
            )
            label_mask = np.asarray(y_all) != -1
            labeled_positions = np.flatnonzero(label_mask)
            labels_all = np.asarray(y_all[labeled_positions], dtype=np.int64)
            if labels_all.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the input")
            try:
                train_pos, test_pos = train_test_split(
                    labeled_positions,
                    test_size=args.test_size,
                    random_state=args.seed,
                    stratify=labels_all,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"cannot create a stratified train/test split: {exc}; "
                    "use --train-features or a larger dataset"
                ) from exc
            X_train = np.asarray(X_all[train_pos])
            y_train = np.asarray(y_all[train_pos], dtype=np.int64)
            X_test = np.asarray(X_all[test_pos])
            y_test = np.asarray(y_all[test_pos], dtype=np.int64)
            params, num_boost_round = load_train_params(args.train_params_json)
            model = lgb.train(
                params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=num_boost_round,
            )
            if args.save_model:
                model.save_model(args.save_model)
            eval_indices = np.asarray(test_pos, dtype=np.int64)
            labels = y_test
            scores = model.predict(X_test)
            train_info = {
                "train_count": int(len(train_pos)),
                "test_count": int(len(test_pos)),
                "train_params": params,
                "num_boost_round": int(num_boost_round),
            }
            eval_files = args.features
            eval_split = "test"
        else:
            X_train_mm, y_train_mm, extractor = vectorize(
                args.train_features, workdir, "train", args.feature_version
            )
            train_mask = np.asarray(y_train_mm) != -1
            train_pos = np.flatnonzero(train_mask)
            X_train = np.asarray(X_train_mm[train_pos])
            y_train = np.asarray(y_train_mm[train_pos], dtype=np.int64)
            if len(np.unique(y_train)) < 2:
                raise SystemExit("training features need both label=0 and label=1")
            X_eval_mm, y_eval_mm, _ = vectorize(
                args.features, workdir, "eval", args.feature_version
            )
            eval_mask = np.asarray(y_eval_mm) != -1
            eval_indices = np.flatnonzero(eval_mask)
            labels = np.asarray(y_eval_mm[eval_indices], dtype=np.int64)
            X_eval = np.asarray(X_eval_mm[eval_indices])
            if labels.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the evaluation input")
            params, num_boost_round = load_train_params(args.train_params_json)
            model = lgb.train(
                params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=num_boost_round,
            )
            if args.save_model:
                model.save_model(args.save_model)
            scores = model.predict(X_eval)
            train_info = {
                "train_count": int(len(y_train)),
                "test_count": int(len(labels)),
                "train_params": params,
                "num_boost_round": int(num_boost_round),
            }
            eval_files = args.features
            eval_split = "eval"

        if labels.size == 0:
            raise SystemExit("no labeled records to evaluate")
        results = evaluate(labels, scores, args.target_fpr, args.partial_auc_max_fpr)
        n_pos = results["counts"]["positive"]
        n_neg = results["counts"]["negative"]
        warnings: List[str] = []
        if n_pos < 30 or n_neg < 30:
            warnings.append(
                f"small evaluation set (positive={n_pos}, negative={n_neg}); "
                "metrics are unstable"
            )
        for target in args.target_fpr:
            if n_neg > 0 and 0 < target < 1.0 / n_neg:
                warnings.append(
                    f"target FPR {target:g} is below the resolution of "
                    f"1/negative={1.0 / n_neg:.6g}; achieved FPR will be 0"
                )
        if args.bootstrap > 0:
            results["bootstrap"] = bootstrap_ci(
                labels, scores, args.target_fpr, args.bootstrap, args.seed
            )

        summary = {
            "input": list(args.features),
            "feature_version": args.feature_version,
            "model": {
                "source": "pretrained" if args.model else "trained",
                "path": args.model if args.model else (args.save_model or None),
            },
            "training": train_info,
            "counts": {
                "evaluated": int(len(labels)),
                **results["counts"],
            },
            "metrics": results["metrics"],
            "score_stats": results["score_stats"],
            "warnings": warnings,
        }
        if args.bootstrap > 0:
            summary["bootstrap"] = results.get("bootstrap", {})

        if args.scores_out:
            write_scores_csv(
                args.scores_out,
                iter_records(eval_files),
                np.asarray(scores),
                eval_indices,
                eval_split,
            )

        if args.json_out:
            parent = os.path.dirname(os.path.abspath(args.json_out))
            os.makedirs(parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

        print(f"Input: {', '.join(args.features)}")
        print(
            f"Model: {summary['model']['source']}"
            + (f" ({args.model})" if args.model else "")
        )
        if train_info:
            print(
                f"Training: {train_info['train_count']} samples; "
                f"evaluating on {train_info['test_count']} samples"
            )
        print(
            f"Samples: {len(labels)} labeled "
            f"(positive={n_pos}, negative={n_neg}, "
            f"prevalence={n_pos / len(labels):.4f})"
        )
        metrics = results["metrics"]
        print(f"ROC AUC: {metrics['roc_auc']:.6f}")
        if metrics.get("partial_auc") is not None:
            print(
                f"Partial AUC (FPR<={metrics['partial_auc_max_fpr']:g}): "
                f"{metrics['partial_auc']:.6f}"
            )
        print(f"PR AUC / average precision: {metrics['average_precision']:.6f}")
        print(f"Brier score: {metrics['brier_score']:.6f}")
        print(f"Log loss: {metrics['log_loss']:.6f}")
        fpr95 = metrics["fpr_at_tpr_0.95"]
        print(
            f"FPR at TPR>=0.95: {fpr95['fpr']:.6f} "
            f"(achieved TPR={fpr95['achieved_tpr']:.6f}, "
            f"threshold={format_float(fpr95['threshold'])})"
        )
        if args.bootstrap > 0:
            boot = summary["bootstrap"]
            print(
                f"Bootstrap 95% CI ({boot['n_boot']} resamples): "
                f"ROC AUC {boot['roc_auc']}, AP {boot['average_precision']}"
            )
            for target, interval in boot.get("tpr_at_fpr", {}).items():
                print(f"  TPR at FPR {target}: {interval}")
        print()
        print("Operating points:")
        print_operating_points(results["operating_points_named"])
        print()
        print("Score distributions:")
        for label_name in ("positive", "negative"):
            stats = results["score_stats"][label_name]
            print(
                f"  {label_name:<8} n={stats['count']} mean={stats['mean']:.6f} "
                f"median={stats['median']:.6f} p1={stats['p1']:.6f} "
                f"p5={stats['p5']:.6f} p95={stats['p95']:.6f} "
                f"p99={stats['p99']:.6f}"
            )
        if warnings:
            print()
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        if args.json_out:
            print()
            print(f"Metrics JSON: {args.json_out}")
        if args.scores_out:
            print(f"Per-sample scores: {args.scores_out}")
    finally:
        if remove_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep_workdir:
            print(f"Workdir kept: {workdir}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
