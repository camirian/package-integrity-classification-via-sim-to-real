#!/usr/bin/env python3
"""YOLOv8-Nano training pipeline for package integrity classification.

Architecture
------------
This module provides a complete **train → validate → export** pipeline for
fine-tuning a YOLOv8-Nano detector on the synthetic dataset produced by
Phase 2.  It is structured around three CLI sub-commands::

    python train.py train     # fine-tune on synthetic data
    python train.py validate  # evaluate on the validation split
    python train.py export    # convert best.pt → ONNX for TensorRT

Design goals:

*   **Reproducibility** — deterministic seeding of ``torch``, ``numpy``, and
    ``random``; full config persisted as JSON alongside the model weights.
*   **Observability** — structured ``logging`` with timestamps, per-class
    metric breakdowns, and a final summary table.
*   **Portability** — ONNX export with embedded metadata (class names, input
    shape, creation timestamp) ready for TensorRT on Jetson Orin Nano.

Requirements
------------
::

    pip install ultralytics onnx onnxruntime-gpu

Usage
-----
::

    # Full pipeline (train → validate → export)
    python train.py train --epochs 100 --batch 16

    # Validate an existing checkpoint
    python train.py validate --weights models/package-integrity-yolov8n/weights/best.pt

    # Export-only
    python train.py export --weights models/package-integrity-yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
logging.basicConfig(format=_LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("train")

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    """Centralised, serialisable training hyper-parameters.

    Every field is documented and has a sensible default.  Override via
    CLI flags (see ``_build_parser``).
    """

    # Paths ----------------------------------------------------------------
    data: Path = field(
        default_factory=lambda: PROJECT_ROOT / "phase-3-model-training" / "config" / "dataset.yaml"
    )
    project: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    name: str = "package-integrity-yolov8n"

    # Base model -----------------------------------------------------------
    base_model: str = "yolov8n.pt"

    # Training hyper-parameters --------------------------------------------
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    lrf: float = 0.01
    weight_decay: float = 5e-4
    cos_lr: bool = True
    patience: int = 15

    # Augmentation ---------------------------------------------------------
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.15

    # Export ---------------------------------------------------------------
    onnx_opset: int = 17
    onnx_half: bool = True  # FP16 for Jetson TensorRT
    onnx_simplify: bool = True

    # Reproducibility ------------------------------------------------------
    seed: int = 42

    # Helpers --------------------------------------------------------------

    def serialise(self, dest: Path) -> None:
        """Write config to JSON for experiment tracking."""
        payload = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self).items()
        }
        payload["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        dest.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Config serialised → %s", dest)


# ═══════════════════════════════════════════════════════════════════════
# Seed management
# ═══════════════════════════════════════════════════════════════════════


def _seed_everything(seed: int) -> None:
    """Seed all relevant RNGs for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is imported transitively by ultralytics

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("RNG seed set to %d", seed)


# ═══════════════════════════════════════════════════════════════════════
# Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════


def _preflight(cfg: TrainingConfig, require_weights: Optional[Path] = None) -> None:
    """Validate environment before expensive operations.

    Checks:
    *   ``dataset.yaml`` exists (for train / validate).
    *   GPU is available.
    *   ``ultralytics`` is importable.
    *   Weights file exists (if provided).
    """
    # Dataset
    if not cfg.data.exists():
        logger.error("Dataset config not found: %s", cfg.data)
        logger.error(
            "Run Phase 2 (generate-synthetic-data.py) first, or update the "
            "'--data' flag."
        )
        sys.exit(1)

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", gpu_name)
        else:
            logger.warning("No CUDA GPU detected — training will be slow.")
    except ImportError:
        logger.warning("PyTorch not found — cannot check GPU availability.")

    # Ultralytics
    try:
        import ultralytics
        logger.info("ultralytics %s", ultralytics.__version__)
    except ImportError:
        logger.error("ultralytics is not installed.  pip install ultralytics")
        sys.exit(1)

    # Weights
    if require_weights is not None and not require_weights.exists():
        logger.error("Weights file not found: %s", require_weights)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════


def _train(cfg: TrainingConfig, run_validation: bool = True) -> Path:
    """Fine-tune YOLOv8-Nano on the synthetic dataset.

    Parameters
    ----------
    cfg
        Training configuration.
    run_validation
        If ``True``, run validation and export after training.

    Returns
    -------
    Path
        Absolute path to ``best.pt``.
    """
    from ultralytics import YOLO

    _preflight(cfg)
    _seed_everything(cfg.seed)

    logger.info("=" * 64)
    logger.info("  YOLOv8-Nano — Training")
    logger.info("=" * 64)
    logger.info("  Dataset     : %s", cfg.data)
    logger.info("  Base model  : %s", cfg.base_model)
    logger.info("  Epochs      : %d", cfg.epochs)
    logger.info("  Batch       : %d", cfg.batch)
    logger.info("  Image size  : %d", cfg.imgsz)
    logger.info("  Optimizer   : %s  (lr0=%g, lrf=%g, wd=%g)",
                cfg.optimizer, cfg.lr0, cfg.lrf, cfg.weight_decay)
    logger.info("-" * 64)

    model = YOLO(cfg.base_model)

    model.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=True,
        # Optimiser
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        # Schedule
        cos_lr=cfg.cos_lr,
        patience=cfg.patience,
        # Augmentation
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        # Misc
        verbose=True,
        seed=cfg.seed,
    )

    best_pt = cfg.project / cfg.name / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("best.pt not found — falling back to last.pt")
        best_pt = cfg.project / cfg.name / "weights" / "last.pt"

    # Persist training config alongside weights
    cfg.serialise(best_pt.parent / "training-config.json")

    logger.info("✅  Training complete.  Best weights: %s", best_pt)

    if run_validation:
        _validate(cfg, best_pt)
        _export(cfg, best_pt)

    return best_pt


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════


def _validate(cfg: TrainingConfig, weights: Path) -> None:
    """Run validation and log per-class metrics.

    Parameters
    ----------
    cfg
        Training configuration (for dataset path and image size).
    weights
        Path to the saved checkpoint.
    """
    from ultralytics import YOLO

    _preflight(cfg, require_weights=weights)

    logger.info("=" * 64)
    logger.info("  YOLOv8-Nano — Validation")
    logger.info("=" * 64)
    logger.info("  Weights : %s", weights)
    logger.info("  Dataset : %s", cfg.data)
    logger.info("-" * 64)

    model = YOLO(str(weights))
    metrics = model.val(data=str(cfg.data), imgsz=cfg.imgsz, verbose=True)

    logger.info("-" * 64)
    logger.info("  %-20s %s", "Metric", "Value")
    logger.info("  %-20s %s", "-" * 20, "-" * 8)
    logger.info("  %-20s %.4f", "mAP@0.5", metrics.box.map50)
    logger.info("  %-20s %.4f", "mAP@0.5:0.95", metrics.box.map)
    logger.info("  %-20s %.4f", "Precision", metrics.box.mp)
    logger.info("  %-20s %.4f", "Recall", metrics.box.mr)
    logger.info("-" * 64)


# ═══════════════════════════════════════════════════════════════════════
# ONNX Export
# ═══════════════════════════════════════════════════════════════════════


def _export(cfg: TrainingConfig, weights: Path) -> Path:
    """Export checkpoint to ONNX for TensorRT deployment.

    The ONNX graph is exported with:
    *   FP16 quantisation (``half=True``) for Jetson GPU efficiency.
    *   Operator-set 17 for broad TensorRT compatibility.
    *   Graph simplification via ``onnx-simplifier``.

    Parameters
    ----------
    cfg
        Training configuration.
    weights
        Path to ``best.pt``.

    Returns
    -------
    Path
        Path to the exported ``.onnx`` file.
    """
    from ultralytics import YOLO

    _preflight(cfg, require_weights=weights)

    logger.info("=" * 64)
    logger.info("  YOLOv8-Nano — ONNX Export")
    logger.info("=" * 64)
    logger.info("  Source    : %s", weights)
    logger.info("  Opset     : %d", cfg.onnx_opset)
    logger.info("  FP16      : %s", cfg.onnx_half)
    logger.info("  Simplify  : %s", cfg.onnx_simplify)
    logger.info("-" * 64)

    model = YOLO(str(weights))

    export_path = model.export(
        format="onnx",
        opset=cfg.onnx_opset,
        imgsz=cfg.imgsz,
        half=cfg.onnx_half,
        simplify=cfg.onnx_simplify,
    )

    # Write export metadata
    meta = {
        "source_weights": str(weights),
        "onnx_path": str(export_path),
        "opset": cfg.onnx_opset,
        "half": cfg.onnx_half,
        "imgsz": cfg.imgsz,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "classes": ["Pristine", "Dented", "Crushed"],
    }
    meta_path = Path(export_path).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    logger.info("✅  ONNX export complete: %s", export_path)
    logger.info("    Metadata: %s", meta_path)
    logger.info(
        "🚀  Next step → Phase 4: deploy to Jetson Orin Nano via TensorRT."
    )
    return Path(export_path)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with sub-commands."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="YOLOv8-Nano training pipeline for package integrity classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune YOLOv8-Nano on synthetic data.")
    p_train.add_argument("--data",   type=str, default=None, help="Path to dataset.yaml")
    p_train.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 100)")
    p_train.add_argument("--batch",  type=int, default=None, help="Batch size (default: 16)")
    p_train.add_argument("--imgsz",  type=int, default=None, help="Image size (default: 640)")
    p_train.add_argument("--seed",   type=int, default=None, help="RNG seed (default: 42)")

    # ── validate ─────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Run validation on an existing checkpoint.")
    p_val.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p_val.add_argument("--data",    type=str, default=None,  help="Path to dataset.yaml")
    p_val.add_argument("--imgsz",   type=int, default=None,  help="Image size")

    # ── export ───────────────────────────────────────────────────────
    p_exp = sub.add_parser("export", help="Export checkpoint to ONNX.")
    p_exp.add_argument("--weights", type=str, required=True, help="Path to best.pt")

    return parser


def _apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> None:
    """Merge CLI arguments into the config (only non-None values)."""
    if getattr(args, "data", None):
        cfg.data = Path(args.data)
    if getattr(args, "epochs", None):
        cfg.epochs = args.epochs
    if getattr(args, "batch", None):
        cfg.batch = args.batch
    if getattr(args, "imgsz", None):
        cfg.imgsz = args.imgsz
    if getattr(args, "seed", None):
        cfg.seed = args.seed


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Parse CLI and dispatch to the appropriate sub-command."""
    parser = _build_parser()
    args = parser.parse_args()

    cfg = TrainingConfig()
    _apply_overrides(cfg, args)

    if args.command == "train":
        _train(cfg)

    elif args.command == "validate":
        weights = Path(args.weights)
        _validate(cfg, weights)

    elif args.command == "export":
        weights = Path(args.weights)
        _export(cfg, weights)


if __name__ == "__main__":
    main()
