#!/usr/bin/env python3
"""Omniverse Replicator pipeline for synthetic package-integrity data.

Architecture
------------
This module implements a complete domain-randomised synthetic data generation
pipeline for training object-detection models that classify package damage
states.  It is designed to run inside **NVIDIA Isaac Sim** via the Script
Editor or the ``--exec`` CLI flag.

The pipeline follows three stages per asset class:

    1. **Scene construction** — load USD asset, ground plane, and lighting.
    2. **Domain randomisation** — randomise camera pose, light intensity /
       colour temperature, and box orientation every frame.
    3. **Rendering & serialisation** — capture RGB + 2-D bounding boxes and
       write them in YOLO format via a custom ``WriterBackend`` subclass.

Output structure (YOLO convention)::

    data/synthetic-dataset/
    ├── dataset.yaml          # auto-generated Ultralytics config
    ├── generation-config.json  # frozen run parameters for reproducibility
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

Usage
-----
Inside Isaac Sim Script Editor::

    exec(open("phase-2-synthetic-data-generation/scripts/generate-synthetic-data.py").read())

Via CLI::

    isaac-sim --exec generate-synthetic-data.py

Notes
-----
*   All tunable parameters are collected in the ``PipelineConfig`` dataclass.
*   Random state is explicitly seeded for full reproducibility.
*   The ``YOLOWriterBackend`` is registered as a first-class Replicator writer.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Isaac Sim / Omniverse imports — available only inside the Omniverse runtime.
# ---------------------------------------------------------------------------
import omni.replicator.core as rep
import omni.usd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
logging.basicConfig(format=_LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("synth-datagen")

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AssetSpec:
    """Specification for a single USD asset and its semantic label."""

    label: str
    class_id: int
    usd_path: Path

    def validate(self) -> None:
        """Raise ``FileNotFoundError`` if the USD file is missing."""
        if not self.usd_path.exists():
            raise FileNotFoundError(
                f"USD asset not found for class '{self.label}': {self.usd_path}"
            )


@dataclass
class CameraRandomisation:
    """Bounds for hemisphere camera sampling (metres)."""

    position_min: Tuple[float, float, float] = (-0.6, -0.6, 0.25)
    position_max: Tuple[float, float, float] = ( 0.6,  0.6, 0.80)
    look_at:      Tuple[float, float, float] = ( 0.0,  0.0, 0.15)


@dataclass
class LightRandomisation:
    """Bounds for point-light randomisation."""

    intensity_min: float = 500.0
    intensity_max: float = 3_000.0
    temperature_min: float = 3_500.0
    temperature_max: float = 7_500.0
    position_min: Tuple[float, float, float] = (-1.0, -1.0, 0.8)
    position_max: Tuple[float, float, float] = ( 1.0,  1.0, 1.5)


@dataclass
class PipelineConfig:
    """Top-level configuration for the synthetic data pipeline.

    All paths are resolved relative to the project root.  Modify fields
    here — they propagate automatically into every sub-system.
    """

    # Paths ----------------------------------------------------------------
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # Assets (populated in ``__post_init__``)
    assets: List[AssetSpec] = field(default_factory=list)

    # Output
    output_dir: Path = field(init=False)

    # Generation parameters ------------------------------------------------
    frames_per_class: int = 500
    train_ratio: float = 0.80
    image_width: int = 640
    image_height: int = 640
    seed: int = 42

    # Domain randomisation -------------------------------------------------
    camera: CameraRandomisation = field(default_factory=CameraRandomisation)
    light: LightRandomisation = field(default_factory=LightRandomisation)

    # Dome light
    dome_intensity: float = 800.0

    def __post_init__(self) -> None:
        self.output_dir = self.project_root / "data" / "synthetic-dataset"

        asset_dir = self.project_root / "phase-1-asset-and-scene-creation" / "assets"
        if not self.assets:
            self.assets = [
                AssetSpec("Pristine", 0, asset_dir / "box-pristine.usd"),
                AssetSpec("Dented",   1, asset_dir / "box-dented.usd"),
                AssetSpec("Crushed",  2, asset_dir / "box-crushed.usd"),
            ]

    # Helpers --------------------------------------------------------------

    @property
    def class_map(self) -> Dict[str, int]:
        """Return ``{label: class_id}`` mapping."""
        return {a.label: a.class_id for a in self.assets}

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class names."""
        return [a.label for a in sorted(self.assets, key=lambda a: a.class_id)]

    def validate_assets(self) -> None:
        """Validate that every referenced USD file exists on disk."""
        for asset in self.assets:
            asset.validate()

    def serialise(self, path: Path) -> None:
        """Persist the full config as JSON for reproducibility."""
        payload: Dict[str, Any] = {
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "frames_per_class": self.frames_per_class,
            "train_ratio": self.train_ratio,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "seed": self.seed,
            "assets": [
                {"label": a.label, "class_id": a.class_id, "usd_path": str(a.usd_path)}
                for a in self.assets
            ],
            "camera": asdict(self.camera),
            "light": asdict(self.light),
            "dome_intensity": self.dome_intensity,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Config serialised → %s", path)


# ═══════════════════════════════════════════════════════════════════════
# YOLO Writer Backend
# ═══════════════════════════════════════════════════════════════════════


class YOLOWriterBackend:
    """Replicator-compatible writer that emits YOLO-format labels.

    Each rendered frame produces:
    *   An RGB image (PNG) in ``{split}/images/{frame_id}.png``
    *   A label file in ``{split}/labels/{frame_id}.txt`` containing one
        normalised bounding-box line per object::

            <class_id> <x_centre> <y_centre> <width> <height>

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (provides paths, class map, split ratio).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._class_map = config.class_map
        self._rng = random.Random(config.seed)
        self._frame_counter: int = 0
        self._split_counts: Dict[str, int] = {"train": 0, "val": 0}

        self._dirs: Dict[str, Path] = {}
        for split in ("train", "val"):
            img_dir = config.output_dir / split / "images"
            lbl_dir = config.output_dir / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            self._dirs[f"{split}_images"] = img_dir
            self._dirs[f"{split}_labels"] = lbl_dir

        logger.info(
            "YOLOWriterBackend initialised  (output=%s, classes=%s)",
            config.output_dir,
            list(self._class_map.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, data: Dict[str, Any]) -> None:
        """Persist one rendered frame as image + label.

        Parameters
        ----------
        data
            Must contain ``"rgb"`` (H×W×4 uint8 RGBA array).  May contain
            ``"bounding_box_2d_tight"`` (structured array with fields
            ``x_min``, ``y_min``, ``x_max``, ``y_max``, ``semanticLabel``).
        """
        from PIL import Image  # deferred: only needed at write time

        rgb: Optional[np.ndarray] = data.get("rgb")
        if rgb is None:
            logger.warning("Frame %d: no RGB data — skipped.", self._frame_counter)
            return

        bbox_data = data.get("bounding_box_2d_tight")

        # Determine split
        split = "train" if self._rng.random() < self._cfg.train_ratio else "val"
        img_dir = self._dirs[f"{split}_images"]
        lbl_dir = self._dirs[f"{split}_labels"]

        frame_id = f"{self._frame_counter:06d}"
        self._frame_counter += 1
        self._split_counts[split] += 1

        # ── Save image ──────────────────────────────────────────────
        img = Image.fromarray(rgb[:, :, :3])  # drop alpha
        img.save(str(img_dir / f"{frame_id}.png"))

        # ── Build label lines ───────────────────────────────────────
        h, w = rgb.shape[:2]
        lines: List[str] = []

        if bbox_data is not None and len(bbox_data) > 0:
            for bbox in bbox_data:
                sem_label: str = bbox["semanticLabel"]
                if sem_label not in self._class_map:
                    continue

                cls = self._class_map[sem_label]
                x_min, y_min = float(bbox["x_min"]), float(bbox["y_min"])
                x_max, y_max = float(bbox["x_max"]), float(bbox["y_max"])

                cx = ((x_min + x_max) / 2.0) / w
                cy = ((y_min + y_max) / 2.0) / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h

                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (lbl_dir / f"{frame_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return generation statistics."""
        return {
            "total_frames": self._frame_counter,
            **self._split_counts,
        }


# ═══════════════════════════════════════════════════════════════════════
# Scene helpers
# ═══════════════════════════════════════════════════════════════════════


def _build_scene(
    asset: AssetSpec,
    cfg: PipelineConfig,
) -> Tuple[rep.NodeType, rep.NodeType, rep.NodeType]:
    """Construct a Replicator scene for a single asset class.

    Returns
    -------
    camera, sphere_light, box
        Handles used by the randomiser graph.
    """
    rep.utils.stage_clear()
    logger.info("Stage cleared.")

    # Ground plane
    rep.create.plane(
        semantics=[("class", "background")],
        position=(0, 0, 0),
        scale=5.0,
    )

    # Dome (ambient) light
    rep.create.light(
        light_type="Dome",
        intensity=cfg.dome_intensity,
        rotation=(270, 0, 0),
    )

    # Sphere (point) light — will be randomised
    sphere_light = rep.create.light(
        light_type="Sphere",
        intensity=1_500,
        position=(0, 0, 1),
    )

    # Camera
    camera = rep.create.camera(
        position=(0.4, 0.4, 0.5),
        look_at=cfg.camera.look_at,
    )

    # USD asset
    box = rep.create.from_usd(
        str(asset.usd_path),
        semantics=[("class", asset.label)],
    )

    logger.info("Scene built for class '%s'  (asset=%s)", asset.label, asset.usd_path)
    return camera, sphere_light, box


def _register_randomisers(
    camera: rep.NodeType,
    light: rep.NodeType,
    box: rep.NodeType,
    cfg: PipelineConfig,
) -> None:
    """Wire up per-frame domain randomisation.

    Randomised quantities:
    *   Camera position (uniform over a rectangular volume).
    *   Light intensity, colour temperature, and position.
    *   Box yaw rotation (full 360°).
    """
    cam = cfg.camera
    lgt = cfg.light

    with rep.trigger.on_frame():
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(cam.position_min, cam.position_max),
                look_at=cam.look_at,
            )

        with light:
            rep.modify.attribute(
                "intensity",
                rep.distribution.uniform(lgt.intensity_min, lgt.intensity_max),
            )
            rep.modify.attribute(
                "colorTemperature",
                rep.distribution.uniform(lgt.temperature_min, lgt.temperature_max),
            )
            rep.modify.pose(
                position=rep.distribution.uniform(lgt.position_min, lgt.position_max),
            )

        with box:
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
            )

    logger.info("Randomisers registered.")


# ═══════════════════════════════════════════════════════════════════════
# Dataset YAML emitter
# ═══════════════════════════════════════════════════════════════════════


def _write_dataset_yaml(cfg: PipelineConfig) -> Path:
    """Write an Ultralytics-compatible ``dataset.yaml``."""
    yaml_path = cfg.output_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Auto-generated by generate-synthetic-data.py\n"
        "#\n"
        "# For Ultralytics YOLOv8 — see:\n"
        "#   https://docs.ultralytics.com/datasets/detect/\n"
        "\n"
        f"path: {cfg.output_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        "\n"
        f"nc: {len(cfg.class_map)}\n"
        f"names: {cfg.class_names}\n"
    )
    logger.info("dataset.yaml → %s", yaml_path)
    return yaml_path


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════


def run(cfg: Optional[PipelineConfig] = None) -> None:
    """Execute the full synthetic data generation pipeline.

    Parameters
    ----------
    cfg
        Pipeline configuration.  If ``None``, the default config is used.
    """
    if cfg is None:
        cfg = PipelineConfig()

    # ── Seed all RNGs ────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Pre-flight checks ────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("  Package Integrity — Synthetic Data Generation")
    logger.info("=" * 64)
    logger.info("  Output dir   : %s", cfg.output_dir)
    logger.info("  Resolution   : %d × %d", cfg.image_width, cfg.image_height)
    logger.info("  Frames/class : %d", cfg.frames_per_class)
    logger.info("  Train ratio  : %.0f%%", cfg.train_ratio * 100)
    logger.info("  Seed         : %d", cfg.seed)
    logger.info("-" * 64)

    try:
        cfg.validate_assets()
        logger.info("All %d asset paths validated.", len(cfg.assets))
    except FileNotFoundError as exc:
        logger.error("Asset validation failed: %s", exc)
        logger.error(
            "Ensure Phase 1 assets exist in %s",
            cfg.project_root / "phase-1-asset-and-scene-creation" / "assets",
        )
        raise

    # ── Prepare output directory ─────────────────────────────────────
    if cfg.output_dir.exists():
        logger.warning("Output directory exists — removing: %s", cfg.output_dir)
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True)

    # ── Persist config for reproducibility ───────────────────────────
    cfg.serialise(cfg.output_dir / "generation-config.json")

    # ── Initialise writer ────────────────────────────────────────────
    writer = YOLOWriterBackend(cfg)

    # ── Generate per-class ───────────────────────────────────────────
    for asset in cfg.assets:
        logger.info("=" * 64)
        logger.info("  Generating: class '%s'", asset.label)
        logger.info("  Asset     : %s", asset.usd_path)
        logger.info("=" * 64)

        camera, light, box = _build_scene(asset, cfg)
        _register_randomisers(camera, light, box, cfg)

        render_product = rep.create.render_product(
            camera, (cfg.image_width, cfg.image_height)
        )

        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_ann.attach([render_product])

        bbox_ann = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        bbox_ann.attach([render_product])

        for i in range(cfg.frames_per_class):
            rep.orchestrator.step()
            writer.write({
                "rgb": rgb_ann.get_data(),
                "bounding_box_2d_tight": bbox_ann.get_data(),
            })
            if (i + 1) % 50 == 0:
                logger.info(
                    "  [%s] %d / %d frames", asset.label, i + 1, cfg.frames_per_class
                )

        rgb_ann.detach([render_product])
        bbox_ann.detach([render_product])
        logger.info("  Class '%s' complete.", asset.label)

    # ── Write dataset.yaml ───────────────────────────────────────────
    _write_dataset_yaml(cfg)

    # ── Summary ──────────────────────────────────────────────────────
    stats = writer.stats
    logger.info("=" * 64)
    logger.info("  ✅  Synthetic data generation complete")
    logger.info("  Total frames : %d", stats["total_frames"])
    logger.info("  Train        : %d", stats["train"])
    logger.info("  Val          : %d", stats["val"])
    logger.info("  Output       : %s", cfg.output_dir)
    logger.info("=" * 64)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
