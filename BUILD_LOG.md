# Architectural Build Log — Package Integrity Classification via Sim-to-Real

## The Core Objective (The "Why")

The primary bottleneck in modern industrial robotics and autonomous systems is the **"Sim-to-Real Gap"**. Relying on manually annotated real-world images to train machine learning models is slow, expensive, and fragile to edge cases. 

Our objective was to engineer a zero-shot Cyber-Physical AI pipeline. By procedurally generating physics-accurate damaged packaging assets in a simulation engine and domain-randomizing the renders, we can train a production-ready YOLOv8 object detection model on **100% synthetic data** capable of running inference on an edge device (Jetson Orin Nano).

## Toolchain & Constraints (The "What")

*   **Simulation Engine**: NVIDIA Omniverse / Isaac Sim 4.x
*   **Synthetic Data Generation**: Omniverse Replicator API
*   **ML Framework**: PyTorch / Ultralytics YOLOv8-Nano
*   **Target Hardware**: NVIDIA Jetson Orin Nano (TensorRT)
*   **Version Control / CI**: Git, GitHub Actions
*   **Constraint 1**: No real-world images used during model training.
*   **Constraint 2**: Scripts must be modular, strongly typed, and entirely reproducible via explicit seed management.

## Execution Sequence (The "How")

### 1. Asset Generation via Rigid Body Physics
Rather than sculpting damage states manually or trying to simulate soft-body tearing in real-time, we utilized Isaac Sim's physics engine as a pre-processing tool.
*   Instanced a `DeformableObject` (cardboard box) with calibrated `youngs_modulus` and `poissons_ratio`.
*   Spawned heavy, rigid spheres above the box and let gravity run its course.
*   Serialized the post-impact meshes directly out to `.usd` format (`box-pristine.usd`, `box-dented.usd`, `box-crushed.usd`).

### 2. Domain Randomization (Omniverse Replicator)
To ensure the model generalizes from simulation to reality, we engineered a world-class Replicator pipeline (`generate-synthetic-data.py`):
*   **Lighting:** Randomized sphere-light intensity and color temperature per frame to simulate varying factory floor conditions.
*   **Camera:** Uniformly sampled camera poses from a bounding hemisphere looking at the target.
*   **Pose:** Applied 360° Z-rotation randomization to the target box.
*   **Custom Writer:** Built a custom `YOLOWriterBackend` to dynamically extract `bounding_box_2d_tight` coordinates and serialize them into normalized YOLO `.txt` labels.

### 3. Model Fine-Tuning & Export
We built an enterprise-grade supervised training script (`train.py`) wrapping the Ultralytics library:
*   Added `argparse` subcommands for clear `train`, `validate`, and `export` lifecycles.
*   Enforced explicit random seeding across `torch`, `numpy`, and `random` to guarantee identical weights across repeated runs.
*   Automated FP16 ONNX export targeted at TensorRT operator set 17 for maximized throughput on Jetson hardware.

## Architectural Trade-offs

1. **Pre-computed Physics vs. Real-time Soft Body Simulation:** Real-time soft body simulation is computationally taxing and unstable. Pre-computing the damage utilizing Isaac Sim's physics engine and exporting the static meshes provided a stable, deterministic base for Replicator to run inference across thousands of frames.
2. **YOLOv8-Nano vs. Heavier Architectures:** We selected the "Nano" variant to ensure maximum FPS on the lower-powered Jetson Orin Nano edge device. While larger models boast higher mAP, inference latency is the critical metric in conveyer-belt tracking systems. 

## What's Next (Wavefront Capabilities)

*   **Isaac ROS Integeration**: Integrating the exported `.onnx` model into an active `isaac_ros_dnn_inference` pipeline.
*   **Physical Hardware Validation**: Deploying the pipeline live against a webcam feed utilizing the `ros-packages/` placeholder directory.
