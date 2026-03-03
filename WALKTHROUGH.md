# Walkthrough Usage Guide — Package Integrity Classification

This guide details exactly how to reproduce the 100% synthetic pipeline utilized in this project, assuming access to an NVIDIA Isaac Sim workstation.

---

## 1. Environment Setup

*Ensure your `.venv` or local environment has the required pip packages.*

```bash
# From the repository root
pip install -r requirements.txt
```

---

## 2. Generate Synthetic Data (Phase 2)

We will use the custom Omniverse Replicator script to generate 1,500 domain-randomized images (500 frames per class) with perfectly matched bounding box annotations.

> **Execution Method:** This strictly needs to be run _inside_ the Omniverse / Isaac Sim context to access the `omni.replicator.core` modules.

```bash
# Launch Isaac Sim in headless execution mode
isaac-sim --exec phase-2-synthetic-data-generation/scripts/generate-synthetic-data.py
```

### Expected Output:
The script will clear any existing data and produce:
```
data/synthetic-dataset/
├── dataset.yaml
├── generation-config.json
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

---

## 3. Train the Model (Phase 3)

With the synthetic data generated, we fine-tune a pre-trained YOLOv8-Nano model. The CLI provided by `train.py` automates the entire process.

```bash
# Fine-tune the model over 100 epochs
python phase-3-model-training/scripts/train.py train --epochs 100 --batch 16
```

### Checkpoint Validation
Once trained, the `best.pt` file can be evaluated against the hidden validation split:

```bash
# Validate against the synthetic validation subset
python phase-3-model-training/scripts/train.py validate --weights models/package-integrity-yolov8n/weights/best.pt
```

---

## 4. Export for Edge Inference

The Jetson Orin Nano requires optimized deployment packages. We export the trainedPyTorch graph out to ONNX using FP16 quantization for direct ingestion by TensorRT.

```bash
# Export optimized ONNX graph for Jetson deployment
python phase-3-model-training/scripts/train.py export --weights models/package-integrity-yolov8n/weights/best.pt
```

### Deployment Handoff
The resulting `best.onnx` file and its metadata JSON will be located right beside your weights directory, ready to be transferred to your edge device!
