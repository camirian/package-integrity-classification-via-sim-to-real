# Package Integrity Classification via Sim-to-Real

## Table of Contents

- [Project Structure](#project-structure)
- [Implementation Guide](#implementation-guide)
  - [Phase 1: Environment Setup & Hardware Configuration](#phase-1-environment-setup--hardware-configuration)
    - [Jetson Orin Nano Setup](#jetson-orin-nano-setup)
    - [Workstation Setup](#workstation-setup)
  - [Phase 2: Creating Deformed Assets in Isaac Sim](#phase-2-creating-deformed-assets-in-isaac-sim)
    - [Step 1: Creating the Pristine Deformable Box](#step-1-creating-the-pristine-deformable-box)
    - [Step 2: Scripting the Deformation](#step-2-scripting-the-deformation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

A revised, high-impact project is proposed: to train an object detection model that can classify the physical state of a package (e.g., "Pristine," "Dented," "Crushed") in real-time on a Jetson Orin Nano. The critical constraint and the primary skill this project demonstrates is that the model will be trained 100% on synthetic data generated in Isaac Sim.

This revised mission is significantly more valuable for a professional portfolio for several key reasons:

*   **Industry Relevance:** It directly addresses a common and costly problem in logistics, manufacturing, and e-commerce: automated quality control and damage detection.
*   **Workflow Mastery:** It requires and demonstrates proficiency across the entire NVIDIA Isaac workflow: 3D simulation (Isaac Sim), synthetic data generation (Omniverse Replicator), and hardware-accelerated deployment (Isaac ROS on Jetson).
*   **Solves a Core Challenge:** Success is objectively measurable and provides a concrete solution to the "sim-to-real gap"—the difficulty of making models trained in simulation work in the real world. This is a primary focus of robotics research and development at NVIDIA and across the industry.

## Project Structure

```
.
├── README.md
├── data/
├── models/
├── notebooks/
├── src/
├── tests/
├── phase_1_asset_and_scene_creation/
│   ├── assets/
│   ├── scenes/
│   └── scripts/
│       ├── create_pristine_box.py
│       └── deform_box.py
├── phase_2_synthetic_data_generation/
│   └── scripts/
├── phase_3_model_training/
│   └── notebooks/
└── phase_4_deployment_and_inference/
    └── ros_packages/
```

## Implementation Guide

This section provides a detailed, step-by-step walkthrough to build the Package Integrity Classification system from the ground up. Each phase is broken down into actionable steps with code examples and configuration details.

### Phase 1: Environment Setup & Hardware Configuration

Proper setup is the foundation of a successful project. This involves preparing both the edge device (Jetson) and the development workstation.

#### Jetson Orin Nano Setup

The most straightforward method to prepare the Jetson Orin Nano Developer Kit is to use the official SD card image, which ensures all necessary components and power profiles are available.

1.  **Flash JetPack 6.2.1:** Download the "JetPack 6.2.1 SD Card Image for Jetson Orin Nano Developer Kit" from the NVIDIA developer website. Use a tool like Balena Etcher to flash this image onto a high-quality microSD card (at least 64 GB, U3/A2 rated).
2.  **Initial Boot:** Insert the microSD card into the Jetson Orin Nano and power it on. Complete the initial Ubuntu setup wizard.
3.  **Verify Version:** Open a terminal on the Jetson and verify the installation:

    ```bash
    cat /etc/nv_tegra_release
    ```

    The output should confirm the R36.4.4 release for JetPack 6.2.1.
4.  **Enable Super Mode:** The new power modes can be controlled via the `nvpmodel` command-line tool or the power icon in the desktop menu bar. To unlock maximum performance for the inference task, set the mode to MAXN SUPER.

    ```bash
    # Set power mode to MAXN SUPER for Orin Nano
    sudo nvpmodel -m 2
    ```

#### Workstation Setup

The workstation is used for simulation and model training. A powerful NVIDIA RTX GPU is required.

1.  **Install Omniverse Kit SDK:** The NVIDIA Omniverse Launcher is deprecated. The new method involves downloading the Omniverse Kit SDK directly.
    *   Download the `.zip` file for the Omniverse Kit SDK for Windows from the [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/kit-sdk-windows/files).
    *   Unzip the downloaded file to a location of your choice.
    *   Navigate to the unzipped directory in your command prompt and run the following command to start the Omniverse Editor:
        ```bash
        omni.app.editor.base.bat
        ```

2.  **Install Isaac Sim:**
    *   **Compatibility Check:** Before installing Isaac Sim, it's crucial to run the compatibility checker. Download the `.zip` file for the compatibility checker from the [Isaac Sim download page](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html).
    *   Run the checker to ensure your system meets the [hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html).
    *   **Download Isaac Sim:** If the compatibility check passes, download the Isaac Sim `.zip` file from the same [download page](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html).
    *   **Post-Installation:** After unzipping Isaac Sim, run the appropriate setup script for your operating system. For Windows, run:
        ```bash
        setup.bat
        ```

3.  **Launch Isaac Sim:**
    *   Navigate to the Isaac Sim directory and run the appropriate launch script. For Windows, run:
        ```bash
        isaac-sim.bat
        ```

4.  **Setup ROS 2 Docker Environment:** The standard, reproducible workflow for Isaac ROS development uses a Docker container. The `isaac_ros_common` repository provides a script to facilitate this.

    *   **Clone the repository:**
        ```bash
        # Make sure you are in the root of your project directory
        git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
        ```
    *   **Launch the Development Container (Windows):**
        1.  Navigate to the `scripts` directory in Windows File Explorer:
            `C:\Users\caare\workspace\projects\robotics_workspace\other_robotics_projects\package_integrity_classification_via_sim-to-real\isaac_ros_common\scripts`
        2.  Right-click inside the folder and select "Git Bash Here".
        3.  In the Git Bash terminal that opens, run the following command:
            ```bash
            ./run_dev.sh
            ```

    This command pulls the latest Isaac ROS development container and starts an interactive session, mounting your workspace directories. All subsequent ROS commands should be run from within this container.

> **For Beginners:** It is highly recommended to go through the [Isaac Sim Quickstart Series](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/introduction/quickstart_index.html#isaac-sim-intro-quickstart-series) to get familiar with the Isaac Sim interface and basic concepts.

### Phase 2: Creating Deformed Assets in Isaac Sim

The core creative challenge of this project is generating the "damaged" box assets. Instead of attempting to simulate tearing in real-time, a more practical engineering approach is to use the physics engine as a pre-processing tool to create the assets we need.

> **Note for WSL Users:** If you are running Isaac Sim on Windows but your project files are located within a WSL (Windows Subsystem for Linux) environment, you can access them directly. In the Isaac Sim Script Editor, use `File > Open` and navigate to the WSL path. The path will look similar to this:
> `\\wsl.localhost\Ubuntu-24.04\home\caaren\package_integrity_classification_via_sim-to-real\phase_1_asset_and_scene_creation\scripts\create_pristine_box.py`
> 
> When saving the deformed assets, you must also use this WSL path format in the `dest_path` argument of the `omni.kit.commands.execute('CreateUsd', ...)` function.

#### Step 1: Creating the Pristine Deformable Box


First, create the base asset using the Isaac Lab APIs. This involves defining a mesh and its physical properties. The parameters for `youngs_modulus` and `poissons_ratio` will require some tuning to achieve a cardboard-like behavior; this experimentation is a valuable part of the simulation process. The following Python code can be run in the Isaac Sim Script Editor.

```python
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg

# Configuration for a deformable cube that resembles a cardboard box
cardboard_box_cfg = DeformableObjectCfg(
    prim_path="/World/PristineBox",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.4, 0.3), # Dimensions of a small box in meters
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0,
            contact_offset=0.005
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=5e5,     # A moderately stiff material
            poissons_ratio=0.2,     # Lower ratio, less lateral deformation
            damping_scale=1.0,
            elasticity_damping=0.005
        )
    )
)

# Create the deformable object in the scene
pristine_box = DeformableObject(cfg=cardboard_box_cfg)
```

#### Step 2: Scripting the Deformation

Now, programmatically drop a heavy object onto the box to create damage. After the deformation occurs, pause the simulation and save the resulting mesh as a new USD file.

```python
import omni.timeline
from pxr import Usd, UsdGeom

# --- Assume the pristine_box from Step 1 exists ---

# Create a heavy rigid sphere to act as the damaging object
rigid_sphere_cfg = sim_utils.SphereCfg(
    prim_path="/World/DamagingSphere",
    radius=0.1,
    mass=50.0 # Heavy mass to cause deformation
)
rigid_sphere_cfg.spawn(translation=(0.0, 0.0, 0.5))

# Let the simulation run for a short duration
timeline = omni.timeline.get_timeline_interface()
timeline.play()
# This part requires manual intervention or a more complex script with callbacks.
# For this guide, assume you manually step the simulation forward until the
# sphere hits the box and deforms it, then you pause it.
# await omni.kit.app.get_app().next_update_async() # In a real script

# --- Manually pause the simulation in the UI at the desired deformation ---

# Save the deformed mesh as a new asset
stage = omni.usd.get_context().get_stage()
deformed_prim = stage.GetPrimAtPath("/World/PristineBox")
# Create a new USD file to save the deformed geometry
omni.kit.commands.execute('CreateUsd',
    dest_path='C:/path/to/your/assets/box_dented.usd',
    stage_identifier=deformed_prim.GetPath().pathString)

# Repeat with more mass/higher drop for a 'box_crushed.usd'
```

This process yields the three core assets needed for the next phase: `box_pristine.usd`, `box_dented.usd`, and `box_crushed.usd`.

## Prerequisites

*Details on required software and hardware will be added here.*

## Installation

*Instructions for setting up the project environment will be added here.*

## Usage

*Instructions on how to run each phase of the project will be added here.*
