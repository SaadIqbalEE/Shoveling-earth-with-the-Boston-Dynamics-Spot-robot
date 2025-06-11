# Spot Robot Excavation Simulation – Isaac Sim

This directory contains the final simulation setup and code for demonstrating autonomous **excavation and dumping** behaviors using the **Boston Dynamics Spot robot** in **NVIDIA Isaac Sim**. The robot remains stationary and manipulates a shovel to dig and dump soil using pre-defined motion trajectories.

## 📁 Directory Contents

```text
DiggingFixedRobot/
│
├── Environment_with_containers_nearby.usd       # Main simulation environment
├── 'Garden Shovel - Large.usd'                  # 3D asset of the shovel
├── Rock-7-solid.usd                             # Rock/soil asset used during digging
├── 'calibrate3 (2).usd'                         # Calibration reference submodule
├── spot_with_shovel.usd                         # Spot robot with shovel attached
│
├── materials/                                   # Material assets used in the scene
├── modified_asset/                              # Preprocessed assets (e.g. joint configs)
│
├── arm_trajectory.py                            # Arm motion logic and trajectory planning
├── dig_and_dump_test.py                         # Main simulation entry point
├── install.sh                                   # Environment setup script
├── README.md
```

## ⚙️ Prerequisites

Before running the simulation:

1. Set environment variables by sourcing the `install.sh` script:

   ```bash
   source ./install.sh
   ```

   Ensure the following variables are defined correctly inside `install.sh`:

   * `ISAACSIM_PATH`: Path to your NVIDIA Isaac Sim installation.
   * `ISAACSIM_PYTHON_EXE`: Full path to Isaac Sim's Python executable (e.g. `${ISAACSIM_PATH}/python.sh`).

## ▶️ How to Run

To launch the simulation:

```bash
${ISAACSIM_PYTHON_EXE} dig_and_dump_test.py
```

This will simulate the robot performing digging and dumping operations within the given environment.

## 🌍 Simulation Assets

* **Environment**:
  `Environment_with_containers_nearby.usd`
  This scene includes containers and uses the following sub-assets:

  * `Rock-7-solid.usd` (simulated soil)
  * `'calibrate3 (2).usd'` (used for calibration or positioning)

* **Spot Robot with Shovel**:
  `spot_with_shovel.usd`
  Built with the following dependencies:

  * `'Garden Shovel - Large.usd'`
  * `materials/` directory for rendering realism

## 🧮 Arm Motion & Behavior

**`arm_trajectory.py`** defines a `spot_arm_trajectory` class which includes:

* `setup(articulation)`
  Initializes the Lula C-Space Trajectory Generator and Kinematic Solver with Spot’s URDF and YAML configuration. Also sets the end-effector link.

* `setup_cspace_trajectory(position, dig_type)`
  Accepts digging position (converted from global to local coordinates) and the motion type:

  * `"dig"`: performs digging motion
  * `"dump"`: performs dumping motion
  * `"test"`: runs both consecutively
    The Z-axis is fixed, and only (x, y) are variable. Dumping uses a fixed trajectory.

* `update()`
  Advances the arm one step along the precomputed motion sequence.

* `_teleport_robot_to_position()`
  Currently sets joint velocities for the initial pose.

The arm behavior is defined using data in the `modified_asset/` directory for joint reference.

## 🧪 Main Simulation File

**`dig_and_dump_test.py`** is the core simulation script. It:

* Loads the Spot robot and environment
* Uses `spot_arm_trajectory` to execute motion routines
* Configures asset and robot paths internally
* Cycles through the selected behavior (dig, dump, or both)
