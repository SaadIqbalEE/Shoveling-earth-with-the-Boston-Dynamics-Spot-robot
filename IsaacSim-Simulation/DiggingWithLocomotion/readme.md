# 🤖 Digging with Locomotion: Boston Dynamics Spot in Isaac Sim

This repository provides a ROS2-integrated simulation for autonomous digging using the Boston Dynamics Spot robot, built on NVIDIA Isaac Sim. The simulation enables real-time interaction with the robot's locomotion and arm control via ROS2 topics, offering stereo camera feedback, joint state monitoring, and a dynamic environment with terrain elements.

---

## 📁 Project Structure

**Directory:**  
`/Shoveling-earth-with-the-Boston-Dynamics-Spot-robot/IsaacSim-Simulation/DiggingWithLocomotion`

**Contents:**

| File/Folder                      | Description                                           |
|----------------------------------|-------------------------------------------------------|
| `spot_digging.py`               | Main simulation script                                |
| `send_cmd.py`                   | ROS2 interface for issuing commands                   |
| `arm_trajectory.py`             | Defines digging arm motion and trajectory logic       |
| `spot_policy.py`                | Locomotion controller using pretrained model          |
| `install.sh`                    | Environment setup script                              |
| `Environment_with_obstacles.usd`| Main simulation scene                                 |
| `Rock-7-solid.usd`              | Simulated soil/rock                                   |
| `Garden Shovel - Large.usd`     | Shovel asset                                          |
| `spot_arm_plastic_shovel.usd`   | Combined Spot + shovel model                          |
| `materials/`                    | Visual materials for simulation realism               |
| `modified_asset/`               | Custom configuration for arm kinematics               |
| `spot_arm/`                     | Contains locomotion models and parameters             |

---

## 🛠️ Prerequisites & Setup

Before running the simulation, configure the environment by sourcing the setup script:

```bash
source ./install.sh
```

Ensure the following environment variables are correctly defined inside `install.sh`:

- `ISAACSIM_PATH`: Path to your NVIDIA Isaac Sim installation
- `ISAACSIM_PYTHON_EXE`: Path to the Isaac Sim Python launcher (e.g., `${ISAACSIM_PATH}/python.sh`)

---

## 🚀 Running the Simulation

### 1. Start ROS2 Control Interface

Open a terminal and launch the interactive ROS2 interface:

```bash
python ./send_cmd.py
```

This creates the topic `robot_control`, through which users can send dig, dump, and test commands.

### 2. Launch the Simulation

In another terminal, run:

```bash
${ISAACSIM_PYTHON_EXE} spot_digging.py
```

This will:

- Load the Spot robot and environment
- Activate locomotion and arm motion
- Respond to commands from the ROS2 `robot_control` topic
- Publish robot state and stereo imagery

---

## 📡 ROS2 Interfaces

| Topic                | Purpose                                 |
|----------------------|-----------------------------------------|
| `/robot_control`     | Accepts commands (`dig`, `dump`, `test`) |
| `/spot/world_pose`   | Publishes robot’s global position       |
| `/spot/joint_states` | Publishes joint angles and velocities   |

Additionally, `rqt` can be used to visualize stereo camera feeds from the robot's front camera.

---

## 🌍 Simulation Environment & Assets

### Environment

- `Environment_with_obstacles.usd`: Base scene containing terrain and props
- Includes:
  - `Rock-7-solid.usd`: Simulated diggable soil
  - `calibrate3 (2).usd`: Calibration object for alignment

### Robot Model

- `spot_arm_plastic_shovel.usd`: Spot robot with attached shovel
  - Combines:
    - `Garden Shovel - Large.usd`
    - `materials/` for realistic visual appearance

---

## 🧠 Arm Motion Logic

The file `arm_trajectory.py` defines the `spot_arm_trajectory` class, responsible for shovel arm kinematics.

**Key Methods:**

- `setup(articulation)`: Initializes the trajectory generator and solver
- `setup_cspace_trajectory(position, dig_type)`:
  - `"dig"`: Executes digging motion
  - `"dump"`: Executes dumping motion
  - `"test"`: Executes both consecutively
- `update()`: Advances arm along the precomputed trajectory
- `_teleport_robot_to_position()`: Initializes arm pose

Joint configurations are based on data from the `modified_asset/` folder.

---

## 🦿 Locomotion Control

Locomotion is controlled via a pretrained model:

- **Model path:** `spot_arm/models/spot_arm_policy.pt`
- **Parameter file:** Located in `spot_arm/params/`

The `spot_policy.py` script defines `SpotFlatTerrainPolicy`, which updates joint velocities to ensure stable and smooth movement across terrain.

---

## 🧪 Core Simulation Script

The file `spot_digging.py` orchestrates the simulation. It:

- Loads the Spot robot and shovel asset
- Initializes the environment and digging logic
- Interfaces with ROS2 for real-time robot control
- Publishes world pose, joint states, and camera data

---

## 📝 License

This project is intended for academic and research purposes. For licensing details regarding Isaac Sim or Spot assets, please refer to their respective documentation.

---

## 📬 Contact

For issues, questions, or contributions, please open an issue in the repository or contact the project maintainers directly.
