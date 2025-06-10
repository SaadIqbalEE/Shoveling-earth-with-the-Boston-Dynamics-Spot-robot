# Robot Simulation Environment Setup

This repository contains two Isaac Sim USD environments designed for robotic simulation and sand interaction using PhysX particles. The environments simulate a controlled sandbox area, incorporating friction adjustments, particle-based sand, and physical obstacles.

## Environment Overview

The simulation environment is based on the Isaac Sim scene `warehouse_with_forklifts`, with all forklifts deactivated. The scene includes:

- A sandbox with four primitive walls (scaled cube meshes).
- Simulated sand using PhysX particle systems.
- Imported obstacle elements (shovel and rock) as STL files.
- A Spot robot model with friction-tuned ground interaction.

### Files in the `environments` Folder

- **Environment_with_obstacles.usd**  
  Contains the rock obstacle placed in the center of the sandbox. Sandbox walls are spaced in the cardinal directions.

- **Environment_with_containers_nearby.usd**  
  Similar to the above, but the sandbox walls (container-like) are positioned closer together.

- **Garden Shovel - Large.usd**  
  USD-converted garden shovel asset used as an obstacle/tool.

- **Rock-7-solid.usd**  
  Rock formation asset used for obstacle placement.

- **spot_arm_plastic_shovel.usd**  
  A plastic shovel attached to the Spot robot's arm.

### Supporting Folders

These three subfolders contain all the dependencies required for the environments and assets to function properly in Isaac Sim:

- **materials/**  
  Contains all material definitions (e.g., textures, shaders) used across assets.

- **modified_asset/**  
  Holds customized versions of assets, such as collision-mesh-altered models or adjusted properties for simulation.

- **spot_arm/**  
  Includes dependencies specific to the Spot robot’s arm.

## Key Setup Details

### Ground Plane Friction
To ensure proper traction for the Spot robot:
- **Dynamic Friction:** `1.9`
- **Static Friction:** `1.3`

### Sandbox Construction
- Built from four identical scaled cube primitives:
  - **Scale:** X: `1`, Y: `0.05`, Z: `0.4`
- Positioned to form a sandbox boundary.
- Collision physics enabled without rigid body dynamics, allowing interaction while remaining static.

### Imported Meshes
- **Shovel:** Garden shovel from Printables.
- **Rock:** Rock formation mesh from Printables.

> ⚠️ Note: The shovel’s collision mesh is box-like and does not follow its curved geometry, reducing the accuracy of sand pickup.

## PhysX Particle Sand Simulation

Isaac Sim’s GPU-accelerated Position-Based Dynamics (PBD) particle system is used to simulate sand-like granular behavior.

### Particle Configuration
- **Type:** Particle Set with `PointsInstancer` prims
- **Sampler:** Particle Sampler used on a mesh to generate initial positions
- **Solid Rest Offset:** `0.015` (controls particle spacing)

### Particle Offsets
- **Contact Offset:** `0.02`
- **Rest Offset:** `0.02`

### Neighbourhood Settings
- **Max Neighbourhood:** `300` (default is `96`)
- **Neighbourhood Scale:** `1.2`

### Solver Quality
- **Iteration Count:** `64`

### Physics Material (Sand)
- **Density:** `2000`
- **Friction:** `0.6`
- **Adhesion:** `0.000001`
- **Adhesion Offset Scale:** `1.2`
- **Particle Adhesion Scale:** `10000`
- **Particle Friction Scale:** `0.2`

For questions or further improvements, please consult the Isaac Sim documentation on [PhysX particles](https://docs.omniverse.nvidia.com/isaacsim/latest/physics/physx-particles.html).
