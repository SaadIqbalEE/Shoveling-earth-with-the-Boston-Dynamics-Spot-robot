# Robot Simulation Environment Setup

This repository contains two Isaac Sim USD environments designed for robotic simulation and sand interaction using PhysX particles. The environments simulate a controlled sandbox area, incorporating friction adjustments, particle-based sand, and physical obstacles.

## Environment Overview

The simulation environment is based on the Isaac Sim scene `warehouse_with_forklifts`, with all forklifts deactivated. The scene includes:

- A sandbox with four primitive walls (scaled cube meshes).
- Simulated sand using PhysX particle systems.
- Imported obstacle elements (shovel and rock) as STL files.
- A Spot robot model with friction-tuned ground interaction.

### Files

- `Environment_with_obstacles.usd`  
  Contains the rock obstacle placed in the center of the sandbox.

- `Environment_without_obstacles.usd`  
  Same as above but without the rock in the filled sandbox.

## Key Setup Details

### Ground Plane Friction
To ensure proper traction for the Spot robot:
- **Dynamic Friction:** `1.9`
- **Static Friction:** `1.3`

### Sandbox Construction
- Built from four scaled cube primitives:
  - **Scale:** X: `1`, Y: `0.05`, Z: `0.4`
- Positioned at cardinal directions to form a boundary.
- Collision physics enabled without rigid body dynamics (static interaction).

### Imported Meshes
- **Shovel:** Garden shovel from Printables.
- **Rock:** Rock formation mesh from Printables.

> Note: The shovel's collision mesh approximates a box, limiting the accuracy of sand pickup.

## PhysX Particle Sand Simulation

Isaac Sim’s GPU-accelerated PBD particle system simulates dry sand behavior.

### Particle Configuration
- **Type:** Particle Set with `PointsInstancer` prims
- **Sampler:** Particle Sampler used on mesh
- **Solid Rest Offset (spacing):** `0.015`

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
