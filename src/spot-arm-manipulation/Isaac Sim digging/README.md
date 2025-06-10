This folder contains the arm manipulation code and the final code versions for excavation and dumping simulation. The robot's position is fixed. 

In arm_trajectory.py class spot_arm_trajectory has functions: 

setup(self, articulation) which takes in the Spot articulation and initializes the LulaCSpaceTrajectoryGenerator object and LulaKinematicSolver object which both take the .urdf and .yaml file locations as argument. Also, the end effector link name is defined. 

setup_cspace_trajectory(self, position, dig_type) which takes in the digging position as a Spot’s local coordinate and the arm action type. In our simulation code, the digging position is defined in global coordinates first and then transformed into local coordinates to use it as an argument. “dig” performs digging, “dump” dumping, and “test” both consecutively. This function generates a predefined-shape trajectory using the timestamped Lula C-Space Generator and the Lula Kinematics Solver. As of now, the digging motion happens only in a fixed global +x direction but will adjust the digging position based on the argument position. Z position of digging is also fixed. Therefore, in the position argument (x, y, z) only (x, y) are taken into account. Dumping is always a fixed trajectory. 

update(self) which applies the next action from the action sequence created by setup_cspace_trajectory to Spot articulation. This function is called on every step in the simulation when executing a trajectory. This function moves the arm one step forward. 

_teleport_robot_to_position(self, articulation_action) which as of now just sets joint velocities on the first step of the trajectory. 

dig_and_dump_test.py is for simulating excavation and dumping. By launching this file in the Linux terminal with the Python executable python.sh in the Isaac Sim folder you can simulate excavation and dumping earth:  

$ ISAACSIMPATH/python.sh CODEPATH/dig_and_dump_test.py 

The correct .usd paths for the environment and the Spot model must be defined in dig_and_dump_test.py and correct .urdf and .yaml paths for Spot in arm_trajectory.py.  

dig_and_dump_test.py calls the functions of spot_arm_trajectory class in arm_trajectory.py. The main loop executes digging, dumping, both or remains static based on what is defined as the arm action.