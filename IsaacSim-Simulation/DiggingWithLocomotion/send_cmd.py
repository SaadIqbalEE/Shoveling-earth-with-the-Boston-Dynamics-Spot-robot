import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import tkinter as tk
from tkinter import ttk


class RobotController(Node):
    """
    ROS2 Node that publishes robot control commands to the '/robot_control' topic.
    """
    def __init__(self):
        super().__init__('robot_gui_publisher')
        self.publisher_ = self.create_publisher(String, '/robot_control', 10)

    def send_command(self, command_str):
        """
        Publishes a command string to the '/robot_control' topic.
        """
        msg = String()
        msg.data = command_str
        self.get_logger().info(f'Publishing: {msg.data}')
        self.publisher_.publish(msg)


def ros_spin(node):
    """
    Starts the ROS2 spinning loop for the provided node.
    Required to process callbacks in a separate thread.
    """
    rclpy.spin(node)


def create_gui(node):
    """
    Creates the GUI using tkinter to control the robot.
    GUI elements are conditionally enabled based on selected operation mode.
    """
    def update_ui():
        """
        Updates button states based on the selected mode.
        """
        mode = selected_mode.get()

        # Disable all control buttons
        set_arrow_buttons_state("disabled")
        dig_button.config(state="disabled")
        dump_button.config(state="disabled")
        rotate_cw_button.config(state="disabled")
        rotate_ccw_button.config(state="disabled")

        # Enable controls relevant to the selected mode
        if mode == "Walk":
            set_arrow_buttons_state("normal")
            rotate_cw_button.config(state="normal")
            rotate_ccw_button.config(state="normal")
        elif mode == "Digging":
            dig_button.config(state="normal")
        elif mode == "Dumping":
            dump_button.config(state="normal")

    def set_arrow_buttons_state(state):
        """
        Enables or disables directional arrow buttons.
        """
        up_button.config(state=state)
        down_button.config(state=state)
        left_button.config(state=state)
        right_button.config(state=state)

    def send_walk_command(direction_code):
        """
        Sends a walk command with a direction code:
        1: up, 2: left, 3: down, 4: right.
        """
        direction_map = {
            1: "walk,1,0,0,0",
            2: "walk,0,1,0,0",
            3: "walk,0,0,1,0",
            4: "walk,0,0,0,1"
        }
        command = direction_map.get(direction_code, "walk,0,0,0,0")
        node.send_command(command)

    def send_rotate_command(direction):
        """
        Sends a rotation command (cw or ccw).
        """
        if direction == "cw":
            node.send_command("rot_cw,0,0,0,0")
        elif direction == "ccw":
            node.send_command("rot_ccw,0,0,0,0")

    def send_dig_command():
        """
        Sends the digging command.
        """
        node.send_command("dig,0,0,0,0")

    def send_dump_command():
        """
        Sends the dumping command.
        """
        node.send_command("dmp,0,0,0,0")

    # Initialize the root window
    root = tk.Tk()
    root.title("SPOT ROBOT MANUAL CONTROL")

    # Mode selection radio buttons
    selected_mode = tk.StringVar()
    selected_mode.set("Walk")

    modes = [("Walk", "Walk"), ("Digging", "Digging"), ("Dumping", "Dumping")]

    for text, value in modes:
        ttk.Radiobutton(root, text=text, value=value, variable=selected_mode, command=update_ui).pack(anchor="w")

    # Direction and rotation buttons
    arrow_frame = ttk.Frame(root)
    arrow_frame.pack(pady=10)

    rotate_ccw_button = ttk.Button(arrow_frame, text="↺", command=lambda: send_rotate_command("ccw"))
    rotate_ccw_button.grid(row=0, column=0, padx=5, pady=5)

    up_button = ttk.Button(arrow_frame, text="↑", command=lambda: send_walk_command(1))
    up_button.grid(row=0, column=1, padx=5, pady=5)

    rotate_cw_button = ttk.Button(arrow_frame, text="↻", command=lambda: send_rotate_command("cw"))
    rotate_cw_button.grid(row=0, column=2, padx=5, pady=5)

    left_button = ttk.Button(arrow_frame, text="←", command=lambda: send_walk_command(2))
    left_button.grid(row=1, column=0, padx=5, pady=5)

    down_button = ttk.Button(arrow_frame, text="↓", command=lambda: send_walk_command(3))
    down_button.grid(row=1, column=1, padx=5, pady=5)

    right_button = ttk.Button(arrow_frame, text="→", command=lambda: send_walk_command(4))
    right_button.grid(row=1, column=2, padx=5, pady=5)

    # Action buttons
    dig_button = ttk.Button(root, text="Execute Digging", command=send_dig_command)
    dig_button.pack(pady=5)

    dump_button = ttk.Button(root, text="Execute Dumping", command=send_dump_command)
    dump_button.pack(pady=5)

    # Initialize button states based on default mode
    update_ui()

    # Start GUI event loop
    root.mainloop()


def main():
    """
    Initializes ROS2, creates the GUI, and starts the ROS spinning in a background thread.
    Cleans up resources on exit.
    """
    rclpy.init()
    node = RobotController()

    # Run ROS spin loop in a separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        # Launch GUI (blocking call)
        create_gui(node)
    finally:
        # Cleanup on GUI close
        node.get_logger().info('Shutting down ROS...')
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
