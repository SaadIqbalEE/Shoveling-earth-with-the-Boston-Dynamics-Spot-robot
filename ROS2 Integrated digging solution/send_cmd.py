import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import tkinter as tk
from tkinter import ttk


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_gui_publisher')
        self.publisher_ = self.create_publisher(String, '/robot_control', 10)

    def send_command(self, command_str):
        msg = String()
        msg.data = command_str
        self.get_logger().info(f'Publishing: {msg.data}')
        self.publisher_.publish(msg)


def ros_spin(node):
    rclpy.spin(node)


def create_gui(node):
    def update_ui():
        mode = selected_mode.get()

        # Disable all
        set_arrow_buttons_state("disabled")
        dig_button.config(state="disabled")
        dump_button.config(state="disabled")
        rotate_cw_button.config(state="disabled")
        rotate_ccw_button.config(state="disabled")
        static_button.config(state="disabled")

        # Enable relevant
        if mode == "Walk":
            set_arrow_buttons_state("normal")
            rotate_cw_button.config(state="normal")
            rotate_ccw_button.config(state="normal")
        elif mode == "Digging":
            dig_button.config(state="normal")
        elif mode == "Dumping":
            dump_button.config(state="normal")
        elif mode == "Static":
            static_button.config(state="normal")

    def set_arrow_buttons_state(state):
        up_button.config(state=state)
        down_button.config(state=state)
        left_button.config(state=state)
        right_button.config(state=state)

    def send_walk_command(direction_code):
        if direction_code == 1:
            command = "walk,1,0,0,0"
        elif direction_code == 2:
            command = "walk,0,1,0,0"
        elif direction_code == 3:
            command = "walk,0,0,1,0"
        elif direction_code == 4:
            command = "walk,0,0,0,1"
        else:
            command = "walk,0,0,0,0"
            
        node.send_command(command)

    def send_rotate_command(direction):
        if direction == "cw":
            node.send_command("rot_cw,0,0,0,0")
        elif direction == "ccw":
            node.send_command("rot_ccw,0,0,0,0")

    def send_dig_command():
        node.send_command("dig,0,0,0,0")

    def send_dump_command():
        node.send_command("dmp,0,0,0,0")

    def send_static_command():
        node.send_command("stat,0,0,0,0")

    root = tk.Tk()
    root.title("SPOT ROBOT MANUAL CONTROL")

    selected_mode = tk.StringVar()
    selected_mode.set("Walk")

    modes = [("Walk", "Walk"), ("Digging", "Digging"), ("Dumping", "Dumping"), ("Static", "Static")]

    for text, value in modes:
        ttk.Radiobutton(root, text=text, value=value, variable=selected_mode, command=update_ui).pack(anchor="w")

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

    dig_button = ttk.Button(root, text="Execute Digging", command=send_dig_command)
    dig_button.pack(pady=5)

    dump_button = ttk.Button(root, text="Execute Dumping", command=send_dump_command)
    dump_button.pack(pady=5)

    static_button = ttk.Button(root, text="Execute Static", command=send_static_command)
    static_button.pack(pady=5)

    update_ui()
    root.mainloop()


def main():
    rclpy.init()
    node = RobotController()

    # Start ROS spin in a background thread
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        # Start GUI (blocking)
        create_gui(node)
    finally:
        # Clean up on GUI close
        node.get_logger().info('Shutting down ROS...')
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)  # Give time for thread to exit
        node.destroy_node()


if __name__ == '__main__':
    main()
