"""Select a destination for the Gazebo course from an interactive terminal."""
import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String


def main():
    if not sys.stdin.isatty():
        raise SystemExit('Open a terminal and run: ros2 run my_car_description_2 select_destination')

    rclpy.init()
    node = rclpy.create_node('course_keyboard')
    publisher = node.create_publisher(String, '/course/destination', 10)
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    print('Press A, B or C to drive to that marker. Press Q to close this terminal controller.', flush=True)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            key = os.read(fd, 1).upper()
            if key in (b'', b'Q'):
                break
            if key in (b'A', b'B', b'C'):
                if publisher.get_subscription_count():
                    publisher.publish(String(data=key.decode('ascii')))
                    print(f'Requested point {key.decode("ascii")}.', flush=True)
                else:
                    print('Waiting for the autonomous course to connect. Try again shortly.', flush=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.try_shutdown()
