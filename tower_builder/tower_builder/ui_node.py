import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from std_msgs.msg import Int32, String, Float32MultiArray

class UINode(Node):
    def __init__(self):
        super().__init__("ui_node")

        self.object_id = None
        self.robot_state = "IDLE"
        self.planned_path = []

        self.create_subscription(Int32, "/ui/object_id", self.object_callback, 10)
        self.create_subscription(Float32MultiArray, "/ui/planned_path", self.path_callback, 10)
        self.create_subscription(String, "/ui/robot_state", self.state_callback, 10)

        cv2.namedWindow("Robot UI", cv2.WINDOW_NORMAL)

    def object_callback(self, msg):
        self.object_id = msg.data

    def path_callback(self, msg):
        data = msg.data
        self.planned_path = [(data[i], data[i+1]) for i in range(0, len(data), 3)]

    def state_callback(self, msg):
        self.robot_state = msg.data

    def render(self):
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)

        cv2.putText(canvas, f"State: {self.robot_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if self.object_id is not None:
            cv2.putText(canvas, f"Object ID: {self.object_id}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        for i in range(len(self.planned_path)-1):
            p1 = (int(self.planned_path[i][0]/5 + 400),
                  int(600 - self.planned_path[i][1]/5))
            p2 = (int(self.planned_path[i+1][0]/5 + 400),
                  int(600 - self.planned_path[i+1][1]/5))
            cv2.line(canvas, p1, p2, (0,0,255), 2)

        cv2.imshow("Robot UI", canvas)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = UINode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render()
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
