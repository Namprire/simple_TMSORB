# mono_slam/trajectory_logger_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')
        self.path_pub = self.create_publisher(Path, '/trajectory', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/slam/pose', self.pose_callback, 10)

        self.path = Path()
        self.path.header.frame_id = "map"
        self.trajectory_file = open('/home/robotech/trajectory.txt', 'w')

    def pose_callback(self, msg: PoseStamped):
        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(msg)
        self.path_pub.publish(self.path)

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        o = msg.pose.orientation
        self.trajectory_file.write(f"{t:.6f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {o.x:.6f} {o.y:.6f} {o.z:.6f} {o.w:.6f}\n")

    def destroy_node(self):
        self.trajectory_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
