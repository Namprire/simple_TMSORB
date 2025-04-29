import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class DatasetPublisher(Node):
    def __init__(self):
        super().__init__('dataset_publisher')

        # データセットのパス（必要に応じて変更）
        self.dataset_path = '/opt/dataset/tum/rgbd_dataset_freiburg1_xyz'
        self.rgb_txt_path = os.path.join(self.dataset_path, 'rgb.txt')

        # 画像ファイルの読み込み準備
        self.rgb_list = self.load_rgb_list(self.rgb_txt_path)
        self.rgb_index = 0

        # ROS設定
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # タイマーで画像送信（30Hzくらい）
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("DatasetPublisher initialized. Publishing images...")

    def load_rgb_list(self, file_path):
        rgb_list = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp, filename = parts[0], parts[1]
                    rgb_list.append((float(timestamp), filename))
        return rgb_list

    def timer_callback(self):
        if self.rgb_index >= len(self.rgb_list):
            self.get_logger().info("All images published.")
            rclpy.shutdown()
            return

        timestamp, filename = self.rgb_list[self.rgb_index]
        full_path = os.path.join(self.dataset_path, filename)

        if not os.path.exists(full_path):
            self.get_logger().warn(f"File not found: {full_path}")
            self.rgb_index += 1
            return

        image = cv2.imread(full_path)
        if image is None:
            self.get_logger().warn(f"Failed to read image: {full_path}")
            self.rgb_index += 1
            return

        # OpenCV -> ROS Image メッセージに変換
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher.publish(msg)
        self.get_logger().info(f"Published image: {filename}")
        self.rgb_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DatasetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
