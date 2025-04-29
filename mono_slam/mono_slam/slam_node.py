import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time
try:
    import tf_transformations
except ImportError:
    import transformations as tf_transformations

class Keyframe:
    def __init__(self, pose, keypoints, descriptors, timestamp):
        self.pose = pose
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.timestamp = timestamp

class MapPoint:
    def __init__(self, position, descriptor):
        self.position = position
        self.descriptor = descriptor
        self.observations = []  # List of (Keyframe, keypoint index)

class ORBSLAMNode(Node):
    def __init__(self):
        super().__init__('orb_slam_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/slam/markers', 10)
        self.path_pub = self.create_publisher(Path, '/camera_path', 10)
        self.keyframe_pub = self.create_publisher(PoseArray, '/keyframes', 10)
        self.map_point_pub = self.create_publisher(MarkerArray, '/map_points', 10)

        self.camera_path = Path()
        self.camera_path.header.frame_id = "map"
        self.keyframes_pose_array = PoseArray()
        self.keyframes_pose_array.header.frame_id = "map"

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = np.array([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]])

        self.keyframes = []
        self.map_points = []
        self.cur_pose = np.eye(4)
        self.prev_kp = None
        self.prev_des = None
        self.lock = threading.Lock()

        self.mapping_thread = threading.Thread(target=self.local_mapping)
        self.mapping_thread.daemon = True
        self.mapping_thread.start()

    def triangulate_points(self, kf1, kf2):
        matches = self.bf.match(kf1.descriptors, kf2.descriptors)
        if len(matches) < 8:
            return []

        pts1 = np.float32([kf1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kf2.keypoints[m.trainIdx].pt for m in matches])
        descriptors = [kf1.descriptors[m.queryIdx] for m in matches]

        P1 = self.K @ kf1.pose[:3]
        P2 = self.K @ kf2.pose[:3]

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d / points_4d[3])[:3].T

        map_points = []
        for i, pt in enumerate(points_3d):
            if not np.isfinite(pt).all():
                continue

            pt_h = np.append(pt, 1.0)
            proj1 = P1 @ pt_h
            proj2 = P2 @ pt_h
            proj1 /= proj1[2]
            proj2 /= proj2[2]

            error1 = np.linalg.norm(proj1[:2] - pts1[i])
            error2 = np.linalg.norm(proj2[:2] - pts2[i])
            reproj_error = (error1 + error2) / 2.0

            if reproj_error < 2.0:
                mp = MapPoint(pt, descriptors[i])
                mp.observations.append((kf1, matches[i].queryIdx))
                mp.observations.append((kf2, matches[i].trainIdx))
                map_points.append(mp)
        return map_points


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_kp is not None:
            matches = self.bf.match(des, self.prev_des)
            if len(matches) >= 8:
                pts1 = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([self.prev_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()
                    with self.lock:
                        self.cur_pose = self.cur_pose @ np.linalg.inv(T)

                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = msg.header.stamp
                        pose_msg.header.frame_id = "map"
                        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = self.cur_pose[:3, 3]
                        quat = tf_transformations.quaternion_from_matrix(self.cur_pose)
                        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat
                        self.pose_pub.publish(pose_msg)

                        self.publish_camera_path(pose_msg)

                        if len(self.keyframes) == 0 or np.linalg.norm(self.keyframes[-1].pose[:3, 3] - self.cur_pose[:3, 3]) > 0.2:
                            self.keyframes.append(Keyframe(self.cur_pose.copy(), kp, des, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))
                            self.publish_keyframes()

                        self.publish_map_markers()
                        self.publish_map_points(msg.header.stamp)

        self.prev_kp = kp
        self.prev_des = des

    def local_mapping(self):
        while rclpy.ok():
            time.sleep(1.0)
            with self.lock:
                if len(self.keyframes) < 2:
                    continue
                kf1 = self.keyframes[-2]
                kf2 = self.keyframes[-1]
                new_map_points = self.triangulate_points(kf1, kf2)
                self.map_points.extend(new_map_points)

    def publish_camera_path(self, pose_msg):
        self.camera_path.header.stamp = pose_msg.header.stamp
        self.camera_path.poses.append(pose_msg)
        self.path_pub.publish(self.camera_path)

    def publish_keyframes(self):
        self.keyframes_pose_array.poses.clear()
        for kf in self.keyframes:
            pose = Pose()
            pose.position.x = kf.pose[0, 3]
            pose.position.y = kf.pose[1, 3]
            pose.position.z = kf.pose[2, 3]
            quat = tf_transformations.quaternion_from_matrix(kf.pose)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
            self.keyframes_pose_array.poses.append(pose)
        self.keyframe_pub.publish(self.keyframes_pose_array)

    def publish_map_points(self, stamp):
        marker_array = MarkerArray()
        for i, mp in enumerate(self.map_points[-500:]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "map_point"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.pose.position.x = float(mp.position[0])
            marker.pose.position.y = float(mp.position[1])
            marker.pose.position.z = float(mp.position[2])

            n_obs = len(mp.observations)
            marker.color.r = max(1.0 - n_obs / 5.0, 0.0)
            marker.color.g = min(n_obs / 5.0, 1.0)
            marker.color.b = 0.0

            marker.lifetime = Duration(sec=0, nanosec=0)
            marker_array.markers.append(marker)

        self.map_point_pub.publish(marker_array)

    def publish_map_markers(self):
        marker_array = MarkerArray()
        for i, kf in enumerate(self.keyframes):
            m = Marker()
            m.header.frame_id = "map"
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.id = i
            m.pose.position.x = kf.pose[0, 3]
            m.pose.position.y = kf.pose[1, 3]
            m.pose.position.z = kf.pose[2, 3]
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ORBSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()




