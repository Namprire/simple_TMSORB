import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib import animation

# -- SETTINGS --
IMAGE_DIR = "rgbd_dataset_freiburg1_xyz/rgb"  # Set this!
CAMERA_MATRIX = np.array([[517.3, 0, 318.6],
                          [0, 516.5, 255.3],
                          [0, 0, 1]])  # fx, fy, cx, cy from TUM dataset

# -- HELPER FUNCTIONS --
def load_images(folder):
    image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    return [cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE) for f in image_files]

def extract_features(img, orb):
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:100]  # limit for speed

def estimate_pose(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, CAMERA_MATRIX, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, CAMERA_MATRIX)
    return R, t, pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]

def triangulate(R, t, kp1, kp2):
    proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = np.hstack((R, t))
    proj1 = CAMERA_MATRIX @ proj1
    proj2 = CAMERA_MATRIX @ proj2

    pts4d = cv2.triangulatePoints(proj1, proj2, kp1.T, kp2.T)
    pts3d = pts4d[:3] / pts4d[3]
    return pts3d.T


def plot_trajectory_and_map_animated(trajectory, map_points):
    trajectory = np.array(trajectory)
    map_points = np.array(map_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    scat = ax.scatter([], [], [], c='red', s=1)
    line, = ax.plot([], [], [], c='blue')

    def init():
        scat._offsets3d = ([], [], [])
        line.set_data([], [])
        line.set_3d_properties([])
        return scat, line

    def update(i):
        if i == 0:
            return init()

        traj_i = trajectory[:i+1]
        map_i = map_points[:i*10]  # reduce number of points for smoother animation

        line.set_data(traj_i[:, 0], traj_i[:, 1])
        line.set_3d_properties(traj_i[:, 2])

        if len(map_i) > 0:
            scat._offsets3d = (map_i[:, 0], map_i[:, 1], map_i[:, 2])

        return scat, line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, interval=100, blit=False)

    plt.tight_layout()
    #ani.save("slam_animation.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
    plt.show()

def plot_trajectory_and_map(trajectory, map_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*np.array(trajectory).T, label="Camera Trajectory", color='blue')
    ax.scatter(*np.array(map_points).T, c='red', s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()

# -- MAIN PIPELINE --
def run_mini_slam():
    orb = cv2.ORB_create(2000)
    images = load_images(IMAGE_DIR)

    pose = np.eye(4)
    trajectory = [pose[:3, 3]]
    map_points = []

    last_kf_pose = pose.copy()
    last_kf_img = images[0]
    last_kf_kp, last_kf_des = extract_features(last_kf_img, orb)

    MIN_TRANSLATION = 0.1  # Insert keyframe if moved > 10 cm

    for i in range(1, len(images)):
        curr_img = images[i]
        curr_kp, curr_des = extract_features(curr_img, orb)
        if curr_des is None or last_kf_des is None:
            continue

        matches = match_features(last_kf_des, curr_des)
        if len(matches) < 20:
            print(f"[Frame {i}] Not enough matches.")
            continue

        R, t, pts1, pts2 = estimate_pose(last_kf_kp, curr_kp, matches)
        if R is None or t is None:
            continue

        # Normalize translation and apply artificial scale
        if np.linalg.norm(t) > 0:
            t = t / np.linalg.norm(t) * 0.2

        # Compute current pose relative to last keyframe
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        pose = last_kf_pose @ np.linalg.inv(T)

        # Triangulate
        pts3d = triangulate(R, t, pts1, pts2)
        pts3d_world = (pose[:3, :3] @ pts3d.T + pose[:3, 3:4]).T

        # Only insert keyframe and map points if motion is large enough
        translation_movement = np.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3])
        if translation_movement > MIN_TRANSLATION:
            print(f"[Frame {i}] Added keyframe. Translation Δ = {translation_movement:.2f}")
            last_kf_pose = pose.copy()
            last_kf_img = curr_img
            last_kf_kp, last_kf_des = curr_kp, curr_des
            map_points.extend(pts3d_world.tolist())
            trajectory.append(pose[:3, 3])
        else:
            print(f"[Frame {i}] Movement too small: Δ = {translation_movement:.3f}")

    print(f"Total keyframes: {len(trajectory)}")
    print(f"Total map points: {len(map_points)}")

    plot_trajectory_and_map_animated(trajectory, map_points)


if __name__ == "__main__":
    run_mini_slam()