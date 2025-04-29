import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation

# -- SETTINGS --
IMAGE_DIR = "rgbd_dataset_freiburg1_xyz/rgb"
GROUND_TRUTH_PATH = "rgbd_dataset_freiburg1_xyz/groundtruth.txt"
CAMERA_MATRIX = np.array([[517.3, 0, 318.6],
                          [0, 516.5, 255.3],
                          [0, 0, 1]])

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
    return matches[:100]

def estimate_pose(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, CAMERA_MATRIX, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None or mask.sum() < 20:
        return None, None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, CAMERA_MATRIX)
    return R, t, pts1[mask_pose.ravel() == 1], pts2[mask_pose.ravel() == 1]

def triangulate_filtered(R, t, kp1, kp2):
    proj1 = CAMERA_MATRIX @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = CAMERA_MATRIX @ np.hstack((R, t))

    # Ensure 2xN shape and float32 type
    if kp1.ndim != 2 or kp2.ndim != 2 or kp1.shape[0] < 2:
        return np.empty((0, 3))  # return empty if data is malformed

    kp1 = kp1.T.astype(np.float32)  # Shape: (2, N)
    kp2 = kp2.T.astype(np.float32)  # Shape: (2, N)

    if kp1.shape[1] < 8 or kp2.shape[1] < 8:
        return np.empty((0, 3))  # not enough points to triangulate

    pts4d = cv2.triangulatePoints(proj1, proj2, kp1, kp2)
    pts3d = pts4d[:3] / pts4d[3]  # Convert from homogeneous
    pts3d = pts3d.T  # Shape: (N, 3)

    # Filter out bad points
    valid = pts3d[:, 2] > 0
    valid = np.logical_and(valid, np.linalg.norm(pts3d, axis=1) < 10)

    return pts3d[valid]

def load_ground_truth(path):
    poses = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            time, x, y, z, qx, qy, qz, qw = map(float, tokens)
            poses.append([x, y, z])
    return np.array(poses)


def load_ground_truth_aligned(gt_path, image_folder):
    import glob
    from bisect import bisect_left

    # Step 1: Load ground truth
    gt_data = {}
    with open(gt_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            t, x, y, z, qx, qy, qz, qw = map(float, tokens)
            gt_data[t] = [x, y, z]

    gt_times = sorted(gt_data.keys())

    # Step 2: Get image timestamps
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    image_times = [float(os.path.splitext(os.path.basename(f))[0]) for f in image_files]

    # Step 3: Find closest ground truth pose for each image timestamp
    aligned_poses = []
    for t_img in image_times:
        idx = bisect_left(gt_times, t_img)
        if idx == 0 or idx >= len(gt_times):
            continue
        # Get closer one of two adjacent timestamps
        t1, t2 = gt_times[idx - 1], gt_times[idx]
        t_gt = t1 if abs(t_img - t1) < abs(t_img - t2) else t2
        aligned_poses.append(gt_data[t_gt])

    return np.array(aligned_poses)


def plot_trajectory_and_map_animated(trajectory, map_points, ground_truth=None, save_path=None):
    trajectory = np.array(trajectory)
    map_points = np.array(map_points)
    if ground_truth is not None:
        ground_truth = np.array(ground_truth)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    scat = ax.scatter([], [], [], c='red', s=1)
    traj_line, = ax.plot([], [], [], c='blue', label='Estimated Trajectory')

    if ground_truth is not None and len(ground_truth) > 1:
        gt_line, = ax.plot([], [], [], c='green', label='Ground Truth')
    else:
        gt_line = None

    ax.legend()

    def init():
        scat._offsets3d = ([], [], [])
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        if gt_line:
            gt_line.set_data([], [])
            gt_line.set_3d_properties([])
        return scat, traj_line, gt_line

    def update(i):
        traj_i = trajectory[:i+1]
        map_i = map_points[:i*10]
        traj_line.set_data(traj_i[:, 0], traj_i[:, 1])
        traj_line.set_3d_properties(traj_i[:, 2])

        if gt_line and i < len(ground_truth):
            gt_i = ground_truth[:i+1]
            gt_line.set_data(gt_i[:, 0], gt_i[:, 1])
            gt_line.set_3d_properties(gt_i[:, 2])

        if len(map_i) > 0:
            scat._offsets3d = (map_i[:, 0], map_i[:, 1], map_i[:, 2])

        return scat, traj_line, gt_line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, interval=100, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=10)  # Save as GIF

    plt.tight_layout()
    plt.show()

# -- MAIN PIPELINE --
def run_mini_slam(ground_truth=None):
    orb = cv2.ORB_create(2000)
    images = load_images(IMAGE_DIR)

    pose = np.eye(4)
    trajectory = [pose[:3, 3]]
    map_points = []

    last_kf_pose = pose.copy()
    last_kf_img = images[0]
    last_kf_kp, last_kf_des = extract_features(last_kf_img, orb)

    MIN_TRANSLATION = 0.1

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

        if np.linalg.norm(t) > 0:
            t = t / np.linalg.norm(t) * 0.2

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        pose = last_kf_pose @ np.linalg.inv(T)

        pts3d = triangulate_filtered(R, t, pts1, pts2)
        pts3d_world = (pose[:3, :3] @ pts3d.T + pose[:3, 3:4]).T

        translation_movement = np.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3])
        if translation_movement > MIN_TRANSLATION:
            print(f"[Frame {i}] Keyframe added. Δ = {translation_movement:.2f}")
            last_kf_pose = pose.copy()
            last_kf_img = curr_img
            last_kf_kp, last_kf_des = curr_kp, curr_des
            map_points.extend(pts3d_world.tolist())
            trajectory.append(pose[:3, 3])
        else:
            print(f"[Frame {i}] Too small movement: Δ = {translation_movement:.3f}")

    print(f"✅ Keyframes: {len(trajectory)}, Map points: {len(map_points)}")
    plot_trajectory_and_map_animated(trajectory, map_points, ground_truth)






if __name__ == "__main__":
    ground_truth = load_ground_truth_aligned("rgbd_dataset_freiburg1_xyz/groundtruth.txt",
                                              "rgbd_dataset_freiburg1_xyz/rgb")
    print(f"Loaded {len(ground_truth)} aligned ground truth poses.")
    run_mini_slam(ground_truth)
