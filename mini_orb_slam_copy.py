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
    valid_points = []
    
    for pt in pts3d.T:
        # 거리 기반 필터링
        dist = np.linalg.norm(pt)
        if 0.1 < dist < 5.0:
            # 재투영 오차 기반 필터링
            proj_pt = CAMERA_MATRIX @ pt
            if proj_pt[2] > 0:  # 카메라 앞에 있는 점만
                valid_points.append(pt)
    
    if not valid_points:  # 유효한 점이 없으면
        return np.array([]).reshape(3, 0)  # 명시적으로 (3,0) shape 지정
    return np.array(valid_points).T

def transform_points(points, pose):
    """안전하게 3D 점들을 변환"""
    if points.size == 0:
        return np.array([])
    if points.shape[1] == 0:
        return np.array([])
    return (pose[:3, :3] @ points + pose[:3, 3:4]).T

def draw_matches(img1, kp1, img2, kp2, matches):
    # Create a visualization of matching features between two frames
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return match_img

def visualize_frame_results(curr_frame, curr_kp, matches_img):
    # Show current frame with detected keypoints
    kp_img = cv2.drawKeypoints(curr_frame, curr_kp, None, 
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display results
    cv2.imshow('Current Frame Keypoints', kp_img)
    cv2.imshow('Feature Matches', matches_img)
    cv2.waitKey(1)

def plot_trajectory_and_map_animated(trajectory, map_points):
    trajectory = np.array(trajectory)
    map_points = np.array(map_points)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SLAM Results')
    scat = ax.scatter([], [], [], c='red', s=1, label='Map Points')
    line, = ax.plot([], [], [], c='blue', label='Camera Trajectory')
    ax.legend()

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

        ax.view_init(elev=20, azim=i)  # Rotate view for better visualization
        return scat, line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                init_func=init, interval=50, blit=False)
    
    # Save as GIF
    ani.save('slam_result_2.gif', writer='pillow', fps=20)
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

def calculate_matching_stats(matches, pts1, pts2):
    # Calculate statistics about feature matching
    if len(matches) == 0:
        return None
    
    distances = [m.distance for m in matches]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Calculate spatial distribution of matches
    pts1_std = np.std(pts1, axis=0)
    pts2_std = np.std(pts2, axis=0)
    spatial_coverage = np.mean([pts1_std, pts2_std])
    
    return {
        'num_matches': len(matches),
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'spatial_coverage': spatial_coverage
    }

def calculate_trajectory_smoothness(trajectory):
    if len(trajectory) < 3:
        return None
    
    # Calculate velocity and acceleration
    trajectory = np.array(trajectory)
    velocities = np.diff(trajectory, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    # Calculate smoothness metrics
    mean_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    mean_acceleration = np.mean(np.linalg.norm(accelerations, axis=1))
    
    return {
        'mean_velocity': mean_velocity,
        'mean_acceleration': mean_acceleration
    }

def analyze_map_points(map_points):
    if len(map_points) == 0:
        return None
    
    points = np.array(map_points)
    
    # Calculate spatial distribution
    mean_pos = np.mean(points, axis=0)
    std_pos = np.std(points, axis=0)
    
    # Calculate density
    distances = np.linalg.norm(points - mean_pos, axis=1)
    point_density = len(points) / (np.max(distances) ** 3)
    
    return {
        'num_points': len(points),
        'mean_position': mean_pos,
        'std_position': std_pos,
        'point_density': point_density
    }

def add_map_points(new_points, map_points, min_distance=0.1):
    """Add new points to the map if they are far enough from existing points"""
    if len(map_points) == 0:  # 맵이 비어있으면 모든 점 추가
        map_points.extend(new_points)
        return

    # Convert to numpy arrays for efficient computation
    new_points = np.array(new_points)
    existing_points = np.array(map_points)
    
    for pt in new_points:
        # Compute distances to all existing points
        distances = np.linalg.norm(existing_points - pt, axis=1)
        # If point is far enough from all existing points, add it
        if np.min(distances) >= min_distance:
            map_points.append(pt.tolist())

def get_next_file_number():
    # Check existing files and get the next number
    existing_files = os.listdir('.')
    max_num = 0
    
    # Check slam result files
    for file in existing_files:
        if file.startswith('slam_result_') and file.endswith('.gif'):
            try:
                num = int(file.split('_')[2].split('.')[0])
                max_num = max(max_num, num)
            except:
                continue
                
    return max_num + 1

# -- MAIN PIPELINE --
def run_mini_slam():
    # Get the next file number
    file_num = get_next_file_number()
    slam_result_file = f'slam_result_{file_num}.gif'
    feature_video_file = f'feature_matching_{file_num}.mp4'
    feature_gif_file = f'feature_matching_{file_num}.gif'
    
    print("Starting Mini-SLAM...")
    orb = cv2.ORB_create(2000)
    images = load_images(IMAGE_DIR)
    print(f"Loaded {len(images)} images")

    # Initialize video writer for feature matching visualization
    first_img = images[0]
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    feature_video = cv2.VideoWriter(feature_video_file, fourcc, 10.0, (w*2, h))

    # Performance tracking
    matching_stats = []
    num_frames_processed = 0
    num_keyframes = 0
    tracking_lost_count = 0

    pose = np.eye(4)
    trajectory = [pose[:3, 3]]
    map_points = []

    last_kf_pose = pose.copy()
    last_kf_img = images[0]
    last_kf_kp, last_kf_des = extract_features(last_kf_img, orb)

    MIN_TRANSLATION = 0.2
    MIN_ROTATION = 0.1
    MIN_MATCHES = 20

    # Create windows for visualization
    cv2.namedWindow('Current Frame Keypoints', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)

    print("Processing frames...")
    for i in range(1, len(images)):
        print(f"Processing frame {i}/{len(images)-1}", end='\r')
        num_frames_processed += 1
        
        curr_img = images[i]
        curr_kp, curr_des = extract_features(curr_img, orb)
        if curr_des is None or last_kf_des is None:
            tracking_lost_count += 1
            continue

        matches = match_features(last_kf_des, curr_des)
        if len(matches) < MIN_MATCHES:
            print(f"\n[Frame {i}] Not enough matches.")
            tracking_lost_count += 1
            continue

        # Calculate matching statistics
        pts1 = np.float32([last_kf_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        stats = calculate_matching_stats(matches, pts1, pts2)
        if stats:
            matching_stats.append(stats)

        # Visualize matches and save to video
        matches_img = draw_matches(last_kf_img, last_kf_kp, curr_img, curr_kp, matches[:30])
        feature_video.write(matches_img)
        visualize_frame_results(curr_img, curr_kp, matches_img)

        R, t, pts1, pts2 = estimate_pose(last_kf_kp, curr_kp, matches)
        if R is None or t is None:
            tracking_lost_count += 1
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
        
        # Transform points safely
        if pts3d.size > 0 and pts3d.shape[1] > 0:
            pts3d_world = transform_points(pts3d, pose)
        else:
            pts3d_world = np.array([])

        # Only insert keyframe and map points if motion is large enough
        translation_movement = np.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3])
        rotation_movement = np.arccos((np.trace(pose[:3, :3] @ last_kf_pose[:3, :3].T) - 1) / 2)

        if translation_movement > MIN_TRANSLATION or rotation_movement > MIN_ROTATION:
            print(f"\n[Frame {i}] Added keyframe. Translation Δ = {translation_movement:.2f}, Rotation Δ = {rotation_movement:.2f}")
            num_keyframes += 1
            last_kf_pose = pose.copy()
            last_kf_img = curr_img
            last_kf_kp, last_kf_des = curr_kp, curr_des
            if pts3d_world.size > 0:
                if len(pts3d_world.shape) == 2:
                    add_map_points(pts3d_world.tolist(), map_points)
            trajectory.append(pose[:3, 3])
        else:
            print(f"\n[Frame {i}] Movement too small: T Δ = {translation_movement:.3f}, R Δ = {rotation_movement:.3f}")

    # Release video writer and windows
    feature_video.release()
    cv2.destroyAllWindows()

    # Calculate final statistics
    trajectory_stats = calculate_trajectory_smoothness(trajectory)
    map_stats = analyze_map_points(map_points)
    
    # Average matching statistics
    avg_matching_stats = {
        'num_matches': np.mean([s['num_matches'] for s in matching_stats]),
        'mean_distance': np.mean([s['mean_distance'] for s in matching_stats]),
        'spatial_coverage': np.mean([s['spatial_coverage'] for s in matching_stats])
    }

    # Print performance evaluation
    print("\n=== SLAM Performance Evaluation ===")
    print("\n1. Tracking Statistics:")
    print(f"   - Total frames processed: {num_frames_processed}")
    print(f"   - Number of keyframes: {num_keyframes}")
    print(f"   - Tracking lost count: {tracking_lost_count}")
    print(f"   - Tracking success rate: {(1 - tracking_lost_count/num_frames_processed)*100:.1f}%")
    
    print("\n2. Feature Matching Quality:")
    print(f"   - Average number of matches: {avg_matching_stats['num_matches']:.1f}")
    print(f"   - Average matching distance: {avg_matching_stats['mean_distance']:.2f}")
    print(f"   - Average spatial coverage: {avg_matching_stats['spatial_coverage']:.2f}")
    
    if trajectory_stats:
        print("\n3. Trajectory Smoothness:")
        print(f"   - Mean velocity: {trajectory_stats['mean_velocity']:.3f}")
        print(f"   - Mean acceleration: {trajectory_stats['mean_acceleration']:.3f}")
    
    if map_stats:
        print("\n4. Map Quality:")
        print(f"   - Total map points: {map_stats['num_points']}")
        print(f"   - Point density: {map_stats['point_density']:.2f}")
        print(f"   - Spatial distribution (std): {np.mean(map_stats['std_position']):.2f}")

    print("\nGenerating visualizations...")
    
    # Convert lists to numpy arrays for visualization
    trajectory_array = np.array(trajectory)
    map_points_array = np.array(map_points)
    
    # Save animated trajectory and map
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SLAM Results')
    scat = ax.scatter([], [], [], c='red', s=1, label='Map Points')
    line, = ax.plot([], [], [], c='blue', label='Camera Trajectory')
    ax.legend()

    def init():
        scat._offsets3d = ([], [], [])
        line.set_data([], [])
        line.set_3d_properties([])
        return scat, line

    def update(i):
        if i == 0:
            return init()

        traj_i = trajectory_array[:i+1]
        if len(map_points_array) > 0:
            map_i = map_points_array[:min(i*10, len(map_points_array))]
        else:
            map_i = np.array([])

        if len(traj_i) > 0:
            line.set_data(traj_i[:, 0], traj_i[:, 1])
            line.set_3d_properties(traj_i[:, 2])

        if len(map_i) > 0:
            scat._offsets3d = (map_i[:, 0], map_i[:, 1], map_i[:, 2])

        ax.view_init(elev=20, azim=i)
        return scat, line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                init_func=init, interval=50, blit=False)
    
    # Save as GIF with numbered filename
    ani.save(slam_result_file, writer='pillow', fps=20)
    plt.show()
    
    print("\nResults saved as:")
    print(f"1. '{slam_result_file}' - 3D trajectory and map visualization")
    print(f"2. '{feature_video_file}' - Feature matching visualization")
    
    print("\nConverting feature matching video to GIF...")
    os.system(f"ffmpeg -i {feature_video_file} -vf 'fps=10,scale=800:-1' -y {feature_gif_file}")
    print(f"3. '{feature_gif_file}' - Feature matching visualization (GIF format)")

if __name__ == "__main__":
    run_mini_slam()