# 🔍 Mini Monocular SLAM with ORB Features

A lightweight Python implementation of a simplified monocular SLAM system inspired by [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2).  
This version focuses on feature-based **tracking** and **sparse 3D mapping** using OpenCV and matplotlib.

<p align="center">
  <img src="preview.gif" width="600"/>
</p>

---

## ✨ Features

- ORB feature detection and matching
- Frame-to-frame pose estimation using essential matrix
- Triangulation of 3D points
- Basic keyframe logic based on motion threshold
- Animated 3D visualization of trajectory and sparse map
- Ground truth comparison (e.g. TUM RGB-D)

---

## 🧰 Requirements

- Python **3.10** recommended  
- Install dependencies:

```bash
pip install numpy opencv-python matplotlib open3d