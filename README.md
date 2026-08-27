# 🏠 Indoor Scene Change Detection

> **Object Detection + Change Classification for Indoor Environments**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLO](https://img.shields.io/badge/YOLO-11s-green)
![RT--DETR](https://img.shields.io/badge/RT--DETR-l-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project addresses the task of **indoor scene change detection** using a two-stage approach:

1. **Object Detection** – YOLO11s and RT-DETR models trained on our custom indoor dataset to detect 56 object classes.
2. **Change Classification** – A rule-based matching algorithm that compares bounding boxes between Before/After image pairs to classify changes as **Add**, **Delete**, **Move**, **Open**, **Close**, **ON**, **OFF** .

The system is designed for robotic and forensic applications where an agent needs to detect and revert changes in indoor environments.

---

## 📊 Dataset

### Statistics
| Metric | Value |
| :--- | :--- |
| **Total Images** | 2000+ |
| **Object Classes** | 56 |
| **Change Types** | Add, Delete, Move, Open, Close, ON, OFF |
| **Room Types** | Bedroom, Kitchen, Dining, Corridor, Drawing |
| **Resolution** | 1920×1080 (collected), 640×640 (training) |
