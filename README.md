# 8-DoF Homography Tracker

A Lucas–Kanade + Gauss–Newton SSD tracker that follows a user-selected ROI under perspective change, rotation, and scale by estimating an **8-DoF homography**. Includes a simple 2-DoF translation baseline for comparison.

<p align="center">
  <img src="tracking.gif" alt="8-DoF Homography Tracker demo" width="1000">
</p>

---

## Repository Layout

- `code/`
  - `1a.py` — 2-DoF SSD tracker on an image pair (select ROI, track into next frame)
  - `1b.py` — live 2-DoF SSD tracker (webcam / live capture)
  - `1c.py` — 2-DoF SSD tracker on a recorded video + export annotated result
  - `3_read_vid.py` — **8-DoF homography tracker** on a recorded video + export annotated result
  - `3_read_vid_seg.py` — alternate homography demo setup / reads from image sequence

- `bus/`
  - image-sequence frames used for quick testing (e.g., `bus-0000000xxx.pgm`)

- `example_videos/`
  - sample videos (e.g., `bird.mp4`, `robot.mp4`)

- `media/`
  - output videos with tracked ROI overlaid
  - typical outputs:
    - `2DoF_tracked_video_1.mp4`, `2DoF_tracked_video_2.mp4`
    - `homography_tracking.mp4`, `homography_tracking_2.mp4`

---

### 1. 2-DoF SSD Tracker (Baseline)

#### 1a) Track ROI on an Image Pair
- **Use:** `code/1a.py`

**What it does:**
- Opens two frames (image0 -> image1)
- Lets you select an ROI on the first frame
- Iteratively estimates translation and draws the updated ROI on the next frame

**Expected output:**
- A window showing the ROI updated on the second image
---

#### 1b) Live 2-DoF Tracking (Webcam)
- **Use:** `code/1b.py`

**What it does:**
- Captures frames from your camera
- Select an ROI once, then track it live with a 2-DoF translation warp

**Expected output:**
- Live window with the tracked bounding box drawn on each frame

---

#### 1c) 2-DoF Tracking on a Recorded Video
- **Use:** `code/1c.py`

**What it does:**
- Reads a video file
- Tracks a selected ROI frame-to-frame (2-DoF translation)
- Writes an annotated output video into `media/`

**Expected output:**
- An exported annotated video (e.g., `media/2DoF_tracked_video_*.mp4`)

---

### 2. High-DoF Tracker (8-DoF Homography)

#### 3) 8-DoF Homography Tracking on a Recorded Video
- **Use:** `code/3_read_vid.py`

**What it does:**
- Reads a video file
- Select an ROI on the first frame (template)
- Iteratively refines an **8-DoF homography** using Lucas–Kanade style linearization + Gauss–Newton updates
- Writes an annotated output video into `media/`

**Expected output:**
- An exported annotated video (e.g., `media/homography_tracking.mp4`)

---

#### 3) Alternate Homography Demo / Second Test
- **Use:** `code/3_read_vid_seg.py`

**What it does:**
- Same tracker, takes different input method (reads image sequence)

**Expected output:**
- Another annotated output (e.g., `media/homography_tracking_2.mp4`)

---

## How to Run

### Requirements
- Python 3.9+ recommended
- Packages:
  - `opencv-python`
  - `numpy`

Install:
- `pip install opencv-python numpy`

### Run a script
From the repository root:

- `python code/1a.py`
- `python code/1b.py`
- `python code/1c.py`
- `python code/3_read_vid.py`
- `python code/3_read_vid_seg.py`

> Most scripts assume hard-coded input paths (like `bus/`, `example_videos/...`, or a file under `media/`).
> If your filenames differ, change the config variables at the top of the script.

---

## Demo Outputs (What to Look At)

Open the exported results in `media/`:

- `2DoF_tracked_video_1.mp4`, `2DoF_tracked_video_2.mp4`
  - Baseline translation tracker: decent for small motion and mostly planar translation, but it drifts when the ROI tilts or scales.

- `homography_tracking.mp4`, `homography_tracking_2.mp4`
  - 8-DoF homography tracker: stays aligned under perspective tilt + rotation + scale, where 2-DoF typically loses lock.

---

## License and Credits

- CMPUT 428/615 Computer Vision Lab 1.2 (2025)
- University of Alberta, Department of Computing Science
- Student work by **Shijie (Bob) Bu**

<div align="right">

<img src="UofAlbertalogo.svg" alt="University of Alberta Logo" width="330px" style="vertical-align: middle;">
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Department of Computing Science
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Spring 2025, Edmonton, AB, Canada
</p>

</div>
