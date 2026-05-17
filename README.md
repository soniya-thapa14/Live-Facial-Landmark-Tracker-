Facial Landmark Tracker

A real-time facial landmark tracking application that detects a human face 
through a webcam and overlays a 3D mesh of 468 landmarks using 
OpenCV and MediaPipe.

----

What it Does

- Accesses your webcam in real time
- Detects your face using MediaPipe Face Mesh
- Draws a 3D triangular mesh (tessel1ation) over your face
- Draws contour outlines around eyes, eyebrows, lips and face oval
- Shows "No face detected" message when no face is found
- Press Q to quit the application

---
How it Works

1. OpenCV captures live video frames from the webcam
2. Each frame is converted from BGR to RGB for MediaPipe
3. MediaPipe Face Mesh detects 468 facial landmarks with (x, y, z) coordinates
4. OpenCV draws the tesselation and contours using those coordinates
5. The result is displayed in real time at 30 frames per second

---

Tools Used

- Python
- OpenCV
- MediaPipe

---

Installation and Setup

1. Clone the repository
```bash
git clone https://github.com/soniya-thapa14/Live-Facial-Landmark-Tracker-.git
cd Live-Facial-Landmark-Tracker
```


2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```
----


Run The Project
```bash
python landmarker.py
```




