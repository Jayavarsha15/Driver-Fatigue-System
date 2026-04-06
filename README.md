# Intelligent Driver Fatigue Monitoring System

This project implements a **real-time Driver Fatigue Monitoring System** using **OpenCV and MediaPipe FaceMesh**. The system analyzes facial behavior and head posture to detect drowsiness and unsafe driving conditions.

It monitors:

* Eye closure (Drowsiness detection)
* Blink rate
* Yawning behavior
* Head posture (Head-down detection)
* Driver distraction
* Continuous fatigue score (0–100)

The system provides **early warnings before fatigue becomes dangerous** and triggers audio alerts during critical conditions such as prolonged eye closure or head-down posture.

---

# How It Works

1. Webcam captures the driver's face in real time

2. MediaPipe detects facial landmarks

3. The system calculates:

   * Eye Aspect Ratio (EAR) → Detects drowsiness
   * Mouth Aspect Ratio (MAR) → Detects yawning
   * Blink rate → Measures alertness
   * Head pose → Detects distraction and collapse
   * PERCLOS → Measures eye closure percentage

4. All signals are combined into a **continuous fatigue score**

5. Driver state is classified as:

* NORMAL
* DISTRACTED
* DROWSY
* HEAD DOWN

6. Audio alert is triggered if dangerous conditions are detected.

---

# System Requirements

* Windows 10 / Windows 11
* Python **3.11.4**
* Webcam
* Minimum 8GB RAM recommended

---

# Installation Guide

Follow these steps to run the project.

---

## Step 1 — Install Python 3.11.4

Download Python 3.11.4 from:

[https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe](https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe)

During installation:

✔ Check **"Add Python to PATH"**

Verify installation:

```bash
python --version
```

Expected output:

```
Python 3.11.4
```

---

## Step 2 — Download Project

Download or clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Driver-Fatigue-Monitoring.git
```

Or download as ZIP and extract.

Open terminal inside the project folder.

---

## Step 3 — Create Virtual Environment

Run:

```bash
python -m venv venv
```

This creates a virtual environment.

---

## Step 4 — Activate Virtual Environment

Run:

```bash
venv\Scripts\activate
```

You should see:

```
(venv)
```

at the beginning of the terminal line.

---

## Step 5 — Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

This installs:

* OpenCV
* MediaPipe
* NumPy
* SciPy
* Playsound
* Imutils

---

## Step 6 — Run the Project

Run:

```bash
python driver_fatigue_monitor.py
```

The webcam will start and monitoring will begin.

Press:

```
q
```

to quit the program.

---

# Project Structure

```
Driver-Fatigue-Monitoring/

driver_fatigue_monitor.py
alarm.wav
requirements.txt
README.md
LICENSE
```

---

# Alerts

Audio alerts are triggered when:

* Driver is drowsy
* Head-down posture is detected
* Fatigue level becomes dangerous

Temporary distractions only affect the fatigue score and do not trigger alarms.

---

# Technologies Used

* Python
* OpenCV
* MediaPipe FaceMesh
* NumPy
* SciPy

---

# Notes

* First run includes automatic calibration.
* Keep face forward and eyes open during calibration.
* Good lighting improves accuracy.

---

If you want, I can give a **"Top 1% GitHub README version"** with badges and visuals.
