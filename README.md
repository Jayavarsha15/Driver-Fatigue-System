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
# Driver Fatigue Monitoring System – Raspberry Pi Setup Process

---

## Step 1: Download Raspberry Pi Imager

Go to the official Raspberry Pi website and download the Raspberry Pi Imager tool:

```
https://www.raspberrypi.com/software/
```

Install and open the Imager on your computer.

---

## Step 2: Flash the OS to MicroSD Card

Insert your MicroSD card (16GB or above) into your computer.

In the Imager, select the following:

- **Device** → Raspberry Pi 2
- **Operating System** → Raspberry Pi OS (32-bit)
- **Storage** → Select your MicroSD card

Click **Next**.

---

## Step 3: Configure Settings Before Writing

Click **EDIT SETTINGS** before writing the OS.

**General Tab – Set Hostname, Username, Password:**

```
Hostname  : raspberrypi
Username  : pi
Password  : raspberry
```

**Network Tab – Set WiFi:**

```
SSID (WiFi Name) : your_hotspot_or_wifi_name
Password         : your_wifi_password
Country          : IN
```

**Regional Tab:**

```
Timezone        : Asia/Kolkata
Keyboard Layout : us
```

Click **SAVE**, then click **YES** when asked to apply the settings.

Click **WRITE** and wait for the process to complete. Once done, safely eject the MicroSD card.

---

## Step 4: Insert SD Card and Boot Raspberry Pi

Insert the MicroSD card into the Raspberry Pi. Connect the monitor via HDMI, attach the keyboard and mouse, and connect the power adapter. The Raspberry Pi will boot automatically.

---

## Step 5: Login
 Write a commond to log in command ssh (hostename@raspberrypi name ).local
When the login prompt appears, enter the credentials you set during the Imager configuration:

```
Username : pi
Password : raspberry
```

---

## Step 6: Check WiFi Connection

After login, verify that the Raspberry Pi is connected to your WiFi network:

```bash
hostname -I
```

If an IP address is displayed, the connection is successful. If not, run the following to check the WiFi status:

```bash
iwconfig
```

To reconnect or configure WiFi manually:

```bash
sudo raspi-config
```

Navigate to **System Options → Wireless LAN** and enter your SSID and password.

---

## Step 7: Update the System

Before installing anything, update the system packages:

```bash
sudo apt update
sudo apt upgrade -y
```

---

## Step 8: Enable the Camera

Run the configuration tool:

```bash
sudo raspi-config
```

Go to **Interface Options → Camera → Enable**, then select **Finish** and reboot:

```bash
sudo reboot
```

Log in again after reboot using the same credentials.

---

## Step 9: Install Required Libraries

Install pip and all required Python libraries:

```bash
sudo apt install python3-pip -y
pip3 install opencv-python mediapipe numpy imutils
```

Wait for all packages to finish installing.

---

## Step 10: Create the Python File

Navigate to your preferred directory or stay in the home directory. Create the project file using nano:

```bash
nano driver_fatigue_monitor.py
```

---

## Step 11: Paste the Code

Inside the nano editor, paste your complete Python fatigue detection code.

Once pasted, save and close the file:

```
Press CTRL + X
Press Y
Press ENTER
```

---

## Step 12: Run the Program

Run the script using Python 3:

```bash
python3 driver_fatigue_monitor.py
```

The camera will activate and the system will begin monitoring the driver's face in real time for eye closure, yawning, and head posture. An alert will trigger through the buzzer when fatigue is detected.

---
