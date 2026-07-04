## Video Demo

YouTube Video: https://youtu.be/G3cJmIWHx50

In this demonstration, I showcase the complete workflow of the AI Face Attendance System, including face registration, facial recognition, automatic attendance marking, and attendance record generation.
# AI Face Attendance System

## Overview

AI Face Attendance System is a Python-based application that automates attendance management using facial recognition technology. The system captures faces through a webcam, identifies registered users, and automatically records attendance with date and time information. This eliminates manual attendance processes and improves accuracy and efficiency.

This project was developed as the final project for Harvard University's CS50 course.

## Features

* Face detection using computer vision techniques
* Face recognition using trained facial encodings
* Automatic attendance marking
* Attendance records stored in CSV format
* Real-time webcam integration
* User registration and dataset generation
* Simple and user-friendly interface

## Technologies Used

* Python
* OpenCV
* face_recognition
* NumPy
* Pandas
* CSV File Handling

## Project Structure

```text
ai-face-attendance/
│
├── main.py
├── attendance.csv
├── requirements.txt
├── README.md
├── dataset/
├── trainer/
└── images/
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Peeyush2006/ai-face-attendance.git
cd ai-face-attendance
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

## How It Works

1. Register user face images.
2. The system extracts facial features and creates face encodings.
3. Webcam captures live video.
4. Detected faces are compared with stored encodings.
5. When a match is found, attendance is automatically recorded.
6. Attendance data is saved with name, date, and timestamp.

## Challenges Faced

One of the main challenges was achieving reliable face recognition under different lighting conditions and camera angles. Another challenge was preventing duplicate attendance entries while maintaining real-time performance.

## Future Improvements

* Database integration (SQLite/MySQL)
* Cloud-based attendance storage
* Mobile application support
* Multiple camera support
* Email attendance reports
* Face mask detection

## Author

Peeyush Tiwari

## License

This project is created for educational purposes as part of the CS50 Final Project.
