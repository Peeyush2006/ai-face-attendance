# AI Face Recognition Attendance System
#### Video Demo: [https://youtu.be/G3cJmIWHx50?si=RwHiRqq1lv_ybXpH]
#### Description:

## What is this project?

AI Face Recognition Attendance System is a web-based application that uses real-time facial recognition to automatically mark student or employee attendance. Instead of manual roll calls or paper registers, this system identifies faces through a webcam and records attendance instantly in a database.

This project was built as my final project for CS50x 2026.

---

## Why I built this?

Manual attendance systems are slow, error-prone, and easy to manipulate (proxy attendance). I wanted to build something that solves a real-world problem using AI. Face recognition seemed like the perfect solution — fast, accurate, and contactless.

---

## Features

- **Face Registration** — Register new users by capturing their face via webcam
- **Real-time Face Recognition** — Automatically identifies registered faces using OpenCV
- **Automatic Attendance Marking** — Records attendance with timestamp in SQLite database
- **Attendance Dashboard** — View attendance records by date, student, or subject
- **Flask Web Interface** — Clean browser-based UI, no desktop app needed
- **Docker Support** — Fully containerized for easy deployment
- **Deployed on Render** — Live demo accessible online

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3 | Core language |
| Flask | Web framework |
| OpenCV | Face detection & recognition |
| face_recognition library | Face encoding & matching |
| SQLite | Database for attendance records |
| Docker | Containerization |
| Render | Cloud deployment |

---

## Project Structure

```
ai-face-attendance/
├── app.py                  # Main Flask application
├── helpers.py              # Face recognition helper functions
├── attendance.db           # SQLite database
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── static/
│   └── styles.css          # Styling
└── templates/
    ├── layout.html         # Base template
    ├── index.html          # Dashboard / home
    ├── register.html       # Face registration page
    ├── attendance.html     # Attendance records view
    └── camera.html         # Live camera feed page
```

---

## How It Works

1. **Register Phase** — User opens the register page, enters their name, and captures 5 face photos via webcam. The system encodes the face and stores it in the database.

2. **Recognition Phase** — On the attendance page, webcam feed starts. OpenCV detects faces in real-time, compares them with stored encodings, and if a match is found (above 60% confidence), attendance is marked automatically.

3. **Dashboard** — All attendance records are displayed in a table with name, date, time, and status.

---

## Design Decisions

**Why Flask over Django?**
Flask is lightweight and perfect for a focused single-purpose app. Django would have been overkill for this project size.

**Why SQLite over MySQL/PostgreSQL?**
CS50 introduced me to SQLite and it's perfect for a self-contained project. No external database server needed.

**Why Docker?**
OpenCV has complex dependencies (`libgl1`, `libglib2.0` etc.) that cause deployment issues. Docker ensures the environment is consistent everywhere.

**Challenge faced — UTC Timezone:**
Attendance timestamps were showing UTC time instead of IST (Indian Standard Time). Fixed this using `pytz` library to convert and display correct local time.

**Challenge faced — Render deployment:**
Free tier on Render doesn't support persistent storage, so the database resets on redeploy. Solved this by using an in-memory demo mode for the live deployment.

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Peeyush2006/ai-face-attendance.git
cd ai-face-attendance

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open browser at `http://localhost:5000`

**Or with Docker:**
```bash
docker build -t face-attendance .
docker run -p 5000:5000 face-attendance
```

---

## Live Demo

Deployed on Render: [Add your Render URL here]

---

## What I learned from CS50

CS50 taught me how to think like a programmer — breaking problems into smaller pieces, debugging systematically, and writing clean code. This project combines everything I learned:
- Week 6 (Python) — Core logic
- Week 7 (SQL) — Database design
- Week 8 (HTML/CSS/JS) — Frontend
- Week 9 (Flask) — Web framework

**This was CS50x!** 🎓

---

## Author

**Peeyush Tiwari**
BCA Student — IGNTU Amarkantak, Madhya Pradesh
GitHub: [Peeyush2006](https://github.com/Peeyush2006)
