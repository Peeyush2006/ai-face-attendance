# AI Attendance System

#### Video Demo: https://youtu.be/G3cJmIWHx50

#### Description

The **AI Attendance System** is a full-stack web application that automates classroom
attendance using **facial recognition**. Instead of a teacher calling out names one by
one, a student simply stands in front of a webcam: the system detects the face,
recognises who it is, and instantly marks that student as *Present* or *Late* in a
database — along with the time and a confidence score.

I built this as my CS50 final project for the BCA department at IGNTU. The goal was to
take everything I had learned in the course — Python, SQL, web development, algorithms,
and a bit of linear algebra — and combine it into one practical tool that a real
department could actually use to save time every morning.

What makes this project a little different from a typical "import a face-recognition
library and call it a day" approach is that **the face-recognition engine is written
from scratch**. I did not use `face_recognition`, `dlib`, or any deep-learning model.
Instead, I implemented the classic **Eigenfaces (Principal Component Analysis)**
algorithm myself, using only NumPy for the linear algebra. This was the most
educational part of the project: I had to understand covariance matrices,
eigenvectors, and dimensionality reduction well enough to code them, not just call
them.

---

## How It Works

The application is split into a **Python backend** (the "brain") and a **browser
frontend** (the "face"), which talk to each other over a small JSON API.

1. **Registration** — A new student's ID, name, and section are entered, and 5 photos
   are captured from the webcam. The backend detects the face in each photo, crops and
   normalises it, and stores it on disk and in the database.
2. **Training** — Once photos exist, the backend trains the Eigenface model on all
   stored faces and saves the result to a compressed `.npz` file so it does not have to
   retrain on every restart.
3. **Live Attendance** — The browser streams webcam frames to the backend ~5 times per
   second. For each frame the backend detects faces, projects them into "face space,"
   finds the closest known student, and — if the match confidence clears the
   configurable threshold — marks attendance and draws a labelled box on the frame,
   which is sent back and displayed live.
4. **Dashboard & Reports** — All attendance data is queried from SQLite to power a
   statistics dashboard, low-attendance alerts, filterable daily reports, and CSV/PDF
   export.

---

## The Face-Recognition Algorithm (Eigenfaces / PCA)

This is the technical heart of the project, all inside `backend/face_rec.py`.

- **Detection:** I use OpenCV's Haar Cascade classifier to *find* where faces are in an
  image. Detection (finding a face) and recognition (identifying whose face it is) are
  two separate problems, and I only use OpenCV for the first one.
- **Preprocessing:** Every detected face is cropped, resized to a fixed 128×128
  grayscale image, and histogram-equalised so that lighting differences matter less.
- **Training (PCA):** Each 128×128 face is flattened into a vector of 16,384 numbers.
  That is far too high-dimensional to compare directly, so I compute the *mean face*,
  subtract it from every image, and then find the principal components (the
  "eigenfaces") that capture the most variation across all the training faces. A key
  trick I used is computing the eigenvectors of the smaller `N×N` matrix
  (`A · Aᵀ`) instead of the enormous `16384×16384` covariance matrix, then mapping
  them back — this is the standard efficiency trick from Turk & Pentland's original
  Eigenfaces paper and makes training practical on a normal laptop.
- **Recognition:** A new face is projected onto the top eigenfaces, producing a short
  vector of ~50 numbers. I compare it to every known student's projection using
  Euclidean distance; the nearest one wins. That distance is then mapped to a friendly
  0–100% confidence score.

Getting this to actually work taught me a lot about *why* dimensionality reduction
matters — comparing 16,384-number vectors directly is both slow and unreliable, while
comparing the 50-number projections is fast and far more robust.

---

## Files

### Backend (`backend/`)

- **`main.py`** — The FastAPI application and the single entry point of the whole
  system. It defines every API route (`/api/dashboard/stats`, `/api/students/register`,
  `/api/process_frame`, `/api/reports`, `/api/settings`, `/api/retrain`), wires the
  recognizer to the database, handles the per-frame recognition loop (including drawing
  the coloured bounding boxes and deciding Present vs. Late), and serves the frontend as
  static files so the whole app runs from one server on port 8000.

- **`face_rec.py`** — The custom machine-learning module. It contains the
  `EigenfaceRecognizer` class (`train`, `predict`, `save`, `load`) implementing PCA from
  scratch with NumPy, plus helper functions for Haar-cascade face detection, face
  preprocessing, and converting images between OpenCV arrays and the base64 strings the
  browser sends. I deliberately kept all the "AI" logic isolated here so it is easy to
  read and test on its own.

- **`database.py`** — Everything related to SQLite. It creates the four tables
  (`students`, `student_photos`, `attendance`, `settings`), provides a clean data-access
  API used by `main.py`, and computes derived statistics (attendance rates, low-
  attendance alerts) with SQL. It also seeds a realistic set of 30 demo students and a
  month of historical attendance so the dashboard is not empty on first run.

### Frontend (`frontend/`)

- **`index.html`** — The single-page interface, organised into five tabbed screens:
  Dashboard, Register, Take Attendance, Reports, and Settings.
- **`app.js`** — All client-side logic: tab routing, accessing the webcam via
  `getUserMedia`, capturing/streaming frames to the backend, rendering the returned
  annotated video, live toast notifications, and CSV export. It also includes a
  *simulated* webcam fallback so the UI still demonstrates well on machines without a
  camera.
- **`styles.css`** — The dark, modern styling for the whole dashboard.

### Supporting files

- **`verify_pca.py`** — A standalone unit test that generates synthetic, clearly-
  separable face patterns, trains the recognizer, saves and reloads the model, and
  asserts 100% recognition accuracy. This let me confirm the PCA maths was correct
  *before* trusting it on real webcam images.
- **`requirements.txt`** — Python dependencies (FastAPI, Uvicorn, NumPy,
  opencv-python-headless, tzdata).
- **`Dockerfile`** — Containerises the app for easy deployment.
- **`start_system.bat`** — A one-click Windows launcher that starts the server and opens
  the browser.
- **`deploy_guide.md`** — Notes on deploying the system.
- **`data/`** — Stores captured face images (`data/faces/`) and the trained model
  (`data/models/model_pca.npz`).

---

## Design Decisions

- **Writing PCA from scratch instead of using a library.** A pre-built face-recognition
  library would have been fewer lines of code, but I would have learned almost nothing.
  Implementing Eigenfaces forced me to genuinely understand the algorithm, and it keeps
  the project lightweight (no heavy deep-learning dependencies).

- **FastAPI + a plain-JavaScript frontend.** I chose FastAPI because it is fast, gives
  automatic request validation through Pydantic, and can serve the static frontend
  itself, so the entire app runs from a single command with no separate frontend build
  step. I kept the frontend in vanilla HTML/CSS/JS rather than a framework like React to
  keep the dependency footprint small and the code readable.

- **SQLite over a client-server database.** For a single-classroom deployment, SQLite is
  perfect: it needs no separate server, the whole database is one file, and it is trivial
  to back up. A `UNIQUE(student_id, date)` constraint on the attendance table guarantees
  a student cannot be marked twice on the same day.

- **A configurable confidence threshold.** Face recognition is never 100% certain, so
  rather than hard-coding a cutoff I exposed the recognition threshold (and the "late"
  cutoff time) in a Settings screen. This lets the operator trade off between strictness
  and convenience for their own lighting and camera conditions.

- **Seeding realistic demo data.** So that the dashboard, reports, and alerts are
  meaningful on a fresh install (and during the demo video), the database seeds a full
  class with a month of plausible attendance history rather than starting blank.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
cd backend
python main.py

# 3. Open the app in a browser
#    http://127.0.0.1:8000
```

On first launch the database is created and seeded automatically. To use live
recognition, register a student (capturing 5 photos), then open the **Take Attendance**
tab.

To run the algorithm's unit test:

```bash
python verify_pca.py
```

---

## Technologies Used

Python · FastAPI · Uvicorn · NumPy · OpenCV (Haar cascades only) · SQLite ·
HTML · CSS · JavaScript
