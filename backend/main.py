import os
import datetime
import uvicorn
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

import face_rec
import database

# Initialize database
database.init_db()

# Create directories for face images and models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, 'data', 'faces')
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'model_pca.npz')

app = FastAPI(title="AI Attendance System API")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recognizer instance
recognizer = face_rec.EigenfaceRecognizer(num_components=50, img_size=(128, 128))

def load_and_train_recognizer(force: bool = False):
    global recognizer
    # Try loading pre-trained model first if not forced
    if not force and os.path.exists(MODEL_PATH):
        if recognizer.load(MODEL_PATH):
            print("Loaded pre-trained PCA model successfully.")
            return True
            
    # Train a new model from photos database & data/faces directory
    print("Training PCA model from database photos and faces directory...")
    faces_list = []
    labels_list = []
    seen_paths = set()
    import cv2
    
    # 1. Check database records
    photos = database.get_all_student_photos()
    for p in photos:
        path = p['photo_path']
        sid = p['student_id']
        
        # Resolve path if absolute path moved
        if not os.path.exists(path):
            fname = os.path.basename(path)
            candidate = os.path.join(FACES_DIR, sid.replace('/', '_'), fname)
            if os.path.exists(candidate):
                path = candidate
                
        real_path = os.path.normpath(os.path.abspath(path))
        if os.path.exists(path) and real_path not in seen_paths:
            seen_paths.add(real_path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_eq = cv2.equalizeHist(img)
                faces_list.append(img_eq)
                labels_list.append(sid)
                
    # 2. Also scan data/faces directory for any student face sets
    if os.path.exists(FACES_DIR):
        for sid_folder in os.listdir(FACES_DIR):
            folder_path = os.path.join(FACES_DIR, sid_folder)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, fname)
                        real_path = os.path.normpath(os.path.abspath(img_path))
                        if real_path not in seen_paths:
                            seen_paths.add(real_path)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_eq = cv2.equalizeHist(img)
                                faces_list.append(img_eq)
                                labels_list.append(sid_folder)
                            
    if len(faces_list) > 0:
        if recognizer.train(faces_list, labels_list):
            recognizer.save(MODEL_PATH)
            print(f"PCA Model trained successfully on {len(faces_list)} images for {len(set(labels_list))} students and saved.")
            return True
    print("PCA Model training failed (insufficient images).")
    return False

# Pydantic schemas
class RegisterRequest(BaseModel):
    student_id: str
    name: str
    course_section: Optional[str] = ""
    photos: List[str]  # 5 base64 encoded images

class SettingsUpdateRequest(BaseModel):
    threshold: float
    late_threshold: int
    camera_source: str
    email_alerts: bool
    database_path: str

# API Routes

@app.on_event("startup")
def startup_event():
    load_and_train_recognizer()

@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    try:
        return database.get_dashboard_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/today")
def get_today_attendance():
    try:
        return database.get_today_attendance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/students/register")
def register_student(req: RegisterRequest):
    if len(req.photos) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 photos are required.")
    
    # Clean student ID (replace slashes for filename safety)
    safe_sid = req.student_id.replace('/', '_')
    student_dir = os.path.join(FACES_DIR, safe_sid)
    os.makedirs(student_dir, exist_ok=True)
    
    # Save student in database
    success = database.add_student(req.student_id, req.name, req.course_section)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save student record to database.")
        
    saved_photos_count = 0
    import cv2
    import numpy as np
    
    for idx, b64_img in enumerate(req.photos):
        try:
            img = face_rec.base64_to_cv2(b64_img)
            if img is None:
                continue
            faces = face_rec.detect_faces(img)
            
            # If no face detected, fallback to center crop for webcam photo
            if len(faces) == 0:
                h, w = img.shape[:2]
                faces = [(int(w * 0.15), int(h * 0.1), int(w * 0.7), int(h * 0.8))]
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                bbox = faces[0]
                face_cropped = face_rec.preprocess_face(img, bbox)
                
                photo_path = os.path.join(student_dir, f"face_{idx+1}.png")
                cv2.imwrite(photo_path, face_cropped)
                
                # Add to DB
                database.add_student_photo(req.student_id, photo_path)
                saved_photos_count += 1
        except Exception as e:
            print(f"Error processing registration image {idx+1}: {e}")
            
    if saved_photos_count < 3:
        # Rollback or warning, but let's require at least 3 successful face encodings
        raise HTTPException(status_code=400, detail=f"Could only detect faces in {saved_photos_count}/5 photos. Please capture again in better lighting.")
        
    # Force clean retrain of model with the new student's photos
    load_and_train_recognizer(force=True)
    
    return {"message": f"Student registered successfully. {saved_photos_count} face encodings saved."}

@app.post("/api/process_frame")
def process_frame(payload: dict = Body(...)):
    frame_b64 = payload.get("frame")
    if not frame_b64:
        raise HTTPException(status_code=400, detail="No frame data provided.")
        
    import cv2
    try:
        img = face_rec.base64_to_cv2(frame_b64)
        if img is None:
            return {"face_detected": False, "recognitions": [], "recognition": None, "annotated_frame": frame_b64}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_rec.detect_faces(img)
        
        # System settings
        settings = database.get_settings()
        threshold_setting = float(settings.get('threshold', 0.50))
        late_threshold_mins = int(settings.get('late_threshold', 15))
        
        face_detected = len(faces) > 0
        recognitions = []
        
        # Determine current time & late status
        tz = ZoneInfo("Asia/Kolkata")
        now = datetime.datetime.now(tz)
        class_start = now.replace(hour=10, minute=15, second=0, microsecond=0)
        late_cutoff = class_start + datetime.timedelta(minutes=late_threshold_mins)
        status = 'Late' if now > late_cutoff else 'Present'
        time_str = now.strftime('%H:%M')
        
        conn = database.get_db_connection()
        
        # 1. First pass: predict each face candidate
        predictions = []
        for bbox in faces:
            x, y, w, h = bbox
            face_img = face_rec.preprocess_face(gray, bbox)
            student_id, distance, confidence = recognizer.predict(face_img, threshold=threshold_setting)
            predictions.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "student_id": student_id,
                "distance": distance,
                "confidence": confidence
            })
            
        # 2. Enforce 1-to-1 matching:
        # A student cannot appear as two distinct physical faces in the same camera frame.
        # Prioritize assigning the student ID to the face with the highest confidence.
        assigned_students = set()
        sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True)
        final_assignments = {}
        
        for idx in sorted_indices:
            pred = predictions[idx]
            sid = pred["student_id"]
            conf = pred["confidence"]
            
            if sid and conf >= threshold_setting and sid not in assigned_students:
                assigned_students.add(sid)
                final_assignments[idx] = (sid, conf)
            else:
                final_assignments[idx] = (None, conf)
                
        # 3. Draw annotations and record attendance for uniquely matched faces
        for idx, pred in enumerate(predictions):
            x, y, w, h = pred["bbox"]
            matched_sid, conf = final_assignments[idx]
            
            if matched_sid:
                row = conn.execute('SELECT name, course_section FROM students WHERE student_id = ?', (matched_sid,)).fetchone()
                name = row['name'] if row else matched_sid
                course_section = row['course_section'] if row else ""
                
                # Mark attendance once per recognized student
                database.mark_attendance(matched_sid, status, time_str, conf * 100)
                
                rec_info = {
                    "student_id": matched_sid,
                    "name": name,
                    "course_section": course_section,
                    "time": time_str,
                    "confidence": f"{round(conf * 100, 1)}%",
                    "status": status,
                    "bbox": [x, y, w, h]
                }
                recognitions.append(rec_info)
                
                # Draw green box
                cv2.rectangle(img, (x, y), (x+w, y+h), (76, 175, 80), 2)
                cv2.rectangle(img, (x, y - 25), (x+w, y), (76, 175, 80), -1)
                cv2.putText(img, f"{name} ({round(conf * 100, 1)}%)", (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Draw red box for Unknown or unassigned face
                cv2.rectangle(img, (x, y), (x+w, y+h), (244, 67, 54), 2)
                cv2.rectangle(img, (x, y - 25), (x+w, y), (244, 67, 54), -1)
                cv2.putText(img, "Unknown", (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
        conn.close()
        
        annotated_b64 = face_rec.cv2_to_base64(img)
        return {
            "face_detected": face_detected,
            "recognitions": recognitions,
            "recognition": recognitions[0] if recognitions else None,
            "annotated_frame": annotated_b64
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"face_detected": False, "recognitions": [], "recognition": None, "annotated_frame": frame_b64}

# Global hardware camera handle
hardware_cap = None

def get_hardware_capture():
    global hardware_cap
    import cv2
    if hardware_cap is None or not hardware_cap.isOpened():
        settings = database.get_settings()
        src_str = settings.get('camera_source', '0')
        try:
            src_idx = int(src_str)
        except ValueError:
            src_idx = src_str
            
        # DirectShow on Windows avoids camera driver hangs and dropped frames
        if isinstance(src_idx, int) and os.name == 'nt' and hasattr(cv2, 'CAP_DSHOW'):
            hardware_cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
        else:
            hardware_cap = cv2.VideoCapture(src_idx)
            
        if hardware_cap is not None and hardware_cap.isOpened():
            hardware_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            hardware_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            hardware_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return hardware_cap

def read_hardware_frame():
    cap = get_hardware_capture()
    if cap is None or not cap.isOpened():
        return None
    # Retry up to 3 times to gracefully absorb transient dropped frames
    for _ in range(3):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    return None

@app.get("/api/camera/raw_frame")
def capture_raw_frame():
    """Raw camera frame for registration preview without bounding boxes or attendance side-effects"""
    frame = read_hardware_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from hardware camera.")
    b64_frame = face_rec.cv2_to_base64(frame)
    return {"frame": b64_frame}

@app.get("/api/camera/native_frame")
def capture_native_frame():
    frame = read_hardware_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from hardware camera.")
        
    b64_frame = face_rec.cv2_to_base64(frame)
    # Process frame through standard detection & recognition pipeline
    return process_frame({"frame": b64_frame})

@app.post("/api/camera/release")
def release_hardware_camera():
    global hardware_cap
    if hardware_cap is not None:
        try:
            hardware_cap.release()
        except Exception:
            pass
        hardware_cap = None
    return {"message": "Hardware camera released."}

@app.get("/api/settings")
def get_settings():
    settings = database.get_settings()
    return {
        "threshold": float(settings.get("threshold", 0.50)),
        "late_threshold": int(settings.get("late_threshold", 15)),
        "camera_source": settings.get("camera_source", "0"),
        "email_alerts": settings.get("email_alerts", "1") == "1",
        "database_path": settings.get("database_path", "./attendance.db")
    }

@app.post("/api/settings")
def save_settings(req: SettingsUpdateRequest):
    settings_dict = {
        "threshold": str(req.threshold),
        "late_threshold": str(req.late_threshold),
        "camera_source": req.camera_source,
        "email_alerts": "1" if req.email_alerts else "0",
        "database_path": req.database_path
    }
    database.update_settings(settings_dict)
    return {"message": "Settings updated successfully."}

@app.get("/api/reports")
def get_reports(date: str, course: str = "All"):
    try:
        return database.get_report_data(date, course)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
def retrain_model():
    success = load_and_train_recognizer(force=True)
    if not success:
        raise HTTPException(status_code=400, detail="Training failed. Make sure you have registered students with photos.")
    return {"message": "Model retrained successfully."}

# Mount frontend files as static resources
frontend_path = os.path.join(BASE_DIR, 'frontend')
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"WARNING: Frontend path {frontend_path} not found. Static files won't be served.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
