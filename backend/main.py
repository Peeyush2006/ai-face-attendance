import os
import datetime
import uvicorn
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

def load_and_train_recognizer():
    global recognizer
    # Try loading pre-trained model first
    if os.path.exists(MODEL_PATH):
        if recognizer.load(MODEL_PATH):
            print("Loaded pre-trained PCA model successfully.")
            return True
            
    # If not found or failed, train a new model from photos database
    print("Training PCA model from database photos...")
    photos = database.get_all_student_photos()
    if not photos:
        print("No student photos found in database. PCA training skipped.")
        return False
        
    faces_list = []
    labels_list = []
    
    for p in photos:
        path = p['photo_path']
        sid = p['student_id']
        if os.path.exists(path):
            # Load grayscale image
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Just in case, equalize hist
                img_eq = cv2.equalizeHist(img)
                faces_list.append(img_eq)
                labels_list.append(sid)
                
    if len(faces_list) > 0:
        if recognizer.train(faces_list, labels_list):
            recognizer.save(MODEL_PATH)
            print(f"PCA Model trained successfully on {len(faces_list)} images and saved.")
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_rec.detect_faces(gray)
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                bbox = faces[0]
                face_cropped = face_rec.preprocess_face(gray, bbox)
                
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
        
    # Retrain model with new student
    load_and_train_recognizer()
    
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
            return {"face_detected": False, "annotated_frame": frame_b64}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_rec.detect_faces(gray)
        
        # System settings
        settings = database.get_settings()
        threshold_setting = float(settings.get('threshold', 0.50))
        late_threshold_mins = int(settings.get('late_threshold', 15))
        
        face_detected = len(faces) > 0
        recognition_result = None
        
        for bbox in faces:
            x, y, w, h = bbox
            face_img = face_rec.preprocess_face(gray, bbox)
            
            # Predict face
            student_id, distance = recognizer.predict(face_img)
            
            # Distance mapping to confidence score:
            # 0 distance -> 1.0 (100% confidence)
            # 5000 distance -> 0.0 (0% confidence)
            confidence = max(0.0, 1.0 - distance / 5000.0)
            
            # Draw overlay on BGR image
            # Green for recognized, red for unknown
            if student_id and confidence >= threshold_setting:
                # Retrieve student name
                conn = database.get_db_connection()
                row = conn.execute('SELECT name, course_section FROM students WHERE student_id = ?', (student_id,)).fetchone()
                conn.close()
                
                name = row['name'] if row else "Student"
                course_section = row['course_section'] if row else ""
                
                # Check late status
                # Standard class start at 10:15 AM
                now = datetime.datetime.now()
                class_start = now.replace(hour=10, minute=15, second=0, microsecond=0)
                late_cutoff = class_start + datetime.timedelta(minutes=late_threshold_mins)
                
                if now > late_cutoff:
                    status = 'Late'
                else:
                    status = 'Present'
                    
                time_str = now.strftime('%H:%M')
                
                # Mark attendance
                database.mark_attendance(student_id, status, time_str, confidence * 100)
                
                recognition_result = {
                    "student_id": student_id,
                    "name": name,
                    "course_section": course_section,
                    "time": time_str,
                    "confidence": f"{round(confidence * 100, 1)}%",
                    "status": status
                }
                
                # Draw green box
                cv2.rectangle(img, (x, y), (x+w, y+h), (76, 175, 80), 2)
                # Label box background
                cv2.rectangle(img, (x, y - 25), (x+w, y), (76, 175, 80), -1)
                cv2.putText(img, f"{name} ({round(confidence * 100, 1)}%)", (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Draw red box for Unknown
                cv2.rectangle(img, (x, y), (x+w, y+h), (244, 67, 54), 2)
                cv2.rectangle(img, (x, y - 25), (x+w, y), (244, 67, 54), -1)
                cv2.putText(img, "Unknown", (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
        annotated_b64 = face_rec.cv2_to_base64(img)
        return {
            "face_detected": face_detected,
            "recognition": recognition_result,
            "annotated_frame": annotated_b64
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"face_detected": False, "annotated_frame": frame_b64}

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
    success = load_and_train_recognizer()
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
