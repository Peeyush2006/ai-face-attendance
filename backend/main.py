import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

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

# Auto-seed bundled models if missing in data/models (e.g. fresh container / empty volume mount)
bundled_models_dir = os.path.join(backend_dir, 'models')
if os.path.exists(bundled_models_dir):
    import shutil
    for mfile in os.listdir(bundled_models_dir):
        target = os.path.join(MODELS_DIR, mfile)
        if not os.path.exists(target):
            try:
                shutil.copyfile(os.path.join(bundled_models_dir, mfile), target)
                print(f"[main] Seeded missing model: {mfile}")
            except Exception as e:
                print(f"[main] Error seeding model {mfile}: {e}")

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

class TemporalAttendanceTracker:
    """
    Stabilizes face recognition over consecutive frames.
    Requires an identity to be recognized in at least required_consecutive frames
    before marking attendance. Prevents single-frame glitches from creating attendance.
    Also caches attendance marked in the current session/day to avoid repeated DB writes.
    """
    def __init__(self, required_consecutive=2, timeout_seconds=2.5):
        self.required_consecutive = required_consecutive
        self.timeout_seconds = timeout_seconds
        self.candidate_streaks = {}  # student_id -> {"count": int, "last_seen": float}
        self.marked_today = set()     # (student_id, date_str)
        
    def update(self, recognized_student_ids, today_str):
        import time
        now = time.time()
        ready_to_mark = []
        
        # Age out old streaks
        stale_keys = [sid for sid, data in self.candidate_streaks.items() if now - data["last_seen"] > self.timeout_seconds]
        for sid in stale_keys:
            del self.candidate_streaks[sid]
            
        # Update current recognized
        for sid in recognized_student_ids:
            if sid not in self.candidate_streaks:
                self.candidate_streaks[sid] = {"count": 1, "last_seen": now}
            else:
                self.candidate_streaks[sid]["count"] += 1
                self.candidate_streaks[sid]["last_seen"] = now
                
            streak = self.candidate_streaks[sid]["count"]
            already_marked = (sid, today_str) in self.marked_today
            
            if streak >= self.required_consecutive and not already_marked:
                ready_to_mark.append(sid)
                self.marked_today.add((sid, today_str))
                
        # Reset streaks for students not present in this frame
        current_set = set(recognized_student_ids)
        for sid in list(self.candidate_streaks.keys()):
            if sid not in current_set and (sid, today_str) not in self.marked_today:
                self.candidate_streaks[sid]["count"] = max(0, self.candidate_streaks[sid]["count"] - 1)
                
        return ready_to_mark

    def reset(self):
        self.candidate_streaks.clear()
        self.marked_today.clear()

temporal_tracker = TemporalAttendanceTracker(required_consecutive=2, timeout_seconds=2.5)

def load_and_train_recognizer(force: bool = False):
    global recognizer
    # Try loading pre-trained model first if not forced
    if not force and os.path.exists(MODEL_PATH):
        if recognizer.load(MODEL_PATH):
            print("Loaded pre-trained PCA model successfully.")
            return True
            
    # Train a new model from verified database photos
    print("Training PCA model from verified student database photos...")
    faces_list = []
    labels_list = []
    seen_paths = set()
    import cv2
    
    # Check database records
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
                
    if len(faces_list) > 0:
        if recognizer.train(faces_list, labels_list):
            recognizer.save(MODEL_PATH)
            print(f"Model trained successfully on {len(faces_list)} images for {len(set(labels_list))} students and saved.")
            return True
    print("Model training skipped or failed (insufficient registered images).")
    return False

# Initialize and load model on module load
load_and_train_recognizer()

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
    
    sid_clean = req.student_id.strip()
    name_clean = req.name.strip()
    if not sid_clean or not name_clean:
        raise HTTPException(status_code=400, detail="Student ID and Full Name are required.")
        
    safe_sid = sid_clean.replace('/', '_')
    student_dir = os.path.join(FACES_DIR, safe_sid)
    
    # 1. Strictly decode and validate each of the 5 photos for valid face presence
    processed_face_crops = []
    import cv2
    import numpy as np
    
    for idx, b64_img in enumerate(req.photos):
        try:
            img = face_rec.base64_to_cv2(b64_img)
            if img is None or img.size == 0:
                raise HTTPException(status_code=400, detail=f"Photo {idx+1} could not be decoded. Please provide valid image data.")
            
            # Detect faces with registration requirement (min 32x32 px)
            detected_boxes = face_rec.detect_faces(img, min_size=32, strict_quality=False)
            
            if len(detected_boxes) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No valid face detected in photo {idx+1}. Arbitrary objects or non-faces cannot be registered. Please face the camera directly with good lighting."
                )
            if len(detected_boxes) > 1:
                # Discard small background noise artifacts if one primary face is dominant
                largest_box = max(detected_boxes, key=lambda b: b[2] * b[3])
                max_area = largest_box[2] * largest_box[3]
                filtered_boxes = [b for b in detected_boxes if (b[2] * b[3]) >= 0.35 * max_area]
                if len(filtered_boxes) == 1:
                    detected_boxes = filtered_boxes
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Multiple faces ({len(detected_boxes)}) detected in photo {idx+1}. Exactly one face must be present during registration."
                    )
                
            bbox = detected_boxes[0]
            face_crop = face_rec.preprocess_face(img, bbox)
            
            # Verify crop quality
            is_valid, reason = face_rec.validate_face_quality(face_crop, min_size=32, min_blur=6.0)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Photo {idx+1} failed quality check ({reason}). Please capture a clear, non-blurred face.")
                
            processed_face_crops.append(face_crop)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error analyzing registration photo {idx+1}: {str(e)}")
            
    # All 5 photos passed strict validation!
    # Clean previous student photos on disk and database to prevent stale / duplicate data
    os.makedirs(student_dir, exist_ok=True)
    for fname in os.listdir(student_dir):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            try:
                os.remove(os.path.join(student_dir, fname))
            except Exception:
                pass
                
    database.delete_student_photos(sid_clean)
    
    # Save student in database
    success = database.add_student(sid_clean, name_clean, req.course_section)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save student record to database.")
        
    for idx, face_crop in enumerate(processed_face_crops):
        photo_path = os.path.join(student_dir, f"face_{idx+1}.png")
        cv2.imwrite(photo_path, face_crop)
        database.add_student_photo(sid_clean, photo_path)
        
    # Retrain model cleanly with new student photos
    load_and_train_recognizer(force=True)
    return {"message": f"Student '{name_clean}' registered successfully. 5 verified face encodings saved."}

@app.post("/api/process_frame")
def process_frame(payload: dict = Body(...)):
    frame_b64 = payload.get("frame")
    if not frame_b64:
        raise HTTPException(status_code=400, detail="No frame data provided.")
        
    import cv2
    try:
        img = face_rec.base64_to_cv2(frame_b64)
        if img is None:
            return {"face_detected": False, "faces_count": 0, "recognitions": [], "recognition": None, "annotated_frame": frame_b64}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_rec.detect_faces(img, min_size=32, strict_quality=False)
        
        # System settings
        settings = database.get_settings()
        threshold_setting = float(settings.get('threshold', 0.50))
        late_threshold_mins = int(settings.get('late_threshold', 15))
        
        tz = ZoneInfo("Asia/Kolkata")
        now = datetime.datetime.now(tz)
        today_str = database.get_ist_today().isoformat()
        class_start = now.replace(hour=10, minute=15, second=0, microsecond=0)
        late_cutoff = class_start + datetime.timedelta(minutes=late_threshold_mins)
        status = 'Late' if now > late_cutoff else 'Present'
        time_str = now.strftime('%H:%M')
        
        if len(detected_faces) == 0:
            # No face detected in frame
            annotated_b64 = face_rec.cv2_to_base64(img)
            return {
                "face_detected": False,
                "faces_count": 0,
                "recognitions": [],
                "recognition": None,
                "annotated_frame": annotated_b64
            }
            
        # 1. Independent recognition prediction for EVERY detected face
        face_candidates = []
        for bbox in detected_faces:
            x, y, w, h = bbox
            face_crop = face_rec.preprocess_face(gray, bbox)
            
            # Predict canonical and mirrored face
            sid1, dist1, conf1 = recognizer.predict(face_crop, threshold=threshold_setting)
            face_crop_flip = cv2.flip(face_crop, 1)
            sid2, dist2, conf2 = recognizer.predict(face_crop_flip, threshold=threshold_setting)
            
            if conf2 > conf1:
                student_id, distance, confidence = sid2, dist2, conf2
            else:
                student_id, distance, confidence = sid1, dist1, conf1
                
            face_candidates.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "student_id": student_id,
                "distance": distance,
                "confidence": confidence
            })
            
        # 2. Enforce 1-to-1 matching constraint:
        # The same student cannot be two physically distinct faces in the same video frame.
        # Assign student identity to the face with the highest confidence.
        assigned_students = set()
        sorted_indices = sorted(range(len(face_candidates)), key=lambda i: face_candidates[i]["confidence"], reverse=True)
        final_assignments = {}
        
        for idx in sorted_indices:
            cand = face_candidates[idx]
            sid = cand["student_id"]
            conf = cand["confidence"]
            
            if sid and conf >= threshold_setting and sid not in assigned_students:
                assigned_students.add(sid)
                final_assignments[idx] = (sid, conf)
            else:
                final_assignments[idx] = (None, conf)
                
        # 3. Build independent face recognition objects and annotate frame
        conn = database.get_db_connection()
        recognitions = []
        conf_by_sid = {}
        
        for idx, cand in enumerate(face_candidates):
            x, y, w, h = cand["bbox"]
            matched_sid, conf = final_assignments[idx]
            
            if matched_sid:
                row = conn.execute('SELECT name, course_section FROM students WHERE student_id = ?', (matched_sid,)).fetchone()
                name = row['name'] if row else matched_sid
                course_section = row['course_section'] if row else ""
                conf_by_sid[matched_sid] = conf
                
                rec_info = {
                    "box": [x, y, w, h],
                    "face_box": [x, y, w, h],
                    "student_id": matched_sid,
                    "student_name": name,
                    "name": name,
                    "course_section": course_section,
                    "confidence": f"{round(conf * 100, 1)}%",
                    "confidence_value": round(float(conf), 3),
                    "recognized": True,
                    "status": status,
                    "time": time_str
                }
                recognitions.append(rec_info)
                
                # Draw green bounding box & 2-line label banner
                cv2.rectangle(img, (x, y), (x+w, y+h), (76, 175, 80), 2)
                banner_h = 36
                banner_y = max(0, y - banner_h)
                cv2.rectangle(img, (x, banner_y), (x+w, y), (76, 175, 80), -1)
                cv2.putText(img, f"Name: {name}", (x + 4, banner_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Status: Recognized ({round(conf * 100, 1)}%)", (x + 4, banner_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Unknown / rejected face
                rec_info = {
                    "box": [x, y, w, h],
                    "face_box": [x, y, w, h],
                    "student_id": None,
                    "student_name": "Unknown",
                    "name": "Unknown",
                    "course_section": "",
                    "confidence": f"{round(conf * 100, 1)}%",
                    "confidence_value": round(float(conf), 3),
                    "recognized": False,
                    "status": "Not recognized",
                    "time": ""
                }
                recognitions.append(rec_info)
                
                # Draw red bounding box & 2-line label banner
                cv2.rectangle(img, (x, y), (x+w, y+h), (244, 67, 54), 2)
                banner_h = 36
                banner_y = max(0, y - banner_h)
                cv2.rectangle(img, (x, banner_y), (x+w, y), (244, 67, 54), -1)
                cv2.putText(img, "Name: Unknown", (x + 4, banner_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "Status: Not recognized", (x + 4, banner_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                
        conn.close()
        
        # 4. Temporal Stabilization & Attendance Logging:
        # Only mark attendance when a student is recognized consistently across consecutive frames
        recognized_sids = [r["student_id"] for r in recognitions if r["recognized"] and r["student_id"]]
        ready_to_mark = temporal_tracker.update(recognized_sids, today_str)
        
        for sid in ready_to_mark:
            c_val = conf_by_sid.get(sid, 0.90)
            database.mark_attendance(sid, status, time_str, c_val * 100)
            print(f"[Attendance] Successfully recorded attendance for {sid} (Confidence: {round(c_val*100, 1)}%)")
            
        annotated_b64 = face_rec.cv2_to_base64(img)
        return {
            "face_detected": len(recognitions) > 0,
            "faces_count": len(recognitions),
            "recognitions": recognitions,
            "recognition": next((r for r in recognitions if r["recognized"]), (recognitions[0] if recognitions else None)),
            "annotated_frame": annotated_b64
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"face_detected": False, "faces_count": 0, "recognitions": [], "recognition": None, "annotated_frame": frame_b64}

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
