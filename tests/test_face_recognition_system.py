import os
import sys
import unittest
import numpy as np
import cv2
import datetime
from zoneinfo import ZoneInfo

# Ensure backend directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

import face_rec
import database
import main

class TestFaceRecognitionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()
        main.load_and_train_recognizer(force=False)
        cls.student_a_id = '2401151028' # Peeyush Kumar Tiwari
        cls.student_b_id = '2401151012' # Rishu Kumar
        cls.student_c_id = '2401151030' # Ahsas Singh
        
        # Load sample face crops for students A and B
        cls.face_a = cv2.imread(os.path.join(BASE_DIR, 'data', 'faces', cls.student_a_id, 'face_1.png'))
        cls.face_b = cv2.imread(os.path.join(BASE_DIR, 'data', 'faces', cls.student_b_id, 'face_1.png'))
        
        # Define face C canvas for registered student C (Ahsas Singh)
        face_c = np.ones((128, 128, 3), dtype=np.uint8) * 180
        cv2.ellipse(face_c, (64, 64), (45, 55), 0, 0, 360, (140, 160, 210), -1)
        cv2.ellipse(face_c, (64, 30), (46, 25), 0, 0, 360, (30, 30, 30), -1)
        cv2.circle(face_c, (46, 52), 6, (255, 255, 255), -1)
        cv2.circle(face_c, (82, 52), 6, (255, 255, 255), -1)
        cv2.circle(face_c, (46, 52), 3, (20, 20, 20), -1)
        cv2.circle(face_c, (82, 52), 3, (20, 20, 20), -1)
        cv2.line(face_c, (38, 44), (54, 44), (30, 30, 30), 2)
        cv2.line(face_c, (74, 44), (90, 44), (30, 30, 30), 2)
        cv2.line(face_c, (64, 55), (64, 75), (100, 120, 160), 2)
        cv2.line(face_c, (60, 75), (68, 75), (100, 120, 160), 2)
        cv2.ellipse(face_c, (64, 92), (18, 8), 0, 0, 180, (60, 70, 140), -1)
        cls.face_c = face_c
        
        assert cls.face_a is not None, f"Face A not found for {cls.student_a_id}"
        assert cls.face_b is not None, f"Face B not found for {cls.student_b_id}"
        assert cls.face_c is not None, f"Face C not found for {cls.student_c_id}"

    def setUp(self):
        # Reset temporal tracker before each test
        main.temporal_tracker.reset()

    # -------------------------------------------------------------
    # TEST 1: No person in front of camera
    # -------------------------------------------------------------
    def test_01_no_person_empty_frame(self):
        print("\n--- Running TEST 1: No person in front of camera ---")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        b64_frame = face_rec.cv2_to_base64(empty_frame)
        
        res = main.process_frame({"frame": b64_frame})
        self.assertFalse(res["face_detected"])
        self.assertEqual(len(res["recognitions"]), 0)
        self.assertIsNone(res["recognition"])
        print("[PASS] TEST 1: Empty frame correctly outputs no face detected and no attendance.")

    # -------------------------------------------------------------
    # TEST 2: Random object in front of camera
    # -------------------------------------------------------------
    def test_02_random_object_in_camera(self):
        print("\n--- Running TEST 2: Random object in front of camera ---")
        # 1. Random noise frame
        noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        b64_noise = face_rec.cv2_to_base64(noise_frame)
        res_noise = main.process_frame({"frame": b64_noise})
        
        # Must not recognize any student
        recognized_noise = [r for r in res_noise["recognitions"] if r["recognized"]]
        self.assertEqual(len(recognized_noise), 0)
        
        # 2. Frame with drawn non-face object (cup/book)
        obj_frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
        cv2.circle(obj_frame, (320, 240), 90, (40, 40, 40), -1)
        cv2.rectangle(obj_frame, (280, 310), (360, 380), (70, 70, 70), -1)
        b64_obj = face_rec.cv2_to_base64(obj_frame)
        res_obj = main.process_frame({"frame": b64_obj})
        
        recognized_obj = [r for r in res_obj["recognitions"] if r["recognized"]]
        self.assertEqual(len(recognized_obj), 0)
        print("[PASS] TEST 2: Random objects rejected; no student recognized.")

    # -------------------------------------------------------------
    # TEST 3: Unknown person's face
    # -------------------------------------------------------------
    def test_03_unknown_person_face(self):
        print("\n--- Running TEST 3: Unknown person's face ---")
        # Create an unregistered human face pattern / stranger model
        # Generate face on synthetic eigenspace holdout
        stranger_crop = np.zeros((128, 128), dtype=np.uint8)
        # Add head contour, eyes, nose, mouth
        cv2.ellipse(stranger_crop, (64, 64), (45, 55), 0, 0, 360, 160, -1)
        cv2.circle(stranger_crop, (46, 50), 6, 40, -1)
        cv2.circle(stranger_crop, (82, 50), 6, 40, -1)
        cv2.ellipse(stranger_crop, (64, 88), (20, 8), 0, 0, 180, 30, -1)
        
        stranger_crop = cv2.equalizeHist(stranger_crop)
        sid, dist, conf = main.recognizer.predict(stranger_crop, threshold=0.50)
        
        self.assertIsNone(sid, f"Stranger face was falsely recognized as '{sid}'")
        self.assertLess(conf, 0.50)
        print(f"[PASS] TEST 3: Unknown stranger face correctly rejected (label={sid}, conf={conf*100:.1f}%).")

    # -------------------------------------------------------------
    # TEST 4: Registered Student A alone
    # -------------------------------------------------------------
    def test_04_registered_student_a_alone(self):
        print("\n--- Running TEST 4: Registered Student A alone ---")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Place Student A in center
        h, w = self.face_a.shape[:2]
        frame[150:150+h, 240:240+w] = self.face_a
        
        b64 = face_rec.cv2_to_base64(frame)
        res = main.process_frame({"frame": b64})
        
        self.assertTrue(res["face_detected"])
        recognized = [r for r in res["recognitions"] if r["recognized"]]
        self.assertEqual(len(recognized), 1)
        self.assertEqual(recognized[0]["student_id"], self.student_a_id)
        print(f"[PASS] TEST 4: Student A correctly recognized alone ({recognized[0]['name']}, conf={recognized[0]['confidence']}).")

    # -------------------------------------------------------------
    # TEST 5: Registered Student B alone
    # -------------------------------------------------------------
    def test_05_registered_student_b_alone(self):
        print("\n--- Running TEST 5: Registered Student B alone ---")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = self.face_b.shape[:2]
        frame[150:150+h, 240:240+w] = self.face_b
        
        b64 = face_rec.cv2_to_base64(frame)
        res = main.process_frame({"frame": b64})
        
        self.assertTrue(res["face_detected"])
        recognized = [r for r in res["recognitions"] if r["recognized"]]
        self.assertEqual(len(recognized), 1)
        self.assertEqual(recognized[0]["student_id"], self.student_b_id)
        print(f"[PASS] TEST 5: Student B correctly recognized alone ({recognized[0]['name']}, conf={recognized[0]['confidence']}).")

    # -------------------------------------------------------------
    # TEST 6: Student A + Student B together in frame
    # -------------------------------------------------------------
    def test_06_student_a_and_b_together(self):
        print("\n--- Running TEST 6: Student A + Student B together ---")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ha, wa = self.face_a.shape[:2]
        hb, wb = self.face_b.shape[:2]
        
        # Place Student A on left side, Student B on right side
        frame[150:150+ha, 80:80+wa] = self.face_a
        frame[150:150+hb, 400:400+wb] = self.face_b
        
        b64 = face_rec.cv2_to_base64(frame)
        res = main.process_frame({"frame": b64})
        
        self.assertTrue(res["face_detected"])
        self.assertEqual(res["faces_count"], 2)
        
        recognized_sids = {r["student_id"] for r in res["recognitions"] if r["recognized"]}
        self.assertIn(self.student_a_id, recognized_sids)
        self.assertIn(self.student_b_id, recognized_sids)
        print(f"[PASS] TEST 6: Both Student A & B independently detected and recognized ({recognized_sids}).")

    # -------------------------------------------------------------
    # TEST 7: Student A + Unknown person
    # -------------------------------------------------------------
    def test_07_student_a_and_unknown_person(self):
        print("\n--- Running TEST 7: Student A + Unknown person ---")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ha, wa = self.face_a.shape[:2]
        frame[150:150+ha, 80:80+wa] = self.face_a
        
        # Add stranger face on right
        stranger_crop = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.ellipse(stranger_crop, (64, 64), (45, 55), 0, 0, 360, (150, 150, 150), -1)
        cv2.circle(stranger_crop, (46, 50), 6, (30, 30, 30), -1)
        cv2.circle(stranger_crop, (82, 50), 6, (30, 30, 30), -1)
        frame[150:150+128, 400:400+128] = stranger_crop
        
        b64 = face_rec.cv2_to_base64(frame)
        res = main.process_frame({"frame": b64})
        
        recognized = [r for r in res["recognitions"] if r["recognized"]]
        unknowns = [r for r in res["recognitions"] if not r["recognized"]]
        
        # Exactly Student A recognized, and stranger is either unknown or rejected
        self.assertTrue(any(r["student_id"] == self.student_a_id for r in recognized))
        for u in unknowns:
            self.assertIsNone(u["student_id"])
            self.assertEqual(u["student_name"], "Unknown")
        print("[PASS] TEST 7: Student A recognized while stranger remains Unknown.")

    # -------------------------------------------------------------
    # TEST 8: Three registered students together
    # -------------------------------------------------------------
    def test_08_three_registered_students_together(self):
        print("\n--- Running TEST 8: Three registered students together ---")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Place 3 registered faces at native 128x128 resolution across the frame
        frame[160:160+128, 40:40+128] = self.face_a
        frame[160:160+128, 240:240+128] = self.face_b
        frame[160:160+128, 440:440+128] = self.face_c
        
        b64 = face_rec.cv2_to_base64(frame)
        res = main.process_frame({"frame": b64})
        
        self.assertEqual(res["faces_count"], 3)
        recognized_sids = {r["student_id"] for r in res["recognitions"] if r["recognized"]}
        self.assertIn(self.student_a_id, recognized_sids)
        self.assertIn(self.student_b_id, recognized_sids)
        self.assertIn(self.student_c_id, recognized_sids)
        print(f"[PASS] TEST 8: All 3 registered students recognized simultaneously ({recognized_sids}).")

    # -------------------------------------------------------------
    # TEST 9: Photo/image of a registered student shown to camera
    # -------------------------------------------------------------
    def test_09_photo_attack_documentation(self):
        print("\n--- Running TEST 9: Photo attack assessment & documentation ---")
        # Simulating a 2D photograph attack of Student A
        photo_attack = self.face_a.copy()
        sid, dist, conf = main.recognizer.predict(cv2.cvtColor(photo_attack, cv2.COLOR_BGR2GRAY), threshold=0.50)
        
        # Standard 2D face recognition alone matches face textures
        print(f"  [ASSESSMENT] 2D Photo match result: sid={sid}, conf={conf*100:.1f}%")
        print("  [DOCUMENTATION] Normal 2D Eigenface/PCA recognition processes pixel textures;")
        print("  it inherently cannot distinguish a live 3D face from a static 2D printout or screen.")
        print("  Our YuNet landmark extractor tracks facial coordinates across video frames.")
        print("  Full production protection against photo/screen spoofing requires 3D depth sensors,")
        print("  IR cameras, or active challenge-response (eye blink / head rotation verification).")
        self.assertEqual(sid, self.student_a_id)
        print("[PASS] TEST 9: Photo presentation test assessed and limitations documented.")

    # -------------------------------------------------------------
    # TEST 10: Student A leaves the frame
    # -------------------------------------------------------------
    def test_10_student_leaves_frame(self):
        print("\n--- Running TEST 10: Student leaves frame ---")
        # Frame 1: Student A present
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        ha, wa = self.face_a.shape[:2]
        frame1[150:150+ha, 240:240+wa] = self.face_a
        res1 = main.process_frame({"frame": face_rec.cv2_to_base64(frame1)})
        self.assertTrue(res1["face_detected"])
        
        # Frame 2: Student A leaves -> empty room
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        res2 = main.process_frame({"frame": face_rec.cv2_to_base64(frame2)})
        self.assertFalse(res2["face_detected"])
        self.assertEqual(len(res2["recognitions"]), 0)
        self.assertIsNone(res2["recognition"])
        
        # Frame 3: Random object placed in scene
        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame3, (200, 200), (350, 350), (120, 120, 120), -1)
        res3 = main.process_frame({"frame": face_rec.cv2_to_base64(frame3)})
        recognized3 = [r for r in res3["recognitions"] if r["recognized"]]
        self.assertEqual(len(recognized3), 0)
        print("[PASS] TEST 10: Leaving frame immediately clears identity; no lingering or fallback on objects.")

    # -------------------------------------------------------------
    # TEST 11: Two students cross/overlap temporarily
    # -------------------------------------------------------------
    def test_11_two_students_crossing(self):
        print("\n--- Running TEST 11: Two students crossing/overlapping ---")
        # Frame 1: A on left (x=100), B on right (x=400)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[150:150+128, 100:100+128] = self.face_a
        frame1[150:150+128, 400:400+128] = self.face_b
        res1 = main.process_frame({"frame": face_rec.cv2_to_base64(frame1)})
        
        sids1 = {r["student_id"] for r in res1["recognitions"] if r["recognized"]}
        self.assertIn(self.student_a_id, sids1)
        self.assertIn(self.student_b_id, sids1)
        
        # Frame 2: Swapped positions (A on right x=400, B on left x=100)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[150:150+128, 100:100+128] = self.face_b
        frame2[150:150+128, 400:400+128] = self.face_a
        res2 = main.process_frame({"frame": face_rec.cv2_to_base64(frame2)})
        
        # Verify both are still recognized correctly at their new positions without identity mixup
        for r in res2["recognitions"]:
            if r["box"][0] < 250:
                self.assertEqual(r["student_id"], self.student_b_id)
            elif r["box"][0] >= 250:
                self.assertEqual(r["student_id"], self.student_a_id)
                
        print("[PASS] TEST 11: Crossed students maintain distinct independent identities without mixup.")

    # -------------------------------------------------------------
    # TEST 12: Attendance recorded once per session/day
    # -------------------------------------------------------------
    def test_12_attendance_recorded_once(self):
        print("\n--- Running TEST 12: Attendance uniqueness & temporal stabilization ---")
        today_str = database.get_ist_today().isoformat()
        conn = database.get_db_connection()
        conn.execute('DELETE FROM attendance WHERE student_id = ? AND date = ?', (self.student_a_id, today_str))
        conn.commit()
        conn.close()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[150:150+128, 240:240+128] = self.face_a
        b64 = face_rec.cv2_to_base64(frame)
        
        # Frame 1: First observation (streak = 1, required = 2) -> Not marked yet
        main.process_frame({"frame": b64})
        conn = database.get_db_connection()
        row1 = conn.execute('SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?', (self.student_a_id, today_str)).fetchone()
        self.assertEqual(row1[0], 0, "Single glitch frame should not immediately mark attendance")
        
        # Frame 2: Second consecutive observation (streak = 2) -> Confirmed & marked!
        main.process_frame({"frame": b64})
        row2 = conn.execute('SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?', (self.student_a_id, today_str)).fetchone()
        self.assertEqual(row2[0], 1, "Stabilized recognition must record attendance")
        
        # Frame 3..10: Continuing frames must not duplicate attendance
        for _ in range(8):
            main.process_frame({"frame": b64})
            
        row_final = conn.execute('SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?', (self.student_a_id, today_str)).fetchone()
        conn.close()
        self.assertEqual(row_final[0], 1, "Attendance must remain exactly 1 record for today")
        print("[PASS] TEST 12: Temporal stabilization required 2 frames; attendance recorded exactly once.")

if __name__ == '__main__':
    unittest.main()
