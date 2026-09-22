import sys
import os
import numpy as np

# Ensure backend folder is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import face_rec

def test_pca_face_recognition():
    print("==================================================")
    print("Running EigenfaceRecognizer Verification Tests")
    print("==================================================")
    
    img_size = (128, 128)
    
    # ----------------------------------------------------
    # TEST 1: Multi-Student Training and Recognition Accuracy
    # ----------------------------------------------------
    print("\n--- TEST 1: Multi-Student Recognition ---")
    recognizer = face_rec.EigenfaceRecognizer(num_components=15, img_size=img_size)
    
    num_students = 4
    images_per_student = 5
    faces = []
    labels = []
    
    np.random.seed(42)
    
    for s_idx in range(num_students):
        student_id = f"STUDENT_{s_idx}"
        base_pattern = np.zeros(img_size)
        base_pattern[s_idx*20:(s_idx+1)*20, :] = 150
        base_pattern[:, s_idx*20:(s_idx+1)*20] += 80
        
        for i_idx in range(images_per_student):
            noise = np.random.normal(0, 10, img_size)
            face_img = np.clip(base_pattern + noise, 0, 255).astype(np.uint8)
            faces.append(face_img)
            labels.append(student_id)
            
    success = recognizer.train(faces, labels)
    assert success, "Training failed!"
    print(f"[OK] Trained successfully on {len(faces)} images across {num_students} students.")
    
    # Verify predictions on registered students
    correct = 0
    for idx, test_face in enumerate(faces):
        expected_label = labels[idx]
        predicted_label, dist, conf = recognizer.predict(test_face, threshold=0.45)
        if predicted_label == expected_label:
            correct += 1
            
    accuracy = correct / len(faces)
    print(f"[OK] Recognition Accuracy on registered users: {correct}/{len(faces)} ({accuracy*100:.1f}%)")
    assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy*100}%"
    
    # ----------------------------------------------------
    # TEST 2: Save and Load Model Persistence
    # ----------------------------------------------------
    print("\n--- TEST 2: Model Persistence (Save / Load) ---")
    model_file = "test_model_pca.npz"
    assert recognizer.save(model_file), "Saving model failed!"
    
    loaded_rec = face_rec.EigenfaceRecognizer(num_components=15, img_size=img_size)
    assert loaded_rec.load(model_file), "Loading model failed!"
    print("[OK] Model saved and reloaded successfully.")
    if os.path.exists(model_file):
        os.remove(model_file)
        
    # ----------------------------------------------------
    # TEST 3: Stranger / Unregistered Face Rejection Test
    # (Testing the bug: 1 user registered, stranger in front of camera)
    # ----------------------------------------------------
    print("\n--- TEST 3: Stranger Rejection Test ---")
    single_user_rec = face_rec.EigenfaceRecognizer(num_components=5, img_size=img_size)
    
    # Register only 1 user ("Peeyush")
    peeyush_faces = []
    peeyush_labels = []
    peeyush_base = np.zeros(img_size)
    peeyush_base[20:50, 20:50] = 200
    peeyush_base[60:90, 60:90] = 180
    
    for i in range(5):
        noise = np.random.normal(0, 8, img_size)
        img = np.clip(peeyush_base + noise, 0, 255).astype(np.uint8)
        peeyush_faces.append(img)
        peeyush_labels.append("PEEYUSH_TIWARI")
        
    single_user_rec.train(peeyush_faces, peeyush_labels)
    print("[OK] Single user model trained for 'PEEYUSH_TIWARI' with 5 photos.")
    
    # 1) Peeyush should be recognized with high confidence
    peeyush_test = np.clip(peeyush_base + np.random.normal(0, 8, img_size), 0, 255).astype(np.uint8)
    p_label, p_dist, p_conf = single_user_rec.predict(peeyush_test, threshold=0.50)
    print(f"Registered user test -> Label: {p_label}, Distance: {p_dist:.1f}, Confidence: {p_conf*100:.1f}%")
    assert p_label == "PEEYUSH_TIWARI", f"Expected PEEYUSH_TIWARI, got {p_label}"
    assert p_conf >= 0.50, f"Expected high confidence, got {p_conf}"
    print("[OK] Registered user correctly recognized.")
    
    # 2) A stranger (completely different face pattern) MUST BE REJECTED as Unknown (None)
    stranger_base = np.zeros(img_size)
    stranger_base[0:30, 80:120] = 170
    stranger_base[80:110, 10:40] = 220
    stranger_test = np.clip(stranger_base + np.random.normal(0, 10, img_size), 0, 255).astype(np.uint8)
    
    s_label, s_dist, s_conf = single_user_rec.predict(stranger_test, threshold=0.50)
    print(f"Stranger test -> Label: {s_label}, Distance: {s_dist:.1f}, Confidence: {s_conf*100:.1f}%")
    assert s_label is None, f"Stranger was falsely recognized as '{s_label}'! Rejection failed."
    print("[OK] Stranger face correctly rejected as Unknown (None).")
    
    # 3) Random background / noise MUST BE REJECTED
    noise_test = np.random.randint(0, 256, img_size, dtype=np.uint8)
    n_label, n_dist, n_conf = single_user_rec.predict(noise_test, threshold=0.50)
    print(f"Noise image test -> Label: {n_label}, Distance: {n_dist:.1f}, Confidence: {n_conf*100:.1f}%")
    assert n_label is None, f"Noise image was falsely recognized as '{n_label}'!"
    print("[OK] Random noise correctly rejected as Unknown (None).")
    
    print("\n==================================================")
    print("ALL TESTS PASSED! Fix is verified.")
    print("==================================================")

if __name__ == "__main__":
    test_pca_face_recognition()
