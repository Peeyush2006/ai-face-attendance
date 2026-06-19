import sys
import os
import numpy as np

# Ensure backend folder is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import face_rec

def test_pca_face_recognition():
    print("Starting EigenfaceRecognizer Unit Test...")
    
    # 1. Initialize Recognizer
    recognizer = face_rec.EigenfaceRecognizer(num_components=10, img_size=(128, 128))
    
    # 2. Generate Mock Face Data
    # 4 students, 5 images each = 20 images
    num_students = 4
    images_per_student = 5
    img_size = (128, 128)
    D = img_size[0] * img_size[1] # 16384
    
    faces = []
    labels = []
    
    # We generate a base "template" pattern for each student and add small random noise
    # so that the 5 images for a student are clustered tightly in the D-dimensional space,
    # and students are clearly distinct from one another.
    np.random.seed(42) # Deterministic
    
    for s_idx in range(num_students):
        student_id = f"STUDENT_{s_idx}"
        # Unique base pattern (e.g. step patterns or sinusoids)
        base_pattern = np.zeros(img_size)
        base_pattern[s_idx*20:(s_idx+1)*20, :] = 150 # Distinct horizontal bands
        base_pattern[:, s_idx*20:(s_idx+1)*20] += 80 # Distinct vertical bands
        
        for i_idx in range(images_per_student):
            # Add noise to base pattern
            noise = np.random.normal(0, 15, img_size)
            face_img = np.clip(base_pattern + noise, 0, 255).astype(np.uint8)
            faces.append(face_img)
            labels.append(student_id)
            
    # 3. Train
    print(f"Training on {len(faces)} images of size {img_size}...")
    success = recognizer.train(faces, labels)
    assert success, "Training failed!"
    print("Training completed successfully.")
    
    # 4. Save and Load Test
    model_file = "test_model_pca.npz"
    save_success = recognizer.save(model_file)
    assert save_success, "Saving model failed!"
    print(f"Model saved successfully to {model_file}.")
    
    new_recognizer = face_rec.EigenfaceRecognizer(num_components=10, img_size=(128, 128))
    load_success = new_recognizer.load(model_file)
    assert load_success, "Loading model failed!"
    print("Model loaded successfully into a new recognizer.")
    
    # Clean up test file
    if os.path.exists(model_file):
        os.remove(model_file)
        
    # 5. Predict / Verify
    print("Verifying prediction accuracy...")
    correct_predictions = 0
    total_predictions = len(faces)
    
    for idx, test_face in enumerate(faces):
        expected_label = labels[idx]
        # Predict using loaded recognizer
        predicted_label, distance = new_recognizer.predict(test_face)
        
        # Calculate confidence
        confidence = max(0.0, 1.0 - distance / 5000.0)
        
        print(f"Image {idx}: Expected={expected_label}, Predicted={predicted_label}, Dist={round(distance, 1)}, Conf={round(confidence*100, 1)}%")
        
        if predicted_label == expected_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {correct_predictions}/{total_predictions} ({accuracy*100}%)")
    
    assert accuracy == 1.0, f"Expected 100% accuracy, but got {accuracy*100}%"
    print("EigenfaceRecognizer Unit Test PASSED!")

if __name__ == "__main__":
    test_pca_face_recognition()
