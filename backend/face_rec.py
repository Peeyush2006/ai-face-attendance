import os
import cv2
import numpy as np
import base64

class EigenfaceRecognizer:
    def __init__(self, num_components=50, img_size=(128, 128)):
        self.num_components = num_components
        self.img_size = img_size
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.labels = []
        self.class_centroids = {}
        self.class_radii = {}
        self.class_mean_faces = {}
        self.class_img_radii = {}
        self.max_recon_error = 22.0

    def train(self, faces_list, labels_list):
        """
        faces_list: list of grayscale 2D numpy arrays of shape self.img_size
        labels_list: list of labels (student_ids) corresponding to each face
        """
        if not faces_list or len(faces_list) == 0:
            return False

        # Filter out invalid, blank, or corrupted images
        valid_faces = []
        valid_labels = []
        for f, l in zip(faces_list, labels_list):
            if f is None or f.size == 0:
                continue
            if f.shape != self.img_size:
                try:
                    f = cv2.resize(f, self.img_size, interpolation=cv2.INTER_AREA)
                except Exception:
                    continue
            # Check standard deviation: blank / pitch-black / uniform images will have std < 5.0
            if float(np.std(f)) < 5.0 or float(np.mean(f)) < 5.0:
                print(f"[face_rec] Skipping corrupted / blank face for label: {l}")
                continue
            valid_faces.append(f)
            valid_labels.append(l)

        if len(valid_faces) == 0:
            print("[face_rec] No valid face images found to train.")
            return False

        # Flatten images to 1D vectors
        X = np.array([f.flatten() for f in valid_faces], dtype=np.float32)
        self.labels = list(valid_labels)
        
        N, D = X.shape
        
        # Calculate mean face across entire dataset
        self.mean_face = np.mean(X, axis=0)
        
        # Center the training images
        A = X - self.mean_face
        
        # PCA computation using eigenvalues/eigenvectors of L = A @ A.T (N x N)
        # instead of the full covariance matrix C = A.T @ A (D x D)
        L = A @ A.T
        
        # Calculate eigenvalues and eigenvectors of L
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort in descending order of eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Map eigenvectors back to original dimensions: U = A.T @ eigenvectors
        U = A.T @ eigenvectors
        
        # Normalize eigenvectors to have unit length
        norms = np.linalg.norm(U, axis=0)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        U = U / norms
        
        # Select top k components
        k = min(self.num_components, N - 1)
        if k <= 0:
            k = 1
        self.eigenfaces = U[:, :k]
        
        # Project training images onto the face space
        self.projections = A @ self.eigenfaces
        
        # Calculate training reconstruction errors to establish face manifold boundary
        recon_A = self.projections @ self.eigenfaces.T
        train_recon_errors = np.linalg.norm(A - recon_A, axis=1) / np.sqrt(D)
        max_train_recon = float(np.max(train_recon_errors)) if len(train_recon_errors) > 0 else 10.0
        # Allow reasonable variance for test lighting/angles while rejecting out-of-distribution strangers
        self.max_recon_error = min(max(max_train_recon * 2.0, 20.0), 38.0)
        
        # Calculate per-class templates, centroids, and intra-class radii
        self.class_centroids = {}
        self.class_radii = {}
        self.class_mean_faces = {}
        self.class_img_radii = {}
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            indices = [i for i, l in enumerate(self.labels) if l == label]
            class_X = X[indices]
            class_proj = self.projections[indices]
            
            # PCA centroid and radius
            centroid = np.mean(class_proj, axis=0)
            self.class_centroids[label] = centroid
            dists = np.linalg.norm(class_proj - centroid, axis=1)
            raw_radius = float(np.max(dists)) if len(dists) > 0 else 500.0
            # Bound intra-class radius to prevent manifold explosion
            self.class_radii[label] = max(min(raw_radius * 2.4, 3000.0), 1200.0)
            
            # Image space template and pixel variance radius
            mean_img = np.mean(class_X, axis=0)
            self.class_mean_faces[label] = mean_img
            img_dists = np.linalg.norm(class_X - mean_img, axis=1) / np.sqrt(D)
            raw_img_radius = float(np.max(img_dists)) if len(img_dists) > 0 else 20.0
            self.class_img_radii[label] = max(min(raw_img_radius * 2.5, 60.0), 20.0)
            
        return True

    def predict(self, face_img, threshold=0.50):
        """
        face_img: grayscale 2D numpy array of shape self.img_size
        threshold: minimum confidence score required to accept match
        returns: (matched_label, min_dist, confidence)
        """
        if self.eigenfaces is None or self.mean_face is None or self.projections is None:
            return None, float('inf'), 0.0
            
        q = face_img.flatten().astype(np.float32)
        D = q.shape[0]
        q_centered = q - self.mean_face
        
        # 1. Project onto the eigenfaces space
        q_proj = q_centered @ self.eigenfaces
        
        # 2. Compute Reconstruction Residual (Distance to Face Space)
        q_recon_centered = q_proj @ self.eigenfaces.T
        recon_error = float(np.linalg.norm(q_centered - q_recon_centered) / np.sqrt(D))
        
        # 3. Find closest training projection candidate
        distances = np.linalg.norm(self.projections - q_proj, axis=1)
        min_idx = int(np.argmin(distances))
        min_sample_dist = float(distances[min_idx])
        candidate_label = self.labels[min_idx]
        
        # Candidate's class centroid and allowable radius in PCA space
        candidate_centroid = self.class_centroids.get(candidate_label)
        if candidate_centroid is not None:
            centroid_dist = float(np.linalg.norm(q_proj - candidate_centroid))
        else:
            centroid_dist = min_sample_dist
            
        allowable_pca_dist = self.class_radii.get(candidate_label, 2500.0)
        
        # 4. Compare with candidate's image template & cosine similarity
        template = self.class_mean_faces.get(candidate_label, self.mean_face)
        img_dist = float(np.linalg.norm(q - template) / np.sqrt(D))
        
        q_norm = np.linalg.norm(q)
        t_norm = np.linalg.norm(template)
        if q_norm > 1e-6 and t_norm > 1e-6:
            cos_sim = float(np.dot(q, template) / (q_norm * t_norm))
        else:
            cos_sim = 0.0
            
        # Maximum allowed thresholds for the candidate class
        allowable_recon = max(self.max_recon_error, 35.0)
        allowable_img_dist = max(self.class_img_radii.get(candidate_label, 25.0) * 1.8, 48.0)
        
        # Confidence score components in [0, 1]
        # Primary factor: PCA projection distance to the enrolled class
        conf_pca = max(0.0, 1.0 - (min_sample_dist / allowable_pca_dist))
        conf_recon = max(0.0, 1.0 - (recon_error / allowable_recon))
        conf_img = max(0.0, 1.0 - (img_dist / allowable_img_dist))
        conf_cos = max(0.0, (cos_sim - 0.60) / 0.40) if cos_sim >= 0.60 else 0.0
        
        # Composite confidence: heavily weighted by PCA projection distance
        confidence = float(0.45 * conf_pca + 0.25 * conf_img + 0.20 * conf_recon + 0.10 * conf_cos)
        
        # Strict Rejection Criteria:
        # 1. PCA distance exceeds allowable radius -> not the enrolled candidate
        # 2. Centroid distance exceeds allowable radius -> not within class cluster
        # 3. High reconstruction error -> face doesn't match facial manifold
        # 4. High image distance -> visual appearance differs from enrolled student
        # 5. Low cosine similarity -> angle in image space is dissimilar
        # 6. Overall confidence is below required threshold
        is_stranger = (min_sample_dist > allowable_pca_dist or
                       centroid_dist > allowable_pca_dist * 1.30 or
                       recon_error > allowable_recon or 
                       (img_dist > allowable_img_dist and min_sample_dist > allowable_pca_dist * 0.5) or 
                       cos_sim < 0.65 or 
                       confidence < threshold)
                       
        if is_stranger:
            return None, min_sample_dist, confidence
            
        return candidate_label, min_sample_dist, confidence

    def save(self, filepath):
        if self.mean_face is None:
            return False
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        labels_arr = np.array(self.labels, dtype=object)
        unique_labels = np.array(list(self.class_centroids.keys()), dtype=object)
        centroids_arr = np.array([self.class_centroids[l] for l in unique_labels])
        radii_arr = np.array([self.class_radii[l] for l in unique_labels])
        templates_arr = np.array([self.class_mean_faces[l] for l in unique_labels])
        img_radii_arr = np.array([self.class_img_radii[l] for l in unique_labels])
        
        np.savez(filepath, 
                 mean_face=self.mean_face, 
                 eigenfaces=self.eigenfaces, 
                 projections=self.projections, 
                 labels=labels_arr,
                 unique_labels=unique_labels,
                 centroids=centroids_arr,
                 radii=radii_arr,
                 templates=templates_arr,
                 img_radii=img_radii_arr,
                 max_recon_error=float(self.max_recon_error))
        return True

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath, allow_pickle=True)
            self.mean_face = data['mean_face']
            self.eigenfaces = data['eigenfaces']
            self.projections = data['projections']
            self.labels = list(data['labels'])
            if 'max_recon_error' in data:
                self.max_recon_error = float(data['max_recon_error'])
            
            if 'unique_labels' in data and 'centroids' in data:
                u_labels = list(data['unique_labels'])
                centroids = data['centroids']
                radii = data['radii'] if 'radii' in data else [800.0] * len(u_labels)
                self.class_centroids = {u_labels[i]: centroids[i] for i in range(len(u_labels))}
                self.class_radii = {u_labels[i]: float(radii[i]) for i in range(len(u_labels))}
                
                if 'templates' in data and 'img_radii' in data:
                    templates = data['templates']
                    img_radii = data['img_radii']
                    self.class_mean_faces = {u_labels[i]: templates[i] for i in range(len(u_labels))}
                    self.class_img_radii = {u_labels[i]: float(img_radii[i]) for i in range(len(u_labels))}
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Initialize Haar Cascade face detector with dynamic fallbacks
face_cascade = None
try:
    if hasattr(cv2, 'CascadeClassifier'):
        # Try standard OpenCV package data
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            xml_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if os.path.exists(xml_path):
                face_cascade = cv2.CascadeClassifier(xml_path)
        
        # Fallback to local file check or manual cascade initialization if not already loaded
        if face_cascade is None or face_cascade.empty():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    else:
        print("WARNING: cv2 has no attribute 'CascadeClassifier'. Running in no-opencv fallback mode.")
except Exception as e:
    print("WARNING: Failed to load Haar Cascade face detector:", e)
    face_cascade = None

def detect_faces(gray_img):
    """
    Detects faces in a grayscale image.
    returns: list of bounding boxes (x, y, w, h)
    """
    if face_cascade is None or face_cascade.empty():
        return []
    try:
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except Exception as e:
        print("ERROR running detectMultiScale:", e)
        return []

def preprocess_face(gray_img, bbox, size=(128, 128)):
    """
    Crops, resizes, and normalizes a face based on its bounding box.
    """
    x, y, w, h = bbox
    cropped = gray_img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    # Perform histogram equalization to normalize lighting conditions
    equalized = cv2.equalizeHist(resized)
    return equalized

def base64_to_cv2(b64_string):
    """
    Converts a base64 encoded image string to an OpenCV BGR image.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_to_base64(img, format=".jpg"):
    """
    Converts an OpenCV image array to a base64 encoded string.
    """
    _, buffer = cv2.imencode(format, img)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"
