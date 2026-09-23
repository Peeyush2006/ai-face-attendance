import os
import cv2
import numpy as np
import base64

def validate_face_quality(gray_crop, min_size=36, min_blur=18.0):
    """
    Validates face crop to reject unusable detections:
    - extremely small faces
    - heavily blurred faces
    - invalid bounding boxes
    - corrupted crops / zero variance / non-face objects
    """
    if gray_crop is None or gray_crop.size == 0:
        return False, "Empty or corrupted crop"
    
    h, w = gray_crop.shape[:2]
    if w < min_size or h < min_size:
        return False, f"Face size too small ({w}x{h} < {min_size})"
        
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.55 or aspect_ratio > 1.45:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f})"
        
    std_val = float(np.std(gray_crop))
    mean_val = float(np.mean(gray_crop))
    if std_val < 12.0 or mean_val < 10.0 or mean_val > 245.0:
        return False, f"Insufficient contrast or uniform lighting (std={std_val:.1f}, mean={mean_val:.1f})"
        
    blur_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if blur_score < min_blur:
        return False, f"Face crop heavily blurred (Laplacian={blur_score:.1f} < {min_blur})"
        
    return True, "Valid"

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
        self.max_recon_error = min(max(max_train_recon * 1.8, 20.0), 38.0)
        
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
            self.class_radii[label] = max(min(raw_radius * 2.2, 2800.0), 1000.0)
            
            # Image space template and pixel variance radius
            mean_img = np.mean(class_X, axis=0)
            self.class_mean_faces[label] = mean_img
            img_dists = np.linalg.norm(class_X - mean_img, axis=1) / np.sqrt(D)
            raw_img_radius = float(np.max(img_dists)) if len(img_dists) > 0 else 20.0
            self.class_img_radii[label] = max(min(raw_img_radius * 2.2, 55.0), 20.0)
            
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
        q_proj_norm = float(np.linalg.norm(q_proj))
        if q_proj_norm < 1e-6:
            return None, float('inf'), 0.0
            
        # 2. Compute Reconstruction Residual (Distance to Face Space - DFFS)
        q_recon_centered = q_proj @ self.eigenfaces.T
        recon_error = float(np.linalg.norm(q_centered - q_recon_centered) / np.sqrt(D))
        
        # 3. Vectorized Subspace Distances and Cosine Similarities against ALL registered projections
        distances = np.linalg.norm(self.projections - q_proj, axis=1)
        proj_norms = np.linalg.norm(self.projections, axis=1)
        cos_sims = (self.projections @ q_proj) / (proj_norms * q_proj_norm + 1e-10)
        
        unique_labels = list(dict.fromkeys(self.labels))
        class_scores = {}
        has_templates = hasattr(self, 'class_mean_faces') and bool(self.class_mean_faces)
        
        for l in unique_labels:
            idxs = [i for i, lab in enumerate(self.labels) if lab == l]
            min_d = float(np.min(distances[idxs]))
            best_c = float(np.max(cos_sims[idxs]))
            
            if has_templates and l in self.class_mean_faces:
                template = self.class_mean_faces[l]
                rmse = float(np.linalg.norm(q - template) / np.sqrt(D))
            else:
                rmse = 0.0
                
            class_scores[l] = {
                "min_d": min_d,
                "best_c": best_c,
                "rmse": rmse
            }
            
        # Sort candidates primarily by cosine similarity in PCA subspace, secondarily by Euclidean distance
        sorted_candidates = sorted(unique_labels, key=lambda l: (class_scores[l]["best_c"], -class_scores[l]["min_d"]), reverse=True)
        candidate_label = sorted_candidates[0]
        cand_score = class_scores[candidate_label]
        
        min_sample_dist = cand_score["min_d"]
        best_cos = cand_score["best_c"]
        best_rmse = cand_score["rmse"]
        
        second_cos = class_scores[sorted_candidates[1]]["best_c"] if len(sorted_candidates) > 1 else -1.0
        
        # Adaptive manifold and distance boundaries
        allowable_recon = max(getattr(self, 'max_recon_error', 20.0), 62.0)
        allowable_dist = max(self.class_radii.get(candidate_label, 1800.0) * 1.5, 3000.0) if hasattr(self, 'class_radii') else 3000.0
        allowable_rmse = max(self.class_img_radii.get(candidate_label, 30.0) * 1.8, 56.0) if hasattr(self, 'class_img_radii') else 56.0
        min_required_cos = 0.65
        
        # Strict Rejection Criteria (Ensure an unknown or object NEVER falsely matches a student):
        # 1. Non-face manifold residual: non-face objects / corrupted crops do not lie on the face manifold
        # 2. Subspace Cosine Similarity: must exhibit clear facial component alignment
        # 3. Subspace Distance: must fall within candidate cluster boundary
        # 4. Image Template RMSE (if enrolled): pixel structure must not drastically deviate
        # 5. Multi-candidate margin: must distinguish cleanly from 2nd best candidate
        is_stranger = (
            recon_error > allowable_recon or
            best_cos < min_required_cos or
            min_sample_dist > allowable_dist or
            (has_templates and candidate_label in self.class_mean_faces and best_rmse > allowable_rmse) or
            (len(unique_labels) > 1 and best_cos < 0.85 and (best_cos - second_cos) < 0.04)
        )
        
        # Composite confidence score:
        # Subspace cosine similarity is scale/lighting invariant and primary (60%)
        # Euclidean distance in eigenspace relative to cluster boundary (25%)
        # Face manifold reconstruction residual (15%)
        conf_cos = max(0.0, min(1.0, (best_cos - 0.45) / 0.50))
        conf_dist = max(0.0, min(1.0, 1.0 - min_sample_dist / allowable_dist))
        conf_recon = max(0.0, min(1.0, 1.0 - (recon_error / allowable_recon)))
        confidence = float(0.60 * conf_cos + 0.25 * conf_dist + 0.15 * conf_recon)
        
        if is_stranger or confidence < threshold:
            # Rejection confirmed - return None for identity
            rejected_conf = max(0.0, min(0.48, confidence if not is_stranger else conf_cos * 0.4))
            return None, min_sample_dist, float(rejected_conf)
            
        return candidate_label, min_sample_dist, float(confidence)

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
        
        # Remove existing file if present to prevent Windows file locking issues in zipfile
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
                
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

# Base directory path for models
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_PATH = os.path.join(_BASE_DIR, 'data', 'models', 'face_detection_yunet_2023mar.onnx')

# Initialize YuNet Deep Learning face detector (Primary, works with OpenCV 4.x and 5.x)
detector_yunet = None
if hasattr(cv2, 'FaceDetectorYN_create') and os.path.exists(_YUNET_PATH):
    try:
        detector_yunet = cv2.FaceDetectorYN_create(
            _YUNET_PATH, "", (320, 320),
            score_threshold=0.55,
            nms_threshold=0.3
        )
        print("[face_rec] YuNet deep-learning face detector initialized successfully.")
    except Exception as e:
        print("[face_rec] Could not initialize YuNet detector:", e)
        detector_yunet = None

# Initialize Haar Cascade face detector (Secondary fallback)
face_cascade = None
try:
    if hasattr(cv2, 'CascadeClassifier'):
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            xml_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if os.path.exists(xml_path):
                face_cascade = cv2.CascadeClassifier(xml_path)
        
        if face_cascade is None or face_cascade.empty():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    face_cascade = None

def detect_faces(img, min_size=36, strict_quality=True):
    """
    Detects and validates faces in an image (accepts BGR or grayscale numpy arrays).
    Rejects non-face objects, false detections, corrupted crops, and unusable faces.
    returns: list of bounding boxes [(x, y, w, h)]
    """
    if img is None or img.size == 0:
        return []

    if len(img.shape) == 2:
        h, w = img.shape
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray_img = img
    else:
        h, w = img.shape[:2]
        bgr_img = img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Primary: YuNet Deep Learning detector
    if detector_yunet is not None:
        try:
            detector_yunet.setInputSize((w, h))
            _, raw_faces = detector_yunet.detect(bgr_img)
            if raw_faces is not None and len(raw_faces) > 0:
                result = []
                for f in raw_faces:
                    score = float(f[14]) if len(f) > 14 else 1.0
                    # Reject detections below detector confidence threshold
                    if score < 0.50:
                        continue
                        
                    x = max(0, int(round(f[0])))
                    y = max(0, int(round(f[1])))
                    bw = max(1, min(w - x, int(round(f[2]))))
                    bh = max(1, min(h - y, int(round(f[3]))))
                    
                    if bw < min_size or bh < min_size:
                        continue
                        
                    if strict_quality:
                        crop = gray_img[y:y+bh, x:x+bw]
                        is_valid, _ = validate_face_quality(crop, min_size=min_size, min_blur=15.0)
                        if not is_valid:
                            continue
                            
                    result.append((x, y, bw, bh))
                if len(result) > 0:
                    return result
        except Exception as e:
            print("[face_rec] YuNet detection error:", e)

    # 2. Secondary fallback: Haar Cascade
    if face_cascade is not None and not face_cascade.empty():
        try:
            faces = face_cascade.detectMultiScale(
                gray_img, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(min_size, min_size)
            )
            if len(faces) > 0:
                result = []
                for (x, y, bw, bh) in faces:
                    x = max(0, int(x))
                    y = max(0, int(y))
                    bw = max(1, min(w - x, int(bw)))
                    bh = max(1, min(h - y, int(bh)))
                    
                    if strict_quality:
                        crop = gray_img[y:y+bh, x:x+bw]
                        is_valid, _ = validate_face_quality(crop, min_size=min_size, min_blur=20.0)
                        if not is_valid:
                            continue
                    result.append((x, y, bw, bh))
                return result
        except Exception as e:
            print("[face_rec] Haar cascade detection error:", e)

    return []

def preprocess_face(img, bbox, size=(128, 128)):
    """
    Crops, resizes, and normalizes a face based on its bounding box.
    Accepts grayscale or BGR images.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h_img, w_img = gray.shape[:2]
    x, y, w, h = bbox
    x = max(0, min(int(x), w_img - 1))
    y = max(0, min(int(y), h_img - 1))
    w = max(1, min(int(w), w_img - x))
    h = max(1, min(int(h), h_img - y))

    cropped = gray[y:y+h, x:x+w]
    if cropped.size == 0:
        cropped = gray

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
