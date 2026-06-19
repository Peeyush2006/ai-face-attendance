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

    def train(self, faces_list, labels_list):
        """
        faces_list: list of grayscale 2D numpy arrays of shape self.img_size
        labels_list: list of labels (student_ids) corresponding to each face
        """
        if not faces_list or len(faces_list) == 0:
            return False
        
        # Flatten images to 1D vectors
        X = np.array([f.flatten() for f in faces_list], dtype=np.float32)
        self.labels = list(labels_list)
        
        N, D = X.shape
        
        # Calculate mean face
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
        return True

    def predict(self, face_img):
        """
        face_img: grayscale 2D numpy array of shape self.img_size
        returns: (matched_label, distance)
        """
        if self.eigenfaces is None or self.mean_face is None or self.projections is None:
            return None, float('inf')
            
        q = face_img.flatten().astype(np.float32)
        q_centered = q - self.mean_face
        
        # Project onto the eigenfaces space
        q_proj = q_centered @ self.eigenfaces
        
        # Calculate Euclidean distances to all training projections
        distances = np.linalg.norm(self.projections - q_proj, axis=1)
        
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        matched_label = self.labels[min_idx]
        
        return matched_label, float(min_dist)

    def save(self, filepath):
        if self.mean_face is None:
            return False
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez(filepath, 
                 mean_face=self.mean_face, 
                 eigenfaces=self.eigenfaces, 
                 projections=self.projections, 
                 labels=np.array(self.labels, dtype=object))
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
            return True
        except Exception:
            return False

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(gray_img):
    """
    Detects faces in a grayscale image.
    returns: list of bounding boxes (x, y, w, h)
    """
    if face_cascade.empty():
        return []
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces

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
