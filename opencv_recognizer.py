import os
import cv2
import numpy as np
import pickle
from pathlib import Path

class OpenCVFaceRecognizer:
    def __init__(self, known_faces_dir='known_faces', threshold=70):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)
        self.threshold = threshold
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Initialize with empty face database
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create a default face to ensure we always have something to train on
        self._add_default_face()
        
        # Load any existing known faces
        self.load_known_faces()
        
        # Ensure we have a trained model
        self._ensure_trained()
    
    def _add_default_face(self):
        """Add a default face to ensure we always have something to train on."""
        default_face = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(default_face, (20, 40), (40, 60), 255, -1)  # Left eye
        cv2.rectangle(default_face, (60, 40), (80, 60), 255, -1)  # Right eye
        cv2.ellipse(default_face, (50, 80), (30, 10), 0, 0, 180, 255, 2)  # Smile
        self.known_face_encodings = [default_face]
        self.known_face_names = ["default"]
    
    def _ensure_trained(self):
        """Ensure the recognizer is trained with at least the default face."""
        if len(self.known_face_encodings) == 0:
            self._add_default_face()
        self.retrain_recognizer()
    
    def load_known_faces(self):
        encodings_file = self.known_faces_dir / 'encodings.pkl'
        if encodings_file.exists():
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    if data and 'encodings' in data and 'names' in data:
                        if data['encodings'] and data['names']:
                            self.known_face_encodings = data['encodings']
                            self.known_face_names = data['names']
                            print(f'Loaded {len(self.known_face_encodings)} known faces')
            except Exception as e:
                print(f"Error loading known faces: {e}")
                # If loading fails, ensure we have at least the default face
                self._add_default_face()
    
    def save_known_faces(self):
        encodings_file = self.known_faces_dir / 'encodings.pkl'
        with open(encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
    
    def register_face(self, image, name):
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_img = gray[y:y+h, x:x+w]
            
            # Preprocess the face
            processed_face = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
            
            # If this is the first face being registered, clear the default face
            if len(self.known_face_names) == 1 and self.known_face_names[0] == "default":
                self.known_face_encodings = []
                self.known_face_names = []
            
            # Add to known faces
            self.known_face_encodings.append(processed_face)
            self.known_face_names.append(name)
            
            # Save and retrain
            self.save_known_faces()
            self.retrain_recognizer()
            
            return True, f"Successfully registered {name}"
        
        return False, "No face detected in the image"
    
    def retrain_recognizer(self):
        """Train the recognizer with the current set of known faces."""
        try:
            if len(self.known_face_encodings) == 0:
                self._add_default_face()
            
            # Ensure all face images are in the correct format
            faces = []
            for face_img in self.known_face_encodings:
                # Ensure the image is grayscale and uint8
                if len(face_img.shape) > 2:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)
                faces.append(face_img)
            
            # Create labels (0 to n-1)
            labels = np.array(list(range(len(faces))), dtype=np.int32)
            
            # Reinitialize the recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Train the recognizer
            if len(faces) > 0:
                self.face_recognizer.train(faces, labels)
                print(f"Recognizer trained with {len(faces)} faces")
            
        except Exception as e:
            print(f"Error in retrain_recognizer: {str(e)}")
            # If training fails, reinitialize with default face
            self._add_default_face()
            self.retrain_recognizer()
    
    def recognize_faces(self, image):
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
            
            # Default values
            name = "Unknown"
            is_unknown = True
            confidence = 100.0
            
            try:
                # Ensure we have a trained model
                if len(self.known_face_encodings) == 0:
                    self._ensure_trained()
                
                # Predict the label and confidence
                label, conf = self.face_recognizer.predict(face_img)
                
                # If we only have the default face, mark all detections as unknown
                if len(self.known_face_names) == 1 and self.known_face_names[0] == "default":
                    name = "Unknown"
                    is_unknown = True
                    confidence = 100.0
                elif 0 <= label < len(self.known_face_names):
                    name = self.known_face_names[label]
                    is_unknown = False
                    confidence = min(float(conf), 100.0)
                
            except Exception as e:
                print(f"Error during face recognition: {e}")
                # If prediction fails, retrain the model
                self.retrain_recognizer()
                confidence = 100.0
            
            results.append({
                'name': name,
                'box': (int(x), int(y), int(w), int(h)),
                'confidence': float(100.0 - confidence),  # Convert to percentage (lower is better)
                'is_unknown': is_unknown
            })
        
        return results
    
    def draw_boxes(self, image, results):
        img = image.copy()
        
        for result in results:
            x, y, w, h = result['box']
            name = result['name']
            confidence = result['confidence']
            
            # Draw rectangle around face
            color = (0, 255, 0) if not result['is_unknown'] else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.1f}%)"
            cv2.putText(img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img