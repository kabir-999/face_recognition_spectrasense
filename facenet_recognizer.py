import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from PIL import Image
from pathlib import Path
import pickle

class FaceNetRecognizer:
    def __init__(self, known_faces_dir='known_faces', threshold=0.8):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')
        
        # Initialize MTCNN for face detection with more sensitive parameters
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device,
            min_face_size=20,  # Lower minimum face size to detect smaller faces
            thresholds=[0.6, 0.7, 0.7],  # Lower thresholds for face detection
            factor=0.709,  # Scale factor for image pyramid
            post_process=False
        )
        
        # Initialize Inception Resnet V1 for face recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)
        
        self.threshold = threshold
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces if they exist
        self.load_known_faces()
    
    def load_known_faces(self):
        encodings_file = self.known_faces_dir / 'encodings.pkl'
        if encodings_file.exists():
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f'Loaded {len(self.known_face_encodings)} known faces')
    
    def save_known_faces(self):
        encodings_file = self.known_faces_dir / 'encodings.pkl'
        with open(encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
    
    def register_face(self, image, name):
        # Convert image to RGB if it's BGR (OpenCV format)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Detect face
        boxes, _ = self.mtcnn.detect(image)
        
        if boxes is not None and len(boxes) > 0:
            # Get the largest face
            box = boxes[0]
            face = extract_face(image, box, image_size=160, margin=40)
            
            # Get face encoding
            face_tensor = face.unsqueeze(0).to(self.device)
            encoding = self.resnet(face_tensor).detach().cpu().numpy()[0]
            
            # Save the new face
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            self.save_known_faces()
            
            return True, "Face registered successfully"
        
        return False, "No face detected in the image"
    
    def recognize_faces(self, image):
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        else:
            image_pil = image
            image_rgb = np.array(image_pil)
        
        # Debug: Save the input image
        debug_dir = Path('debug')
        debug_dir.mkdir(exist_ok=True)
        debug_image_path = debug_dir / 'debug_input.jpg'
        cv2.imwrite(str(debug_image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Debug: Saved input image to {debug_image_path}")
        
        # Detect faces with timing
        print("Detecting faces with MTCNN...")
        boxes, probs = self.mtcnn.detect(image_pil)
        
        print(f"MTCNN detection results - Boxes: {boxes}, Probabilities: {probs}")
        
        results = []
        
        if boxes is not None:
            print(f"Found {len(boxes)} face(s) in the image")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                prob = probs[i] if probs is not None and i < len(probs) else 1.0
                print(f"Face {i+1}: Box=({x1}, {y1}, {x2}, {y2}), Confidence={prob:.2f}")
                
                # Extract face
                face = extract_face(image_pil, box, image_size=160, margin=40)
                
                # Get face encoding
                face_tensor = face.unsqueeze(0).to(self.device)
                encoding = self.resnet(face_tensor).detach().cpu().numpy()[0]
                
                # Compare with known faces
                name = "Unknown"
                confidence = 0.0
                
                if len(self.known_face_encodings) > 0:
                    # Calculate distances to known faces
                    distances = [np.linalg.norm(encoding - known_enc) 
                               for known_enc in self.known_face_encodings]
                    
                    if distances:
                        min_distance = min(distances)
                        if min_distance < self.threshold:
                            best_match_idx = np.argmin(distances)
                            name = self.known_face_names[best_match_idx]
                            confidence = 1.0 - (min_distance / self.threshold)
                
                results.append({
                    'box': (x1, y1, x2, y2),
                    'name': name,
                    'confidence': confidence,
                    'is_unknown': name == "Unknown"
                })
        
        return results

    def draw_boxes(self, image, results):
        img_with_boxes = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['box']
            name = result['name']
            confidence = result['confidence']
            
            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(img_with_boxes, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img_with_boxes, label, (x1 + 6, y2 - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return img_with_boxes
