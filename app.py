from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np # Import numpy
import os
from datetime import datetime
import face_recognition # Import face_recognition
import pickle # Import pickle for caching
import random # Import random for selecting subset of images
import geocoder # Import geocoder for location services
import json
import time
import shutil
import threading
from collections import deque


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
KNOWN_FACES_DIR = "known_faces"
METADATA_FILE = os.path.join(KNOWN_FACES_DIR, "metadata.json")
CACHE_FILE = os.path.join(KNOWN_FACES_DIR, "face_encodings_cache.pkl")
OUTPUT_FRAMES_DIR = "/Users/kabirmathur/Documents/a_s/Kabir_Mathur"
IDENTIFIED_PERSONS_DIR = "/Users/kabirmathur/Documents/a_s/identified_persons"
os.makedirs(IDENTIFIED_PERSONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Global variable to control monitoring state
monitoring_active = False
latest_identification = {
    "name": None,
    "time": None,
    "location": None,
    "face_encoding": None,
    "face_location": None,
    "is_unknown": False
}
ANTI_SPOOF_THRESHOLD = 0.95 # Adjusted threshold for anti-spoofing (increased for stricter classification)

# Face Recognition variables (initialized globally)
known_face_encodings = []
known_face_names = []
last_saved_time = {}

# Frame buffer for real-time streaming
current_frame = None
frame_lock = threading.Lock()
processing_thread = None
camera_active = False

def save_face_encoding(name, encoding, image):
    """Save a new face encoding and its image to the known faces directory."""
    # Create output directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    # Load or create metadata
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"known_faces": []}
    
    # Create person's directory if it doesn't exist
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    # Save the face image
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Load or create encodings for this person
    encodings_file = os.path.join(person_dir, 'encodings.pkl')
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            person_encodings = pickle.load(f)
    else:
        person_encodings = []
        metadata['known_faces'].append(name)
    
    # Add new encoding
    person_encodings.append(encoding)
    
    # Save updated encodings
    with open(encodings_file, 'wb') as f:
        pickle.dump(person_encodings, f)
    
    # Save updated metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    
    # Reload known faces
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()
    
    print(f"Saved new face: {name} ({filename})")
    return True

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Create the known_faces directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    # If metadata file doesn't exist, create it
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({"known_faces": []}, f)
        return known_face_encodings, known_face_names
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Process each known person
    for person_name in metadata['known_faces']:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.exists(person_dir):
            continue
            
        # Load encodings if they exist
        encodings_file = os.path.join(person_dir, 'encodings.pkl')
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    person_encodings = pickle.load(f)
                    known_face_encodings.extend(person_encodings)
                    known_face_names.extend([person_name] * len(person_encodings))
            except Exception as e:
                print(f"Error loading encodings for {person_name}: {e}")
    
    # Save to cache
    if known_face_encodings:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
    
    print(f"Loaded {len(known_face_encodings)} known faces from {len(set(known_face_names))} different people.")
    return known_face_encodings, known_face_names

# Load known faces on startup
known_face_encodings, known_face_names = load_known_faces()

# Initialize anti-spoofing model as None (optional)
anti_spoof_model = None
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'antispoof_vit_traced.pt')

try:
    # Try to load the anti-spoofing model if available
    if os.path.exists(model_path):
        anti_spoof_model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        anti_spoof_model.eval()
        anti_spoof_model = anti_spoof_model.to(torch.float32).to(device)
        print("Anti-spoofing model loaded successfully")
    else:
        print("Warning: Anti-spoofing model not found. Running without anti-spoofing.")
except Exception as e:
    print(f"Warning: Could not load anti-spoofing model: {e}")
    print("Running without anti-spoofing functionality")

# Define preprocessing for the anti-spoofing model (kept for compatibility)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Reverted to 0.5 mean/std
])

# Initialize YOLO face detector with explicit configuration
try:
    # Try loading with the latest API first
    yolo_model = YOLO('yolov8n.pt')
    # Test the model with a dummy input to ensure it's properly loaded
    yolo_model.predict(torch.zeros((1, 3, 640, 640), device=device), verbose=False)
    print("YOLO model loaded successfully with default settings")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Trying alternative loading method...")
    try:
        # Fallback to explicit model loading with CPU first
        yolo_model = YOLO('yolov8n.pt', task='detect')
        yolo_model = yolo_model.cpu()
        # Test with a small image
        yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("YOLO model loaded successfully with CPU-first method")
    except Exception as e2:
        print(f"Failed to load YOLO model: {e2}")
        raise RuntimeError("Could not initialize YOLO model. Please check your installation and try again.")

# Move model to the appropriate device
yolo_model.to(device)
if device.type == 'cuda':
    yolo_model = yolo_model.half()  # Use half precision for CUDA
    print("Using half precision (FP16) for YOLO model on CUDA")
else:
    yolo_model = yolo_model.float()  # Use full precision for CPU/MPS
    print("Using full precision (FP32) for YOLO model")

# Get class names from the YOLO model
class_names = yolo_model.names
face_class_id = None
for class_id, name in class_names.items():
    if name == 'person': # YOLOv8n typically detects 'person' for faces
        face_class_id = class_id
        break

if face_class_id is None:
    print("Warning: 'person' class not found in YOLO model. Face detection might not work as expected.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_monitoring')
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    print("=" * 50)
    print("Monitoring started via Flask route.")
    print("monitoring_active =", monitoring_active)
    print("=" * 50)
    return "Monitoring started"

@app.route('/stop_monitoring')
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    print("=" * 50)
    print("Monitoring stopped via Flask route.")
    print("monitoring_active =", monitoring_active)
    print("=" * 50)
    return "Monitoring stopped"

@app.route('/video_feed')
def video_feed():
    print("Video feed endpoint called - starting frame generation")
    response = Response(generate_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering
    return response

@app.route('/latest_identification')
def latest_identification_data():
    # Create a new dictionary with only the data we want to send
    response_data = {
        'name': latest_identification.get('name'),
        'time': latest_identification.get('time'),
        'location': latest_identification.get('location'),
        'is_unknown': latest_identification.get('is_unknown', False)
    }
    return jsonify(response_data)

def clear_known_faces():
    """Clear all known face data and reload from disk."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    # Reload known faces from disk
    load_known_faces()

@app.route('/clear_faces', methods=['POST'])
def clear_faces():
    """Clear all known faces and reset the system."""
    try:
        # Remove the entire known_faces directory
        if os.path.exists(KNOWN_FACES_DIR):
            shutil.rmtree(KNOWN_FACES_DIR)
        
        # Recreate the directory structure
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        
        # Reset global variables
        global known_face_encodings, known_face_names
        known_face_encodings = []
        known_face_names = []
        
        # Create fresh metadata file
        with open(METADATA_FILE, 'w') as f:
            json.dump({"known_faces": []}, f)
        
        return jsonify({
            "status": "success", 
            "message": "Successfully cleared all known faces. The system has been reset."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to clear known faces: {str(e)}"
        }), 500

@app.route('/register_face', methods=['POST'])
def register_face():
    global latest_identification, known_face_encodings, known_face_names
    
    data = request.get_json()
    name = data.get('name')
    
    if not name or not latest_identification.get('is_unknown') or not latest_identification.get('face_encoding'):
        return jsonify({
            "status": "error", 
            "message": "Invalid request or no face to register. Make sure a face is detected and marked as unknown."
        }), 400
    
    # Get the face encoding and image
    face_encoding = np.array(latest_identification['face_encoding'])
    face_location = latest_identification['face_location']
    
    # Save the new face encoding
    face_encoding = latest_identification['face_encoding']
    face_image = latest_identification.get('last_frame')
    
    if face_image is None:
        return jsonify({"status": "error", "message": "No face image available"}), 400
        
    # Convert face_encoding from list back to numpy array if needed
    if isinstance(face_encoding, list):
        face_encoding = np.array(face_encoding)
    
    # Extract face from frame using face_location
    top, right, bottom, left = face_location
    face_image = face_image[top:bottom, left:right]
    
    # Clear and reload known faces to ensure consistency
    clear_known_faces()
    
    # Save the new face encoding and image to disk
    save_face_encoding(name, face_encoding, face_image)
    
    # Reload known faces to include the newly added one
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()
    
    # Update the latest identification with the new face data
    latest_identification.update({
        'name': name,
        'is_unknown': False,
        'face_encoding': face_encoding.tolist(),
        'face_location': face_location,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print(f"Successfully registered new face: {name}")
    print(f"Total known faces after registration: {len(known_face_encodings)}")
    
    return jsonify({"status": "success", "message": f"Face registered as {name}"})

def process_frames_background(cap):
    """Background thread that processes frames for face recognition"""
    global current_frame, monitoring_active, latest_identification
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for face recognition
    
    print("Background processing thread started")
    
    while camera_active:
        if not monitoring_active:
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame in background thread")
            time.sleep(0.1)
            continue
        
        # Store raw frame for streaming (always update for smooth video)
        with frame_lock:
            current_frame = frame.copy()
        
        # Only do heavy processing every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            frame_count += 1
            continue
            
        frame_count += 1
        
        # Convert frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        latest_identification['last_frame'] = rgb_frame.copy()
        
        # Run YOLO detection
        results = yolo_model(frame)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if face_class_id is not None and int(box.cls[0]) != face_class_id:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                # Anti-spoofing check (only if model is available)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                is_real = True  # Default to true if no anti-spoofing model
                if anti_spoof_model is not None:
                    try:
                        face_image = Image.fromarray(face_rgb)
                        input_tensor = preprocess(face_image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = anti_spoof_model(input_tensor)
                            spoof_prob = torch.sigmoid(output).item()
                        is_real = spoof_prob < 0.5  # Threshold can be adjusted
                    except Exception as e:
                        print(f"Anti-spoofing check failed: {e}")
                        is_real = True  # Default to true if check fails
                
                if is_real:
                    # Face recognition
                    face_image_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(face_image_rgb)
                    
                    if face_locations:
                        face_encodings_list = face_recognition.face_encodings(face_image_rgb, face_locations)
                        
                        if face_encodings_list:
                            face_encoding = face_encodings_list[0]
                            
                            if len(known_face_encodings) > 0:
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if face_distances[best_match_index] < 0.6:
                                    identified_name = known_face_names[best_match_index]
                                    is_unknown = False
                                else:
                                    identified_name = "Unknown"
                                    is_unknown = True
                            else:
                                identified_name = "Unknown"
                                is_unknown = True
                            
                            # Draw on the frame
                            with frame_lock:
                                if is_unknown:
                                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                    cv2.putText(current_frame, "Unknown - Click to Register", (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                else:
                                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(current_frame, identified_name, (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Update identification info
                            now = datetime.now()
                            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                            g = geocoder.ip('me')
                            location = g.city if g.city else "Unknown Location"
                            
                            latest_identification["name"] = identified_name
                            latest_identification["time"] = current_time
                            latest_identification["location"] = location
                            latest_identification["face_encoding"] = face_encoding.tolist()
                            latest_identification["face_location"] = face_locations[0]
                            latest_identification["is_unknown"] = is_unknown
                            
                            # Save known faces
                            if not is_unknown:
                                current_timestamp = time.time()
                                if identified_name not in last_saved_time or (current_timestamp - last_saved_time[identified_name]) > 5:
                                    filename = f"{identified_name},{current_time},{location}.jpg".replace(" ", "_").replace(":", "-")
                                    filepath = os.path.join(IDENTIFIED_PERSONS_DIR, filename)
                                    cv2.imwrite(filepath, frame)
                                    last_saved_time[identified_name] = current_timestamp
    
    print("Background processing thread stopped")

def open_camera_jetson():
    # Try standard camera indices first (works on Mac/Windows/Linux)
    print("Attempting to open camera...")
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully opened camera at index {camera_index}")
                return cap
            else:
                print(f"✗ Camera {camera_index} opened but cannot read frames")
                cap.release()
        else:
            print(f"✗ Failed to open camera at index {camera_index}")
    
    # Try GStreamer pipeline for Jetson Nano CSI camera (if on Jetson)
    print("Trying GStreamer pipeline for Jetson Nano...")
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("✓ Successfully opened camera using GStreamer pipeline.")
        return cap
    
    print("✗ Failed to open camera with any method.")
    return None

def generate_frames():
    """Lightweight frame generator - just streams frames, processing happens in background"""
    global monitoring_active, current_frame, camera_active, processing_thread
    
    cap = open_camera_jetson()

    if not cap or not cap.isOpened():
        print("=" * 70)
        print("ERROR: Could not open camera!")
        print("Possible reasons:")
        print("1. Camera is being used by another application (Vision Encoder tab?)")
        print("2. Camera permissions not granted")
        print("3. No camera available")
        print("=" * 70)
        # Generate a blank frame with an error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Multi-line error message
        lines = [
            "Camera Error!",
            "Camera may be in use by Vision Encoder tab",
            "or another application.",
            "Stop other camera apps and refresh."
        ]
        
        y_offset = 150
        for i, line in enumerate(lines):
            font_scale = 0.8 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            color = (0, 0, 255) if i == 0 else (255, 255, 255)
            
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = y_offset + (i * 40)
            cv2.putText(img, line, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\n' 
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            time.sleep(0.5)  # Reduce CPU usage

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warm-up the camera
    for _ in range(5):
        cap.read()
    
    # Start background processing thread
    camera_active = True
    processing_thread = threading.Thread(target=process_frames_background, args=(cap,), daemon=True)
    processing_thread.start()
    print("Started background processing thread for face recognition")
    
    # Initialize current_frame with first frame
    success, frame = cap.read()
    if success:
        with frame_lock:
            current_frame = frame.copy()

    # Lightweight streaming loop - just yields frames as fast as possible
    while camera_active:
        # Get the latest processed frame from background thread
        with frame_lock:
            if current_frame is None:
                # No frame yet, show waiting message
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if not monitoring_active:
                    text = "Monitoring Paused - Click 'Start Monitoring' to begin"
                else:
                    text = "Initializing camera..."
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                text_x = (blank_frame.shape[1] - text_size[0]) // 2
                text_y = (blank_frame.shape[0] + text_size[1]) // 2
                cv2.putText(blank_frame, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                frame_to_stream = blank_frame
            else:
                frame_to_stream = current_frame.copy()
        
        # Encode and yield frame immediately (no processing delay!)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', frame_to_stream, encode_param)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small sleep to target ~30 FPS
        time.sleep(0.033)  # ~30 FPS
    
    # Cleanup
    camera_active = False
    if processing_thread:
        processing_thread.join(timeout=2.0)
    cap.release()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5006, debug=False, threaded=True)
