from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
from PIL import Image
import numpy as np
import os
import socket
from datetime import datetime
import time
import threading
import geocoder
import json
import shutil
from facenet_recognizer import FaceNetRecognizer

def get_location():
    """Return a static location string since location functionality is not needed."""
    return "Location not tracked"

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

# Global variable to control monitoring state
monitoring_active = False
latest_identification = {
    "name": "Unknown",
    "time": None,
    "location": "Location not available",
    "face_encoding": None,
    "face_location": None,
    "is_unknown": True,
    "face_detected": False,
    "confidence": 0.0
}
ANTI_SPOOF_THRESHOLD = 0.95 # Adjusted threshold for anti-spoofing (increased for stricter classification)

# Initialize FaceNet recognizer
face_recognizer = FaceNetRecognizer(known_faces_dir='known_faces', threshold=0.5)

# Frame buffer for real-time streaming
current_frame = None
frame_lock = threading.Lock()

# Initialize face detection with OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading Haar Cascade. Face detection may not work properly.")
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
    # This is just a wrapper for compatibility
    return face_recognizer.known_face_encodings, face_recognizer.known_face_names

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
    try:
        # Ensure we have the latest data
        global latest_identification
        
        # Debug: Print the current state of latest_identification
        print("\n--- Latest Identification Data ---")
        print(f"Name: {latest_identification.get('name')}")
        print(f"Face Detected: {latest_identification.get('face_detected')}")
        print(f"Is Unknown: {latest_identification.get('is_unknown')}")
        print(f"Confidence: {latest_identification.get('confidence')}")
        
        # Create a new dictionary with only the data we want to send
        response_data = {
            'name': str(latest_identification.get('name', 'Unknown')),  # Ensure it's a string
            'time': latest_identification.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'location': str(latest_identification.get('location', 'Location not available')),  # Ensure it's a string
            'is_unknown': bool(latest_identification.get('is_unknown', True)),  # Ensure it's a boolean
            'face_detected': bool(latest_identification.get('face_detected', False)),  # Ensure it's a boolean
            'confidence': float(latest_identification.get('confidence', 0.0))  # Ensure it's a float
        }
        
        # Debug output
        print("\n--- Sending Response ---")
        print(f"Response Data: {response_data}")
        
        response = jsonify(response_data)
        
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        
        return response
        
    except Exception as e:
        print(f"Error in latest_identification_data: {str(e)}")
        return jsonify({
            'error': str(e),
            'name': 'Error',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_unknown': True,
            'face_detected': False,
            'confidence': 0.0
        }), 500


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
    try:
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'}), 400
            
        # Get the latest frame with a face
        with frame_lock:
            if current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available'}), 400
                
            # Register the face using FaceNet
            success, message = face_recognizer.register_face(current_frame, name)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Successfully registered {name}',
                    'count': len(face_recognizer.known_face_names)
                })
            else:
                return jsonify({'success': False, 'message': message}), 400
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error registering face: {str(e)}'
        }), 500

def process_frames_background(cap):
    """Background thread that processes frames for face recognition"""
    global current_frame, latest_identification, monitoring_active, frame_lock
    
    print("Starting face recognition background process...")
    
    while monitoring_active:
        try:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()
            
            # Make sure frame is valid
            if frame is None or frame.size == 0:
                print("Warning: Received empty frame")
                time.sleep(0.1)
                continue
            
            # Convert the image to RGB (MTCNN expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with FaceNet
            print("\n--- Processing Frame ---")
            print(f"Frame shape: {rgb_frame.shape}")
            
            results = face_recognizer.recognize_faces(rgb_frame)
            
            # Default identification when no faces are found
            current_identification = {
                "name": "No face detected",
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "location": get_location(),
                "is_unknown": False,
                "face_detected": False,
                "confidence": 0.0
            }
            
            print(f"Recognition results: {results}")
            
            # If faces are found, process them
            if results:
                # Draw boxes and get the first face's info
                frame = face_recognizer.draw_boxes(frame, results)
                first_face = results[0]
                
                # Ensure we have valid values for all fields
                current_identification = {
                    'name': str(first_face.get('name', 'Unknown')),  # Ensure it's a string
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'location': get_location(),
                    'is_unknown': bool(first_face.get('is_unknown', True)),  # Default to True for unknown
                    'confidence': float(first_face.get('confidence', 0.0)),
                    'face_detected': True
                }
                
                print(f"Identified: {first_face['name']} (confidence: {first_face['confidence']:.2f})")
            
            # Update the latest identification
            with frame_lock:
                latest_identification = current_identification
                
        except Exception as e:
            print(f"Error in process_frames_background: {e}")
            time.sleep(0.1)  # Prevent tight loop on error
            
    print("Background processing thread stopped")

def open_camera_jetson():
    # Try standard camera indices first (works on Mac/Windows/Linux)
    print("Attempting to open camera...")
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Try to read a frame to verify the camera works
            ret, frame = cap.read()
            if ret:
                print(f"✓ Successfully opened camera at index {camera_index}")
                # Set a reasonable resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
    
    print("✗ Failed to open any camera. Please check your camera connection.")
    return None

def generate_frames():
    """Lightweight frame generator - just streams frames, processing happens in background"""
    global monitoring_active, current_frame, camera_active, processing_thread, frame_lock
    
    cap = open_camera_jetson()
    
    if not cap or not cap.isOpened():
        print("=" * 70)
        print("ERROR: Could not open camera!")
        print("Possible reasons:")
        print("1. Camera is being used by another application")
        print("2. Camera permissions not granted")
        print("3. No camera available")
        print("=" * 70)
        
        # Generate a blank frame with an error message
        while True:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Multi-line error message
            lines = [
                "Camera Error!",
                "Please check:",
                "1. Camera is connected",
                "2. No other app is using the camera",
                "3. Camera permissions are granted"
            ]
            
            y_offset = 100
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

    # Main loop for capturing and processing frames
    while camera_active:
        # Read a new frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera")
            time.sleep(0.1)
            continue
            
        # Update the current frame for the background thread
        with frame_lock:
            current_frame = frame.copy()
            
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
