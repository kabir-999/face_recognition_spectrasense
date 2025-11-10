import cv2

def test_webcam():
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened camera at index {i}")
            ret, frame = cap.read()
            if ret:
                print(f"Successfully read frame from camera {i}")
                # Save the frame to check if it's working
                cv2.imwrite(f'webcam_test_{i}.jpg', frame)
                print(f"Saved test frame from camera {i} as 'webcam_test_{i}.jpg'")
            else:
                print(f"Could not read frame from camera {i}")
            cap.release()
        else:
            print(f"Failed to open camera at index {i}")

if __name__ == "__main__":
    test_webcam()
