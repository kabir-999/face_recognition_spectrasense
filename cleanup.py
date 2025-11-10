import os
import shutil
import json

def reset_face_data():
    # Define directories
    known_faces_dir = "known_faces"
    
    # Remove existing known_faces directory if it exists
    if os.path.exists(known_faces_dir):
        shutil.rmtree(known_faces_dir)
        print(f"Removed existing directory: {known_faces_dir}")
    
    # Create fresh directory structure
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Create metadata file
    metadata = {
        "known_faces": []
    }
    
    with open(os.path.join(known_faces_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nSystem has been reset. All face data has been cleared.")
    print("New face data will be stored in the 'known_faces' directory.")
    print("Each person will have their own subdirectory containing their face images and encodings.\n")

if __name__ == "__main__":
    print("This will delete all existing face data and reset the system.")
    confirm = input("Are you sure you want to continue? (y/n): ")
    if confirm.lower() == 'y':
        reset_face_data()
    else:
        print("Operation cancelled. No changes were made.")
