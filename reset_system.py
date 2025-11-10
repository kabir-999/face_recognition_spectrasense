import os
import shutil
import json

def reset_system():
    # Create necessary directories
    output_dir = "known_faces"
    
    # Remove existing directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create fresh directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a metadata file to store known faces
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump({"known_faces": []}, f)
    
    print("System reset complete. All known faces have been cleared.")
    print(f"New face data will be saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    reset_system()
