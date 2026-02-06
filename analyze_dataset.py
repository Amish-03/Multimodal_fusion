
import os
from collections import defaultdict
from PIL import Image

dataset_root = r"C:\Users\amish\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset"

def analyze_directory(path, category):
    print(f"Analyzing {category} dataset at: {path}")
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return

    file_counts = defaultdict(int)
    extensions = defaultdict(int)
    image_sizes = defaultdict(int)
    
    total_files = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            extensions[ext] += 1
            
            # Count files per subdirectory (assuming leaf directories are classes/subjects)
            # relative_path = os.path.relpath(root, path)
            # parent_dir = relative_path.split(os.sep)[0] 
            # file_counts[parent_dir] += 1 # This might be too verbose if there are many folders

            # Check dimensions for the first few images of each extension
            if extensions[ext] <= 5 and ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                 try:
                     with Image.open(os.path.join(root, file)) as img:
                         image_sizes[f"{ext} {img.size}"] += 1
                 except Exception as e:
                     print(f"Error reading image {file}: {e}")

    print(f"Total files in {category}: {total_files}")
    print("File extensions:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count}")
    
    print("Sample image dimensions (Extension (Width, Height)): count:")
    for size, count in image_sizes.items():
        print(f"  {size}: {count}")
    print("-" * 20)

analyze_directory(os.path.join(dataset_root, "hand dataset"), "Hand")
analyze_directory(os.path.join(dataset_root, "iris dataset"), "Iris")
