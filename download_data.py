import kagglehub

# Download latest version
print("Starting download for 'olankadhim/multimodal-biometric-dataset-mulb'...", flush=True)
try:
    path = kagglehub.dataset_download("olankadhim/multimodal-biometric-dataset-mulb")
    print("Download successful!")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"An error occurred during download: {e}")
