import os
import requests
import bz2
import shutil

MODELS = {
    "scrfd_10g_bnkps.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx",
    "w600k_r50.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    "shape_predictor_68_face_landmarks.dat.bz2": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "genderage.onnx": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/genderage.onnx"
}

OUTPUT_DIR = "models"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename, url in MODELS.items():
        dest_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(dest_path) or (filename.endswith('.bz2') and os.path.exists(dest_path[:-4])):
            print(f"{filename} (or extracted version) already exists. Skipping.")
            continue
            
        download_file(url, dest_path)
        
        if filename.endswith('.bz2'):
            extracted_path = dest_path[:-4]
            print(f"Extracting {filename}...")
            try:
                with bz2.BZ2File(dest_path) as fr, open(extracted_path, "wb") as fw:
                    shutil.copyfileobj(fr, fw)
                print(f"Extracted to {extracted_path}")
                os.remove(dest_path) # Clean up bz2
            except Exception as e:
                print(f"Extraction failed: {e}")

    print("\nAll models set up. You can now run 'python main.py'.")

if __name__ == "__main__":
    main()
