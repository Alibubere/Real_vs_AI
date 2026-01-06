import requests
import zipfile
import os

def download_dataset():
    """Download and extract the Real vs AI dataset"""
    url = "https://storage.googleapis.com/kaggle-data-sets/9067954/14215730/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260103T131302Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=42ea14e8367a6adc242eef295f67c845bb54b16e43f18a7e2a35e0cfb84fc7aeaa3d57a94a670e9d12ad6a5bf2c92c7a418456b1a7373dee626a8f08faee61ccce23fadb3bc9a9ccd4dfaf49d21c6dba8d55d3a81b05a8bcf91d79263416904c1bdd573331d8eb127c1eb102087608043e1eed8aff29732155519f4797c6b357a107eea6165a4a2b57130d0ff27c4a829d7b696de682926ad82e54b3c23229b1dc4ae7de1f5c3a4a9e3bd9013dfa24319368a14b6f817dee36c10409986f5b13853e6273f5ceaf41dce088eb765c9faa6d6d6cd88f4f87bdbf3126e55feaa1cfb6e07182aada0d8ba5c489bc8d563d9116f9238f33933870b49fdd0d6c331c50"
    
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open("dataset.zip", "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="")
    
    print("\nExtracting dataset...")
    os.makedirs("data", exist_ok=True)
    
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    
    os.remove("dataset.zip")
    print("Dataset downloaded and extracted to data/ folder")

if __name__ == "__main__":
    download_dataset()