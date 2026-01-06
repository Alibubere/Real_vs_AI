# Real vs AI Image Classification

A deep learning project to classify images as real or AI-generated using a fine-tuned ResNet50 model.

## Author
**Ali Bubere**  
Machine Learning Engineer  
Email: [alibubere989@gmail.com]  
GitHub: [[Ali Bubere](https://github.com/Alibubere)]  
LinkedIn: [Mohammad Ali Bubere](https://www.linkedin.com/in/mohammad-ali-bubere-a6b830384/)

## Dataset

The project uses a dataset containing real and AI-generated images for binary classification.

**Dataset Link:** [Download Dataset](https://storage.googleapis.com/kaggle-data-sets/9067954/14215730/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260103T131302Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=42ea14e8367a6adc242eef295f67c845bb54b16e43f18a7e2a35e0cfb84fc7aeaa3d57a94a670e9d12ad6a5bf2c92c7a418456b1a7373dee626a8f08faee61ccce23fadb3bc9a9ccd4dfaf49d21c6dba8d55d3a81b05a8bcf91d79263416904c1bdd573331d8eb127c1eb102087608043e1eed8aff29732155519f4797c6b357a107eea6165a4a2b57130d0ff27c4a829d7b696de682926ad82e54b3c23229b1dc4ae7de1f5c3a4a9e3bd9013dfa24319368a14b6f817dee36c10409986f5b13853e6273f5ceaf41dce088eb765c9faa6d6d6cd88f4f87bdbf3126e55feaa1cfb6e07182aada0d8ba5c489bc8d563d9116f9238f33933870b49fdd0d6c331c50)

### Auto Download Script
```python
import requests
import zipfile
import os

def download_dataset():
    url = "https://storage.googleapis.com/kaggle-data-sets/9067954/14215730/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260103%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260103T131302Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=42ea14e8367a6adc242eef295f67c845bb54b16e43f18a7e2a35e0cfb84fc7aeaa3d57a94a670e9d12ad6a5bf2c92c7a418456b1a7373dee626a8f08faee61ccce23fadb3bc9a9ccd4dfaf49d21c6dba8d55d3a81b05a8bcf91d79263416904c1bdd573331d8eb127c1eb102087608043e1eed8aff29732155519f4797c6b357a107eea6165a4a2b57130d0ff27c4a829d7b696de682926ad82e54b3c23229b1dc4ae7de1f5c3a4a9e3bd9013dfa24319368a14b6f817dee36c10409986f5b13853e6273f5ceaf41dce088eb765c9faa6d6d6cd88f4f87bdbf3126e55feaa1cfb6e07182aada0d8ba5c489bc8d563d9116f9238f33933870b49fdd0d6c331c50"
    
    response = requests.get(url)
    with open("dataset.zip", "wb") as f:
        f.write(response.content)
    
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    
    os.remove("dataset.zip")
    print("Dataset downloaded and extracted to data/ folder")

if __name__ == "__main__":
    download_dataset()
```

## Project Structure
```
real_vs_ai/
├── configs/
│   └── config.yaml          # Training configuration
├── src/
│   ├── dataset.py          # Dataset class
│   ├── dataloader.py       # Data loading utilities
│   ├── features/
│   │   └── transform.py    # Image transformations
│   └── models/
│       ├── model.py        # Model architecture
│       ├── train.py        # Training loop
│       └── train_utils.py  # Training utilities
├── logs/                   # Training logs
├── checkpoints/           # Model checkpoints
├── main.py               # Main training script
└── README.md
```

## Features
- **ResNet50 Transfer Learning**: Fine-tuned pre-trained ResNet50 model
- **Binary Classification**: Real vs AI-generated images
- **Checkpoint Management**: Automatic saving/loading of best and latest models
- **Training Metrics**: Loss and accuracy tracking for both training and validation
- **Configurable Training**: YAML-based configuration system
- **Logging**: Comprehensive logging system

## Requirements
```
torch>=1.9.0
torchvision>=0.10.0
PyYAML>=5.4.1
Pillow>=8.3.0
requests>=2.25.1
```

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset using the auto-download script above
4. Update `configs/config.yaml` with your data path

## Usage
```bash
python main.py
```

## Configuration
Edit `configs/config.yaml` to customize:
- Data directory path
- Training parameters (batch size, learning rate, epochs)
- Checkpoint settings

## Model Architecture
- **Base Model**: ResNet50 (ImageNet pre-trained)
- **Fine-tuning**: Last layer (layer4) + fully connected layer
- **Output**: 2 classes (Real, AI-generated)
- **Optimizer**: SGD with momentum

## Training Features
- **Resume Training**: Automatic checkpoint loading
- **Best Model Tracking**: Saves best model based on validation loss
- **Accuracy Monitoring**: Real-time training and validation accuracy
- **Error Handling**: Robust error handling for corrupted batches

## Results
The model tracks both training and validation metrics:
- Training/Validation Loss
- Training/Validation Accuracy
- Best model selection based on validation performance