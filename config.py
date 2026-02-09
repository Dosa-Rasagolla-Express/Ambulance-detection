import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

VIDEOS_DIR = PROJECT_ROOT / 'vedios'
FRAMES_DIR = PROJECT_ROOT / 'data' / 'frames'
DATASET_DIR = PROJECT_ROOT / 'dataset'
MODELS_DIR = PROJECT_ROOT / 'models'
RUNS_DIR = PROJECT_ROOT / 'runs'
OUTPUT_DIR = PROJECT_ROOT / 'output'

PRETRAINED_MODEL = MODELS_DIR / 'traffic_yolov8.pt'
FINETUNED_MODEL = RUNS_DIR / 'ambulance_finetune' / 'weights' / 'best.pt'

DATA_YAML = DATASET_DIR / 'data.yaml'

TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'img_size': 640,
    'learning_rate': 0.001,
    'freeze_layers': 10,
    'patience': 15,
    'workers': 4,
    'device': 0,
}

DETECTION_CONFIG = {
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'ambulance_class_id': 0,
    'max_det': 300,
}

CLASS_NAMES = {
    0: 'ambulance',
}

FRAME_EXTRACTION_FPS = 2

AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
}

VALIDATION_CONFIG = {
    'save_json': True,
    'save_hybrid': False,
    'conf': 0.001,
    'iou': 0.6,
    'max_det': 300,
    'half': False,
    'plots': True,
}

OUTPUT_CONFIG = {
    'save_txt': True,
    'save_conf': True,
    'save_crop': False,
    'show_labels': True,
    'show_conf': True,
    'line_width': 2,
}

COLORS = {
    'ambulance': (0, 255, 0),
    'bbox': (0, 255, 255),
    'text': (255, 255, 255),
}

LOG_CONFIG = {
    'log_dir': OUTPUT_DIR / 'logs',
    'tensorboard': True,
    'wandb': False,
}

def setup_directories():
    directories = [
        VIDEOS_DIR,
        FRAMES_DIR,
        DATASET_DIR,
        MODELS_DIR,
        RUNS_DIR,
        OUTPUT_DIR,
        OUTPUT_DIR / 'detections',
        OUTPUT_DIR / 'logs',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("Ambulance Detection System - Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  Videos: {VIDEOS_DIR}")
    print(f"  Frames: {FRAMES_DIR}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Runs: {RUNS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nModels:")
    print(f"  Pretrained: {PRETRAINED_MODEL}")
    print(f"  Finetuned: {FINETUNED_MODEL}")
    print(f"\nDataset Config: {DATA_YAML}")
    print(f"\nTraining Config:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nDetection Config:")
    for key, value in DETECTION_CONFIG.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 60)
    
    setup_directories()
    print("\nAll directories created/verified!")
