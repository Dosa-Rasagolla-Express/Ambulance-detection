from ultralytics import YOLO
import torch
from config import *

def train_model():
    
    print("Starting Fine-tuning Process\n")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    
    print(f"Loading pretrained model: {PRETRAINED_MODEL}")
    model = YOLO(str(PRETRAINED_MODEL))
    
    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    print()
    
    print("Starting training...\n")
    results = model.train(
        data=str(DATA_YAML),
        epochs=TRAINING_CONFIG['epochs'],
        imgsz=TRAINING_CONFIG['img_size'],
        batch=TRAINING_CONFIG['batch_size'],
        lr0=TRAINING_CONFIG['learning_rate'],
        freeze=TRAINING_CONFIG['freeze_layers'],
        patience=TRAINING_CONFIG['patience'],
        device=device,
        project=str(RUNS_DIR),
        name='ambulance_finetune',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        workers=TRAINING_CONFIG['workers'],
        
        hsv_h=AUGMENTATION_CONFIG['hsv_h'],
        hsv_s=AUGMENTATION_CONFIG['hsv_s'],
        hsv_v=AUGMENTATION_CONFIG['hsv_v'],
        degrees=AUGMENTATION_CONFIG['degrees'],
        translate=AUGMENTATION_CONFIG['translate'],
        scale=AUGMENTATION_CONFIG['scale'],
        shear=AUGMENTATION_CONFIG['shear'],
        perspective=AUGMENTATION_CONFIG['perspective'],
        flipud=AUGMENTATION_CONFIG['flipud'],
        fliplr=AUGMENTATION_CONFIG['fliplr'],
        mosaic=AUGMENTATION_CONFIG['mosaic'],
        mixup=AUGMENTATION_CONFIG['mixup'],
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved: {FINETUNED_MODEL}")
    print(f"Results saved: {RUNS_DIR / 'ambulance_finetune'}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    train_model()
