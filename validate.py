from ultralytics import YOLO
from config import *
import torch

def validate_model():
    
    print("Validating Ambulance Detection Model\n")
    print("=" * 60)
    
    if not FINETUNED_MODEL.exists():
        print(f"ERROR: Trained model not found at {FINETUNED_MODEL}")
        print("Please train the model first: python train.py")
        return
    
    print(f"Loading model: {FINETUNED_MODEL}")
    model = YOLO(str(FINETUNED_MODEL))
    
    print(f"Running validation on: {DATA_YAML}\n")
    metrics = model.val(
        data=str(DATA_YAML),
        split='val',
        imgsz=TRAINING_CONFIG['img_size'],
        batch=TRAINING_CONFIG['batch_size'],
        conf=VALIDATION_CONFIG['conf'],
        iou=VALIDATION_CONFIG['iou'],
        max_det=VALIDATION_CONFIG['max_det'],
        plots=VALIDATION_CONFIG['plots'],
        save_json=VALIDATION_CONFIG['save_json'],
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)
    print(f"   Precision:    {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)")
    print(f"   Recall:       {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)")
    print(f"   mAP@0.5:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
    print("=" * 60)
    
    print("\nInterpretation:")
    print("=" * 60)
    
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    
    if precision >= 0.7:
        print("  Precision: EXCELLENT - Low false positive rate")
    elif precision >= 0.5:
        print("  Precision: GOOD - Acceptable false positive rate")
    else:
        print("  Precision: NEEDS IMPROVEMENT - High false positive rate")
    
    if recall >= 0.6:
        print("  Recall: EXCELLENT - Detects most ambulances")
    elif recall >= 0.4:
        print("  Recall: GOOD - Detects many ambulances")
    else:
        print("  Recall: NEEDS IMPROVEMENT - Misses many ambulances")
    
    if map50 >= 0.6:
        print("  mAP@50: EXCELLENT - High quality detections")
    elif map50 >= 0.4:
        print("  mAP@50: GOOD - Decent quality detections")
    else:
        print("  mAP@50: NEEDS IMPROVEMENT - Low quality detections")
    
    print("=" * 60)
    
    print("\nPer-Class Metrics:")
    print("=" * 60)
    for i, class_name in CLASS_NAMES.items():
        if i < len(metrics.box.ap_class_index):
            p = metrics.box.p[i] if i < len(metrics.box.p) else 0
            r = metrics.box.r[i] if i < len(metrics.box.r) else 0
            print(f"   {class_name:12s} - Precision: {p:.4f}, Recall: {r:.4f}")
    print("=" * 60)
    
    if hasattr(metrics, 'speed'):
        print("\nSpeed Metrics:")
        print("=" * 60)
        print(f"   Preprocess:  {metrics.speed['preprocess']:.1f}ms")
        print(f"   Inference:   {metrics.speed['inference']:.1f}ms")
        print(f"   Postprocess: {metrics.speed['postprocess']:.1f}ms")
        total_time = sum(metrics.speed.values())
        fps = 1000 / total_time if total_time > 0 else 0
        print(f"   Total:       {total_time:.1f}ms ({fps:.1f} FPS)")
        print("=" * 60)
    
    print("\nValidation complete!")
    print(f"Results saved to: {RUNS_DIR / 'ambulance_finetune'}")
    
    return metrics

if __name__ == "__main__":
    validate_model()
