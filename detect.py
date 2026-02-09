from ultralytics import YOLO
import cv2
from pathlib import Path
from datetime import datetime
from config import *

def detect_ambulances(
    source='vedios',
    save_video=True,
    save_txt=True,
    conf_threshold=None,
    show=False
):
    
    print("Ambulance Detection System\n")
    print("=" * 60)
    
    if not FINETUNED_MODEL.exists():
        print(f"ERROR: Trained model not found at {FINETUNED_MODEL}")
        print("Please train the model first: python train.py")
        return
    
    print(f"Loading model: {FINETUNED_MODEL}")
    model = YOLO(str(FINETUNED_MODEL))
    
    if conf_threshold is None:
        conf_threshold = DETECTION_CONFIG['confidence_threshold']
    
    print("\nDetection Configuration:")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {DETECTION_CONFIG['iou_threshold']}")
    print(f"   Source: {source}")
    print("=" * 60 + "\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f"detect_{timestamp}"
    
    print("Starting detection...\n")
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=DETECTION_CONFIG['iou_threshold'],
        max_det=DETECTION_CONFIG['max_det'],
        save=save_video,
        save_txt=save_txt,
        save_conf=OUTPUT_CONFIG['save_conf'],
        save_crop=OUTPUT_CONFIG['save_crop'],
        show_labels=OUTPUT_CONFIG['show_labels'],
        show_conf=OUTPUT_CONFIG['show_conf'],
        line_width=OUTPUT_CONFIG['line_width'],
        project=str(OUTPUT_DIR / 'detections'),
        name=output_name,
        verbose=True,
        stream=True,
        show=show
    )
    
    total_detections = 0
    total_frames = 0
    
    for result in results:
        total_frames += 1
        if result.boxes is not None:
            total_detections += len(result.boxes)
    
    print("\n" + "=" * 60)
    print("Detection Complete!")
    print("=" * 60)
    print(f"   Total frames processed: {total_frames}")
    print(f"   Total ambulances detected: {total_detections}")
    if total_frames > 0:
        print(f"   Average detections per frame: {total_detections/total_frames:.2f}")
    print(f"\n   Results saved to: {OUTPUT_DIR / 'detections' / output_name}")
    print("=" * 60)
    
    return results

def detect_single_video(video_path, conf_threshold=0.25):
    return detect_ambulances(
        source=video_path,
        conf_threshold=conf_threshold
    )

def detect_all_videos(conf_threshold=0.25):
    return detect_ambulances(
        source='vedios',
        conf_threshold=conf_threshold
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
        detect_single_video(video_path, conf)
    else:
        detect_all_videos(conf_threshold=0.25)
