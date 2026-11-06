"""Experimental open-vocabulary detection for shops and roadside stalls."""

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


PROMPTS = ["shop", "tea stall", "roadside stall", "market stall", "shelter", "awning"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def scan_videos(input_dir: Path, output_path: Path, sample_seconds: float = 2.0, confidence: float = 0.15) -> pd.DataFrame:
    model = YOLO("yolov8s-world.pt")
    model.set_classes(PROMPTS)
    rows = []
    for video in sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS):
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        stride = max(1, round(fps * sample_seconds))
        for sampled_index, result in enumerate(model.predict(
            source=str(video), stream=True, vid_stride=stride, conf=confidence, verbose=False
        )):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for class_id, score in zip(boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
                rows.append({
                    "video": video.name,
                    "timestamp_seconds": round(sampled_index * sample_seconds, 1),
                    "detected_class": model.names[int(class_id)],
                    "confidence": round(float(score), 3),
                })
    detections = pd.DataFrame(rows, columns=["video", "timestamp_seconds", "detected_class", "confidence"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(output_path, index=False)
    return detections
