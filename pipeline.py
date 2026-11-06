"""Local, anonymous video analytics for naka footage.

The pipeline intentionally does not perform face recognition, age/gender inference,
or cross-day identity matching. Track IDs are valid only inside one video.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
VEHICLE_CLASSES = {"bicycle", "motorcycle", "car", "bus", "truck"}
GENDER_LABELS = ("Male", "Female")


class GenderClassifier:
    """Face-based apparent-gender estimate for aggregate, track-level counts."""

    def __init__(self, gender_model: Path, gender_proto: Path, face_model: Path,
                 threshold: float = 0.60):
        # Some recent OpenCV builds omit readNetFromCaffe but retain the generic loader.
        self.gender_net = cv2.dnn.readNet(str(gender_model), str(gender_proto))
        self.face_detector = cv2.FaceDetectorYN.create(
            str(face_model), "", (320, 320), 0.6, 0.3, 5000
        )
        self.threshold = threshold

    def classify(self, person_crop: Any) -> tuple[str, float]:
        if person_crop is None or person_crop.size == 0:
            return "Unknown", 0.0
        height, width = person_crop.shape[:2]
        if width < 32 or height < 32:
            return "Unknown", 0.0
        self.face_detector.setInputSize((width, height))
        detected = self.face_detector.detect(person_crop)
        faces = detected[1] if isinstance(detected, tuple) else None
        if faces is None or len(faces) == 0:
            return "Unknown", 0.0
        face = max(faces, key=lambda row: float(row[2] * row[3]))
        x, y, w, h = [int(value) for value in face[:4]]
        x, y = max(0, x), max(0, y)
        face_crop = person_crop[y:min(height, y + h), x:min(width, x + w)]
        if face_crop.size == 0:
            return "Unknown", 0.0
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        self.gender_net.setInput(blob)
        scores = self.gender_net.forward()[0]
        class_index = int(scores.argmax())
        confidence = float(scores[class_index])
        return (GENDER_LABELS[class_index], confidence) if confidence >= self.threshold else ("Unknown", confidence)


def as_names(names: Any) -> dict[int, str]:
    return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def analyze_video(video_path: Path, output_dir: Path, model: YOLO, sample_seconds: float,
                  tracker: str, confidence: float, save_annotated: bool,
                  reliable_tracking: bool = False, image_size: int = 512,
                  frame_stride: int = 4, gender_classifier: GenderClassifier | None = None) -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    frame_stride = 1 if reliable_tracking else max(1, frame_stride)
    sample_every = max(1, round(fps * sample_seconds / frame_stride))
    rows: list[dict] = []
    observed_track_ids: set[int] = set()
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    track_hits: dict[int, int] = {}
    gender_by_track: dict[int, str] = {}
    gender_by_track: dict[int, str] = {}
    object_totals: dict[str, int] = {}
    writer = None
    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    annotated_raw_path = output_dir / f"{video_path.stem}_annotated_raw.mp4"
    names = as_names(model.names)
    analysis_id = uuid.uuid4().hex
    # IDs belong to one video/run; never carry tracker state into another clip.
    for active_tracker in getattr(getattr(model, "predictor", None), "trackers", []):
        active_tracker.reset()

    # Track continuously; sampling controls exported observations only.
    result_stream = enumerate(model.track(
        source=str(video_path), stream=True, persist=True, tracker=tracker,
        conf=confidence, imgsz=image_size, max_det=1000, vid_stride=frame_stride, verbose=False,
    ))

    for frame_index, result in result_stream:
        if save_annotated and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_raw_path), fourcc, fps, (width, height))
        if save_annotated and writer is not None:
            writer.write(result.plot())

        boxes = result.boxes
        people = 0
        vehicles = 0
        counts: dict[str, int] = {}
        frame_track_ids: list[int] = []

        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.int().cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(classes)
            for class_id, score, track_id in zip(classes, scores, ids):
                label = names.get(class_id, str(class_id))
                counts[label] = counts.get(label, 0) + 1
                object_totals[label] = object_totals.get(label, 0) + 1
                if label == "person":
                    people += 1
                if label in VEHICLE_CLASSES:
                    vehicles += 1
                if track_id >= 0 and label == "person":
                    observed_track_ids.add(track_id)
                    frame_track_ids.append(track_id)
                    first_seen.setdefault(track_id, frame_index)
                    last_seen[track_id] = frame_index
                    track_hits[track_id] = track_hits.get(track_id, 0) + 1

        if frame_index % sample_every != 0:
            continue
        timestamp = frame_index * frame_stride / fps
        gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
        if gender_classifier is not None and boxes is not None and len(boxes) > 0:
            person_indexes = [index for index, class_id in enumerate(classes)
                              if names.get(class_id, str(class_id)) == "person"]
            for index in person_indexes:
                track_id = ids[index]
                if track_id not in gender_by_track:
                    x1, y1, x2, y2 = boxes.xyxy[index].int().cpu().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    label, _ = gender_classifier.classify(result.orig_img[y1:y2, x1:x2])
                    if track_id >= 0:
                        gender_by_track[track_id] = label
                    else:
                        gender_by_track[index * -1 - 1] = label
                label = gender_by_track.get(track_id if track_id >= 0 else index * -1 - 1, "Unknown")
                gender_counts[label] += 1
        row = {
            "video": video_path.name,
            "day": video_path.stem,
            "timestamp_seconds": round(timestamp, 2),
            "timestamp": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
            "people_count": people,
            "male_count": gender_counts["Male"],
            "female_count": gender_counts["Female"],
            "unknown_gender_count": gender_counts["Unknown"],
            "vehicle_count": vehicles,
            "anonymous_track_count": len(set(frame_track_ids)),
            "confidence": round(float(sum(scores) / len(scores)), 3) if boxes is not None and len(boxes) else 0.0,
            "object_counts": json.dumps(counts, sort_keys=True),
        }
        rows.append(row)

    if writer is not None:
        writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg = get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if ffmpeg and annotated_raw_path.exists():
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", str(annotated_raw_path), "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(annotated_path)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                annotated_raw_path.unlink(missing_ok=True)
            except subprocess.CalledProcessError:
                # Never expose the mp4v intermediate as a browser video.
                annotated_raw_path.unlink(missing_ok=True)
                raise RuntimeError("Could not convert the annotated video to browser-compatible H.264 MP4.")
        elif annotated_raw_path.exists():
            annotated_raw_path.replace(annotated_path)

    observations = pd.DataFrame(rows)
    video_output = output_dir / f"{video_path.stem}_observations.csv"
    observations.to_csv(video_output, index=False)

    minimum_hits = max(2, round(fps * 0.1))
    confirmed_track_ids = {track_id for track_id, hits in track_hits.items() if hits >= minimum_hits}
    peak_people = int(observations.people_count.max()) if not observations.empty else 0
    crowd_counts = observations.people_count.astype(int).tolist() if not observations.empty else []
    # Net crowd changes are separate from the person-track total; they cannot
    # establish how many distinct people arrived or departed.
    arrivals = sum(max(0, current - previous) for previous, current in zip(crowd_counts, crowd_counts[1:]))
    departures = sum(max(0, previous - current) for previous, current in zip(crowd_counts, crowd_counts[1:]))
    unique_people_estimate = len(observed_track_ids)
    gender_totals = Counter(gender_by_track.values())

    summary = {
        "video": video_path.name,
        "day": video_path.stem,
        "duration_seconds": round(frame_count / fps, 2) if fps else 0,
        "sample_seconds": sample_seconds,
        "average_people": round(float(observations.people_count.mean()), 2) if not observations.empty else 0,
        "peak_people": peak_people,
        "male_people_estimated": int(gender_totals.get("Male", 0)),
        "female_people_estimated": int(gender_totals.get("Female", 0)),
        "unknown_gender_estimated": int(gender_totals.get("Unknown", 0)),
        "average_vehicles": round(float(observations.vehicle_count.mean()), 2) if not observations.empty else 0,
        "total_people_estimated": unique_people_estimate,
        "count_method": "person_track_ids_v3",
        "anonymous_tracks_seen_in_this_video": unique_people_estimate,
        "confirmed_person_tracks": len(confirmed_track_ids),
        "raw_person_track_ids": len(observed_track_ids),
        "arrivals_estimated": arrivals,
        "departures_estimated": departures,
        "reliable_tracking": True,
        "analysis_id": analysis_id,
        "person_track_ids": json.dumps(sorted(observed_track_ids)),
        "object_sample_totals": object_totals,
        "note": "Total people counts distinct tracking IDs labelled person, once per video. Vehicles and other classes are excluded. ID switches may count one person more than once. Arrivals/departures are net crowd changes, not verified crossings.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze naka videos locally with YOLO tracking.")
    parser.add_argument("--input", type=Path, default=Path("input_videos"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--reliable-tracking", action="store_true", help="Track every frame for better within-video unique estimates.")
    parser.add_argument("--gender-model", type=Path, default=Path("models/gender_net.caffemodel"))
    parser.add_argument("--gender-proto", type=Path, default=Path("models/gender_deploy.prototxt"))
    parser.add_argument("--face-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--gender-confidence", type=float, default=0.60)
    parser.add_argument("--disable-gender", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in args.input.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No videos found in {args.input.resolve()}")

    model = YOLO(args.model)
    gender_classifier = None
    if not args.disable_gender:
        try:
            gender_classifier = GenderClassifier(args.gender_model, args.gender_proto,
                                                  args.face_model, args.gender_confidence)
        except (OSError, cv2.error) as error:
            print(f"Gender model unavailable; using Unknown: {error}")
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for video in videos:
        print(f"Analyzing {video.name} ...")
        rows, summary = analyze_video(video, args.output, model, args.sample_seconds,
                                      args.tracker, args.confidence, args.save_annotated,
                                      args.reliable_tracking, args.imgsz, args.frame_stride,
                                      gender_classifier)
        all_rows.extend(rows)
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(args.output / "observations.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "daily_summary.csv", index=False)
    (args.output / "daily_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {args.output.resolve()}")


if __name__ == "__main__":
    main()
