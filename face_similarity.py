"""Local comparison of consenting participants in two short videos.

Face features and crops are held in memory only for the comparison.
"""

from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import cv2
import numpy as np


MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_models(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in MODEL_URLS.items():
        path = model_dir / name
        if not path.exists():
            partial = path.with_suffix(".download")
            try:
                with urlopen(url, timeout=45) as response, partial.open("wb") as destination:
                    copyfileobj(response, destination)
                if partial.stat().st_size < 100_000:
                    raise ValueError(f"Incomplete model download: {name}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        paths.append(path)
    return paths[0], paths[1]


def video_groups(video: Path, detector, recognizer, interval_seconds: float = 1.0,
                 max_samples: int = 60, group_threshold: float = 0.48) -> tuple[list[dict], int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video.name}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps * interval_seconds))
    groups: list[dict] = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                sampled += 1
                height, width = frame.shape[:2]
                detector.setInputSize((width, height))
                _, faces = detector.detect(frame)
                used_groups: set[int] = set()
                if faces is not None:
                    faces = sorted(faces, key=lambda face: float(face[14]), reverse=True)
                for face in faces if faces is not None else []:
                    if min(float(face[2]), float(face[3])) < 40:
                        continue
                    aligned = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned).ravel().astype(np.float32)
                    norm = np.linalg.norm(feature)
                    if norm <= 0:
                        continue
                    feature /= norm
                    x, y, w, h = (int(value) for value in face[:4])
                    x0, y0 = max(0, x), max(0, y)
                    crop = frame[y0:min(height, y + h), x0:min(width, x + w)]
                    preview = cv2.imencode(".jpg", crop)[1].tobytes() if crop.size else b""
                    timestamp = round(frame_index / fps, 1)
                    candidates = [
                        (max(float(feature @ prior) for prior in group["features"]), index)
                        for index, group in enumerate(groups) if index not in used_groups
                    ]
                    best_score, best_index = max(candidates, default=(-1.0, -1))
                    if best_score < group_threshold:
                        groups.append({"features": [feature], "observations": 1,
                                       "preview": preview, "timestamp": timestamp})
                        used_groups.add(len(groups) - 1)
                    else:
                        group = groups[best_index]
                        group["observations"] += 1
                        if len(group["features"]) < 8:
                            group["features"].append(feature)
                        used_groups.add(best_index)
            frame_index += 1
    finally:
        capture.release()
    return groups, sampled


def group_score(first: dict, second: dict) -> float:
    similarities = np.asarray(first["features"]) @ np.asarray(second["features"]).T
    strongest = np.sort(similarities.ravel())[-min(3, similarities.size):]
    return float(np.median(strongest))


def compare_videos(first: Path, second: Path, model_dir: Path) -> dict:
    detector_path, recognizer_path = ensure_models(model_dir)
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    first_groups, first_frames = video_groups(first, detector, recognizer)
    second_groups, second_frames = video_groups(second, detector, recognizer)
    if not first_groups or not second_groups:
        raise ValueError("No clear face frames found in one or both videos. Use well-lit footage with faces at least 40 pixels wide.")
    scores = np.array([[group_score(a, b) for b in second_groups] for a in first_groups])
    ranked_pairs = sorted(
        ((float(scores[row, column]), row, column)
         for row in range(len(first_groups)) for column in range(len(second_groups))),
        reverse=True,
    )
    selected_pairs = []
    used_first: set[int] = set()
    used_second: set[int] = set()
    for _, row, column in ranked_pairs:
        if row not in used_first and column not in used_second:
            selected_pairs.append((row, column))
            used_first.add(row)
            used_second.add(column)
    candidates = [
        {"Video 1 face": f"Face {row + 1}", "Video 2 face": f"Face {column + 1}",
         "Similarity": round(float(scores[row, column]), 3),
         "Observations 1": first_groups[row]["observations"],
         "Observations 2": second_groups[column]["observations"]}
        for row, column in selected_pairs
    ]
    return {
        "first_groups": len(first_groups), "second_groups": len(second_groups),
        "sampled_frames_first": first_frames, "sampled_frames_second": second_frames,
        "candidates": sorted(candidates, key=lambda item: item["Similarity"], reverse=True),
        "scores": scores.round(3).tolist(),
        "previews_first": [{"image": group["preview"], "timestamp": group["timestamp"]} for group in first_groups],
        "previews_second": [{"image": group["preview"], "timestamp": group["timestamp"]} for group in second_groups],
    }
