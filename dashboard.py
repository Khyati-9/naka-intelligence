"""Simple Streamlit dashboard for pipeline outputs."""

from pathlib import Path
import json
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).parent
INPUT = ROOT / "input_videos"
OUTPUT = ROOT / "outputs"
LOG_PATH = OUTPUT / "analysis.log"

def run_analysis(command: list[str]) -> tuple[int, str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    LOG_PATH.write_text(output, encoding="utf-8")
    return completed.returncode, output

st.set_page_config(page_title="Naka Intelligence", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetric"] { background: #202938; border: 1px solid #344054; border-radius: 12px; padding: 12px; }
    [data-testid="stMetricLabel"] { color: #b8c2d1 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    page = st.radio("View", ["Dashboard", "Face comparison"], label_visibility="collapsed")

if page == "Face comparison":
    st.title("Face comparison")
    st.caption("Upload two short clips of the same consenting group from different angles. Faces are grouped within each clip and anonymous similarity candidates are shown. Clips and face features are not kept after comparison.")
    face_cols = st.columns(2)
    first_face_video = face_cols[0].file_uploader("First video", type=["mp4", "mov", "avi", "mkv"], key="face_first")
    second_face_video = face_cols[1].file_uploader("Second video", type=["mp4", "mov", "avi", "mkv"], key="face_second")
    if st.button("Compare faces", disabled=not (first_face_video and second_face_video)):
        import face_similarity
        importlib.reload(face_similarity)
        try:
            with st.spinner("Sampling faces and comparing the two clips..."):
                with tempfile.TemporaryDirectory(prefix="naka_self_face_") as temporary:
                    first_path = Path(temporary) / f"first{Path(first_face_video.name).suffix.lower()}"
                    second_path = Path(temporary) / f"second{Path(second_face_video.name).suffix.lower()}"
                    first_path.write_bytes(first_face_video.getvalue())
                    second_path.write_bytes(second_face_video.getvalue())
                    face_result = face_similarity.compare_videos(first_path, second_path, ROOT / "models")
            st.write(f"Detected face groups: **{face_result['first_groups']}** in video 1 and **{face_result['second_groups']}** in video 2")
            st.caption(f"Sampled {face_result['sampled_frames_first']} and {face_result['sampled_frames_second']} frames. Higher scores mean more similar faces. Grouping and cross-angle candidates can be wrong; verify against the consented participants.")
            for video_number, previews in enumerate([face_result.get("previews_first", []), face_result.get("previews_second", [])], start=1):
                if not previews:
                    continue
                st.write(f"Video {video_number} face labels")
                preview_columns = st.columns(min(5, len(previews)))
                for face_index, preview in enumerate(previews):
                    with preview_columns[face_index % len(preview_columns)]:
                        st.caption(f"Face {face_index + 1} · {preview['timestamp']:.1f}s")
                        if preview["image"]:
                            st.image(preview["image"], width=120)
            st.dataframe(pd.DataFrame(face_result["candidates"]), use_container_width=True, hide_index=True)
            with st.expander("All group-to-group similarity scores"):
                st.dataframe(pd.DataFrame(face_result["scores"], index=[f"Video 1 · Face {i + 1}" for i in range(face_result["first_groups"])], columns=[f"Video 2 · Face {i + 1}" for i in range(face_result["second_groups"])]), use_container_width=True)
        except (ValueError, OSError, cv2.error) as error:
            st.error(str(error))
    st.stop()

st.title("Naka Intelligence")
st.caption("Anonymous, approximate video analytics using local YOLO + ByteTrack. No identities or face embeddings are stored.")

with st.sidebar:
    st.header("Run analysis")
    uploaded_videos = st.file_uploader(
        "Step 1 · Upload videos for analysis",
        type=["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg", "m4v"],
        accept_multiple_files=True,
        help="Add one or more naka recordings directly to input_videos/.",
    )
    if "auto_saved_uploads" not in st.session_state:
        st.session_state["auto_saved_uploads"] = set()
    if uploaded_videos:
        INPUT.mkdir(parents=True, exist_ok=True)
        for uploaded_video in uploaded_videos:
            safe_name = Path(uploaded_video.name).name
            upload_key = f"{safe_name}:{uploaded_video.size}"
            if upload_key not in st.session_state["auto_saved_uploads"]:
                (INPUT / safe_name).write_bytes(uploaded_video.getvalue())
                st.session_state["auto_saved_uploads"].add(upload_key)
        st.success("Uploaded video(s) are ready in input_videos/")
    if st.button("Step 1 · Add uploaded videos", disabled=not uploaded_videos):
        INPUT.mkdir(parents=True, exist_ok=True)
        saved_names = []
        for uploaded_video in uploaded_videos:
            safe_name = Path(uploaded_video.name).name
            (INPUT / safe_name).write_bytes(uploaded_video.getvalue())
            saved_names.append(safe_name)
        st.success(f"Added {len(saved_names)} video(s) to input_videos/")
        st.rerun()
    sample_seconds = st.slider(
        "Sample every (seconds)", 1, 10, 2,
        help="The pipeline processes one frame every N seconds. Lower values give a smoother trend but take longer. At 2 seconds, a 30-minute video uses about 900 sampled frames.",
    )
    st.caption(f"Current setting: one frame every {sample_seconds} seconds")
    model = st.text_input("YOLO model", "yolo26n.pt", help="Nano is faster; use a larger model only when small/distant people are missed.")
    confidence = st.slider("Detection confidence", 0.10, 0.60, 0.20, 0.05,
                           help="Lower values find more people but can add false detections.")
    reliable_tracking = False
    st.caption("Fast mode processes every 4th frame and tracks those frames. This keeps total people estimates responsive; enable full-frame tracking in code only for validation runs.")
    save_annotated = st.checkbox(
        "Save annotated videos", value=False,
        help="Creates an output video with boxes, labels, confidence scores, and tracking IDs. This is slower because it processes every frame; leave off for faster counts.",
    )
    st.caption("After adding files, click Analyze videos to process them.")
    if st.button("Step 2 · Analyze videos"):
        command = [sys.executable, str(ROOT / "pipeline.py"), "--input", str(INPUT),
                   "--output", str(OUTPUT), "--model", model,
                   "--sample-seconds", str(sample_seconds), "--confidence", str(confidence),
                   "--frame-stride", "4"]
        command.extend(["--imgsz", "512"])
        if reliable_tracking:
            command.append("--reliable-tracking")
        if save_annotated:
            command.append("--save-annotated")
        with st.status("Calculating people and male/female estimates…", expanded=True) as status:
            try:
                return_code, analysis_output = run_analysis(command)
                if return_code == 0:
                    status.update(label="Analysis complete", state="complete")
                    st.success("Analysis complete. Results are shown below.")
                    st.rerun()
                status.update(label="Analysis failed", state="error")
                st.error("The analysis process failed. See the details below.")
                st.code(analysis_output[-6000:] or "No diagnostic output was produced.")
            except subprocess.TimeoutExpired:
                status.update(label="Analysis timed out", state="error")
                st.error("Analysis exceeded the 30-minute limit. Reduce video length or increase the sample interval.")

summary_path = OUTPUT / "daily_summary.csv"
observations_path = OUTPUT / "observations.csv"
if not summary_path.exists() or not observations_path.exists():
    st.info("Put videos in input_videos/ and click Analyze videos.")
    st.stop()

summary = pd.read_csv(summary_path)
observations = pd.read_csv(observations_path)
def analysis_is_complete(record: pd.Series) -> bool:
    """Detect old/partial output so the UI does not present it as final."""
    video_observations = observations[observations["day"] == record["day"]]
    sample = max(float(record.get("sample_seconds", sample_seconds)), 0.1)
    expected = max(1, int(float(record.get("duration_seconds", 0)) // sample) + 1)
    return len(video_observations) >= expected

all_analysis_complete = all(analysis_is_complete(record) for _, record in summary.iterrows())
total_people_detections = int(observations["people_count"].sum())
st.subheader("Overall people summary")
cols = st.columns(7)
cols[0].metric("Videos", len(summary))
overall_sampled_average = float(observations["people_count"].mean()) if not observations.empty else 0.0
cols[1].metric("Avg people / sample", f"{overall_sampled_average:.1f}" if all_analysis_complete else "Calculating…")
cols[2].metric("Highest observed crowd", int(summary.peak_people.max()) if all_analysis_complete else "Calculating…")
cols[3].metric("Average vehicles", f"{summary.average_vehicles.mean():.1f}" if all_analysis_complete else "Calculating…")
cols[4].metric("Total people detections", total_people_detections if all_analysis_complete else "Calculating…")
male_total = int(pd.to_numeric(summary.get("male_people_estimated", 0), errors="coerce").fillna(0).sum())
female_total = int(pd.to_numeric(summary.get("female_people_estimated", 0), errors="coerce").fillna(0).sum())
cols[5].metric("Male estimate", male_total if all_analysis_complete else "Calculating…")
cols[6].metric("Female estimate", female_total if all_analysis_complete else "Calculating…")
st.caption("Average people is calculated across all sampled observations from all videos. Total people detections is the sum across sampled observations, not the number of unique people. Metrics show Calculating… while any video is incomplete.")
st.caption("Male/female values are model estimates from visible faces, counted once per temporary track where a face is detected. Low-confidence or missing-face cases are Unknown.")

available_days = summary["day"].tolist()
requested_video = st.query_params.get("video")
session_video = st.session_state.get("selected_day")
selected_video = (
    requested_video if requested_video in available_days
    else session_video if session_video in available_days
    else available_days[0]
)
selected_day = selected_video
selected_summary = summary[summary["day"] == selected_day].iloc[0]
st.session_state["selected_day"] = selected_day

st.caption("Click View annotated video to open the player directly below that row.")
open_annotation_day = st.session_state.get("open_annotation_day")
infrastructure_counts: dict[str, dict[str, int]] = {}
infrastructure_path = OUTPUT / "infrastructure_detections.csv"
if infrastructure_path.exists():
    infrastructure_data = pd.read_csv(infrastructure_path)
    for (video, detected_class), count in infrastructure_data.groupby(["video", "detected_class"]).size().items():
        infrastructure_counts.setdefault(video, {})[detected_class] = int(count)

with st.container(border=True):
    table_widths = [3, 1, 1, 1.4, 1, 1, 1.3, 1.3, 1, 1.6]
    header = st.columns(table_widths)
    for column, label in zip(header, ["File name", "Duration (min)", "Sample (sec)", "Total people", "Arrivals", "Departures", "Avg vehicles", "Shop/stall", "Shelter/awning", "Annotation"]):
        column.caption(label)
    st.divider()

    for index, (_, record) in enumerate(summary.iterrows()):
        day = record["day"]
        annotated_path = OUTPUT / f"{day}_annotated.mp4"
        row = st.columns(table_widths)
        infrastructure_for_video = infrastructure_counts.get(record["video"], {})
        shop_stall_count = sum(infrastructure_for_video.get(label, 0) for label in ["shop", "tea stall", "roadside stall", "market stall"])
        shelter_awning_count = infrastructure_for_video.get("shelter", 0) + infrastructure_for_video.get("awning", 0)
        complete = analysis_is_complete(record)
        infrastructure_ready = infrastructure_path.exists()
        values = [
            record["video"], f"{record['duration_seconds'] / 60:.2f}", f"{record['sample_seconds']:.0f}",
            str(int(record["raw_person_track_ids"])) if complete and pd.notna(record.get("raw_person_track_ids")) else "Reanalyse",
            str(int(record["arrivals_estimated"])) if complete and pd.notna(record.get("arrivals_estimated")) else "Reanalyse",
            str(int(record["departures_estimated"])) if complete and pd.notna(record.get("departures_estimated")) else "Reanalyse",
            f"{record['average_vehicles']:.1f}" if complete else "Calculating…",
            str(shop_stall_count) if infrastructure_ready else "Calculating…",
            str(shelter_awning_count) if infrastructure_ready else "Calculating…",
        ]
        for column, value in zip(row[:-1], values):
            column.write(value)

        if annotated_path.exists():
            button_label = "Hide annotated video" if open_annotation_day == day else "View annotated video"
            if row[-1].button(button_label, key=f"view_annotation_{day}", type="tertiary"):
                open_annotation_day = None if open_annotation_day == day else day
                st.session_state["open_annotation_day"] = open_annotation_day
                st.session_state["selected_day"] = day
                st.rerun()
        else:
            row[-1].caption("Not available")

        if open_annotation_day == day and annotated_path.exists():
            st.caption(f"Annotated video · {record['video']}")
            video_col, _ = st.columns([1, 2])
            with video_col:
                st.video(str(annotated_path))
            st.download_button("Download annotated video", data=annotated_path.read_bytes(), file_name=annotated_path.name, mime="video/mp4", key=f"download_{day}")
            st.markdown("**Unique insight from this video**")
            if not complete:
                st.info("Calculating… this video has not produced all expected sampled observations yet.")
            else:
                video_observations = observations[observations["day"] == day].copy()
                busiest = video_observations.loc[video_observations["people_count"].idxmax()]
                infrastructure_labels = [
                    label for label, count in infrastructure_for_video.items() if count > 0
                ]
                insight = (
                    f"The busiest sampled moment was **{busiest['timestamp']}**, with "
                    f"**{int(busiest['people_count'])} visible people** and "
                    f"**{int(busiest['vehicle_count'])} vehicles**. "
                    f"The video average was **{record['average_people']:.1f} people**."
                )
                if infrastructure_labels:
                    insight += f" Infrastructure candidates in this video: **{', '.join(infrastructure_labels)}**."
                else:
                    insight += " No completed shop/stall infrastructure result is available for this video yet."
                st.info(insight)

        if index < len(summary) - 1:
            st.divider()

infrastructure_path = OUTPUT / "infrastructure_detections.csv"
st.caption("Shop and stall scanning runs separately so the normal people and vehicle scan stays fast. Existing infrastructure results are included in the table when available.")
if st.button("Run shop/stall scan later"):
    import infrastructure
    try:
        with st.spinner("Scanning sampled frames for shops and stalls..."):
            infrastructure_result = infrastructure.scan_videos(INPUT, OUTPUT / "infrastructure_detections.csv", sample_seconds=float(sample_seconds))
        st.session_state["infrastructure_result"] = infrastructure_result
        st.rerun()
    except Exception as error:
        st.error(f"Infrastructure rescan failed: {error}")

infrastructure_result = st.session_state.get("infrastructure_result")
if infrastructure_result is None and infrastructure_path.exists():
    infrastructure_result = pd.read_csv(infrastructure_path)
    st.session_state["infrastructure_result"] = infrastructure_result
st.subheader("Video-wise analysis")
selected = observations[observations["day"] == selected_day].copy()

video_cols = st.columns(7)
video_cols[0].metric("Video", str(selected_summary["video"]))
video_cols[1].metric("Average people", f"{selected_summary['average_people']:.1f}")
video_cols[2].metric("Peak people", int(selected_summary["peak_people"]))
video_cols[3].metric("Average vehicles", f"{selected_summary['average_vehicles']:.1f}")
video_cols[4].metric("Anonymous tracks", int(selected_summary["anonymous_tracks_seen_in_this_video"]))
video_cols[5].metric("Male estimate", int(selected_summary.get("male_people_estimated", 0)))
video_cols[6].metric("Female estimate", int(selected_summary.get("female_people_estimated", 0)))

st.caption("Anonymous tracks are temporary IDs inside this video, not confirmed people and not cross-day identities.")

allowed_objects = {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "shop", "tea stall", "roadside stall", "market stall", "shelter", "awning",
}
object_counts: dict[str, int] = {}
for value in selected["object_counts"].fillna("{}").tolist():
    for label, count in json.loads(value).items():
        if label in allowed_objects:
            object_counts[label] = object_counts.get(label, 0) + int(count)

infrastructure_path = OUTPUT / "infrastructure_detections.csv"
if infrastructure_path.exists():
    infrastructure_for_video = pd.read_csv(infrastructure_path)
    infrastructure_for_video = infrastructure_for_video[infrastructure_for_video["video"] == selected_summary["video"]]
    for label, count in infrastructure_for_video["detected_class"].value_counts().items():
        if label in allowed_objects:
            object_counts[label] = object_counts.get(label, 0) + int(count)

if object_counts:
    object_table = pd.DataFrame(
        sorted(object_counts.items(), key=lambda item: item[1], reverse=True),
        columns=["detected_class", "sampled_detections"],
    )
    st.write("Vehicles, shops, and stalls detected in sampled frames")
    st.dataframe(object_table, use_container_width=True, hide_index=True)
else:
    st.info("No vehicle, shop, or stall detections are available for this video yet. Run Scan shops and stalls for infrastructure candidates.")

detail_fig = px.line(
    selected.sort_values("timestamp_seconds"),
    x="timestamp_seconds",
    y=["people_count", "vehicle_count"],
    labels={"timestamp_seconds": "Seconds from video start", "value": "Count", "variable": "Metric"},
)
st.plotly_chart(detail_fig, use_container_width=True)

st.subheader("Demographic data")
st.info(
    "Men/women and broad age groups are aggregate observation fields entered by interval. "
    "They are never linked to track IDs or individual records. Confidence and coverage show how complete each observation is. "
    "The pipeline does not identify the same person across different days."
)

st.subheader("People and vehicle trends")
trend = observations.sort_values(["day", "timestamp_seconds"])
fig = px.line(trend, x="timestamp_seconds", y="people_count", color="day",
              labels={"timestamp_seconds": "Seconds from video start", "people_count": "Visible people"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Vehicle trends")
vehicle_fig = px.line(trend, x="timestamp_seconds", y="vehicle_count", color="day",
                     labels={"timestamp_seconds": "Seconds from video start", "vehicle_count": "Visible vehicles"})
st.plotly_chart(vehicle_fig, use_container_width=True)

st.warning("Anonymous track totals are per-video estimates and must not be interpreted as recurring identities across days.")

st.subheader("Download data")
csv_path = OUTPUT / "observations.csv"
json_path = OUTPUT / "daily_summary.json"
download_cols = st.columns(2)
download_cols[0].download_button(
    "Download observations CSV", csv_path.read_bytes(), csv_path.name, "text/csv"
)
if json_path.exists():
    download_cols[1].download_button(
        "Download summary JSON", json_path.read_bytes(), json_path.name, "application/json"
    )

with st.expander("How to read the dashboard"):
    st.markdown(
        "**Sample every (seconds)** controls time resolution. A setting of 2 means the model examines one frame every 2 seconds; it does not shorten the video. Lower values provide a smoother trend but take longer.\n\n"
        "**People count** is the number visible in each sampled frame. **Peak** is the largest sampled count. **Anonymous tracks** are temporary IDs within one video and can include missed detections or ID switches.\n\n"
        "The generic model detects common vehicle classes. Shops, tea stalls, and shelters need a local custom detector before they can be reported reliably."
    )
