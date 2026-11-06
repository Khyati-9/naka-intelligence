# Naka Intelligence MVP

This is a free, local pipeline for rough people and vehicle counts from short naka videos.

The pipeline also includes an optional OpenCV face-based apparent-gender classifier. It reports Male/Female/Unknown estimates only; these are not verified gender attributes and should not be used for decisions about individuals.

It intentionally does **not** perform face recognition, gender/age inference, or cross-day identity matching. Track IDs are temporary and valid only within one video.

## Setup

```bash
cd "naka intelligence"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put one or more videos in `input_videos/`. Name them by day, for example:

```text
day1.mp4
day2.mp4
day3.mp4
```

## Run from the terminal

```bash
python pipeline.py --input input_videos --output outputs --save-annotated
```

The first run downloads the selected YOLO weights. Outputs include:

- `outputs/observations.csv` — sampled counts and detections
- `outputs/daily_summary.csv` — per-video summary
- `outputs/daily_summary.json` — machine-readable summary
- `outputs/*_annotated.mp4` — optional visual output

## Open the dashboard

```bash
streamlit run dashboard.py
```

Then open the local URL shown by Streamlit. Put videos in `input_videos/`, click **Analyze videos**, and inspect the charts and tables.

## What this MVP estimates

- Visible people at sampled timestamps
- Crowd changes through each video
- Visible vehicle counts
- Common detector classes returned by the selected model
- Temporary anonymous tracks within each video
- Model-estimated male/female/unknown counts when a visible face is available

Tea stalls, shops, shelters, and other local infrastructure are not guaranteed by the generic model. For those, use a custom model or add a later detector trained on representative local examples.
