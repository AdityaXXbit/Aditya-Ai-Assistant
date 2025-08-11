def server(input, output, session):
    from shiny import app, ui, reactive, render
    from video_processing import process_video  # This imports your processing function
    import tempfile
    import os
    from pathlib import Path
    import cv2
    import asyncio

    from ultralytics import YOLO

    # Load YOLO model once
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)  # live webcam capture

    # --- Video processing on button press ---
    @reactive.Calc
    def process_result():
        if not input.process():
            return None
        file_info = input.video()
        if not file_info:
            return "Please upload a video first."
        # Save uploaded video to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / file_info.name
            with open(video_path, "wb") as f:
                f.write(file_info.read())
            outdir = Path(tmpdir) / "outputs"
            outdir.mkdir()
            report = process_video(
                str(video_path), str(outdir), use_local_whisper=True, api_key=input.apikey()
            )
            summary_file = report.get("summary_file")
            if summary_file and os.path.exists(summary_file):
                return open(summary_file, encoding="utf-8").read()
            else:
                return "No summary available."

    output.summary = render.text(process_result)

    # Fix: Replace lambda with a proper function for UI output
    @output()
    @render.ui
    def frames():
        # You can customize this to show frame thumbnails or any other UI
        return ui.div()

    # --- Async generator to replace reactive.timer ---
    @reactive.Calc
    async def tick():
        while True:
            await asyncio.sleep(0.5)  # 500 ms interval
            yield

    # --- Live annotated video frame ---
    @output()
    @render.image
    async def annotated_frame():
        async for _ in tick():
            ret, frame = cap.read()
            if not ret:
                return
            results = model(frame)
            annotated = results[0].plot()
            ret2, buffer = cv2.imencode(".jpg", annotated)
            if not ret2:
                return
            yield {"content": buffer.tobytes(), "format": "jpeg"}

    # --- Detected object labels ---
    @output()
    @render.text
    async def detected_labels():
        async for _ in tick():
            ret, frame = cap.read()
            if not ret:
                yield "No frame"
                continue
            results = model(frame)
            classes = results[0].names  # id -> name mapping
            detected = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
            labels = [classes[int(c)] for c in detected]
            yield "Detected objects: " + ", ".join(labels) if labels else "No objects detected"

    @session.on_ended
    def cleanup():
        cap.release()
