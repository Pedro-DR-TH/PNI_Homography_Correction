import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

st.title("Barnes Maze — Isometric to Top-Down")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    cap.release()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    if "points" not in st.session_state:
        st.session_state.points = []

    st.write("Click points around the maze edge, in order.")
    coords = streamlit_image_coordinates(pil_img, key="frame_click")

    if coords:
        pt = (coords["x"], coords["y"])
        if pt not in st.session_state.points:
            st.session_state.points.append(pt)

    st.write("Points so far:", st.session_state.points)

    if st.button("Reset points"):
        st.session_state.points = []

    if st.button("Process video") and len(st.session_state.points) >= 4:
        src_pts = np.array(st.session_state.points, dtype=np.float32)

        size = 800
        margin = 50
        center = size / 2
        radius = center - margin
        n_points = len(src_pts)
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False) - np.pi/2
        dst_pts = np.array([
            [center + radius*np.cos(a), center + radius*np.sin(a)]
            for a in angles
        ], dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        warped_check = cv2.warpPerspective(frame, H, (size, size))
        st.image(cv2.cvtColor(warped_check, cv2.COLOR_BGR2RGB), caption="Warp check")

        original_name = uploaded_video.name
        base_name = os.path.splitext(original_name)[0]
        output_filename = f"HA_{base_name}.mp4"

        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, output_filename)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (size, size))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            warped = cv2.warpPerspective(f, H, (size, size))
            out.write(warped)
            i += 1
            progress.progress(min(i / total_frames, 1.0))

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            st.download_button("Download top-down video", f, file_name=output_filename)
