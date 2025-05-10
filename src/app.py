import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Page configuration
st.set_page_config(page_title="Gesture Detection", layout="wide")

# Add CSS for video container
st.markdown(
    """
    <style>
        .video-container {
            max-width: 500px !important;
            width: 100%;
            margin: 0 auto;
        }
        .video-container img {
            width: 100%;
            height: auto;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize stop button state
if "stop_analysis" not in st.session_state:
    st.session_state.stop_analysis = False

# Constants
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_DURATION = 60  # 1 minute

# Configure page header
st.title("✨ Gesture Detection Analysis")
st.markdown("Upload a video to detect hand gestures (max 50MB, 1 minute)")


# Setup MediaPipe
@st.cache_resource
def load_recognizer():
    try:
        base_options = python.BaseOptions(
            model_asset_path="models/gesture_recognizer.task",
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.GestureRecognizer.create_from_options(options)
    except Exception as e:
        st.error(f"Failed to load gesture recognizer: {e}")
        raise


def process_video(video_file):
    try:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.getvalue())
            video_path = tmp_file.name

        # Process video
        results = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        if duration > MAX_DURATION:
            st.error(f"Video duration exceeds {MAX_DURATION} seconds limit")
            return None

        # Initialize progress bar and placeholder
        progress_bar = st.progress(0)

        # Create container with limited width
        with st.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("⏹️ Stop Analysis", key="stop_button"):
                st.session_state.stop_analysis = True

        recognizer = load_recognizer()

        frame_idx = 0
        while True:
            # Check for stop condition
            if st.session_state.stop_analysis:
                st.warning("Analysis stopped by user")

                # Return results even if stopped
                if results:
                    st.success("Analysis stopped. Results available.")
                else:
                    st.info("No gestures detected before stopping.")

                return results

            success, frame = cap.read()
            if not success:
                break

            # Process every 5th frame for efficiency
            if frame_idx % 5 == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Detect gestures with confidence threshold
                recognition_result = recognizer.recognize(mp_image)

                if (
                    recognition_result.gestures
                    and recognition_result.gestures[0][0].category_name != "None"
                ):
                    gesture = recognition_result.gestures[0][0]
                    if (
                        gesture.score > 0.5
                    ):  # Only include gestures with >50% confidence
                        timestamp = frame_idx / fps
                        results.append(
                            {
                                "time": f"{timestamp:.1f}s",
                                "gesture": gesture.category_name,
                                "confidence": f"{gesture.score:.2f}",
                            }
                        )

                # Update progress and preview
                progress = min(float(frame_idx) / total_frames, 1.0)
                progress_bar.progress(progress)
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            frame_idx += 1

        return results

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

    finally:
        if "cap" in locals():
            cap.release()
        if "video_path" in locals():
            import os

            try:
                os.unlink(video_path)
            except:
                pass


# Main interface
# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov"], help="Maximum file size: 50MB"
)

# Col 1 for video preview
col1, col2 = st.columns([1, 2])

results = None  # Video processing results

with col1:
    if uploaded_file is not None:
        # Reset stop flag before starting new analysis
        st.session_state.stop_analysis = False

        file_size = len(uploaded_file.getvalue())
        st.info(f"File size: {file_size / 1024 / 1024:.1f}MB")

        if file_size > MAX_VIDEO_SIZE:
            st.error("File too large!")
        else:
            if st.button("🎯 Analyze Gestures", type="primary"):
                with st.spinner("Processing video..."):
                    results = process_video(uploaded_file)

with col2:
    # Display results
    if results:
        # Display filtered results in a nice table
        st.markdown("### 📊 Detected Gestures")
        df = pd.DataFrame(results)

        if not df.empty:
            st.dataframe(
                df,
                column_config={
                    "time": st.column_config.TextColumn("Time"),
                    "gesture": st.column_config.TextColumn("Gesture"),
                    "confidence": st.column_config.NumberColumn(
                        "Confidence", format="%.2f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No gestures detected in the video")
