# %% 
import streamlit as st
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Softmax,
                                     Input, Flatten, Bidirectional, Permute, multiply)

import numpy as np
import mediapipe as mp
import math

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import time

# from collections import deque
# import queue
# import threading
# from pathlib import Path
# from typing import List
# import asyncio

## App
st.write("# AI Personal Trainer Web App")
st.write("❗❗ Disclaimer: I'm currently working to engineer some better features for the exercise recognition model. Currently, the model uses the the x, y, and z coordinates of each anatomical landmark from the MediaPipe pose model. I'm developing and testing models that use normalized coordinates and/or joint angles as features.")
st.write("Stay Tuned!")
st.write("## Settings")

threshold1 = st.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)

st.write("## Activate the AI 🤖🏋️‍♂️")

## Mediapipe
mp_pose = mp.solutions.pose # Pre-trained pose estimation model from Google Mediapipe
mp_drawing = mp.solutions.drawing_utils # Supported Mediapipe visualization tools
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache
def draw_landmarks(image, results):
    """
    This function draws keypoints and landmarks detected by the human pose estimation model
    
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )

@st.cache    
def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_landmarks(image, results)   
    return cv2.flip(image, 1)

## Stream From Webcam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
class VideoProcessor:
    def recv(self, frame):        
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # async def recv_queued(self, frames: List[av.AudioFrame]) -> av.AudioFrame:
        
    #     await asyncio.sleep(0.1)

    #     # Return empty frames to be silent.
    #     new_frames = []
    #     for frame in frames:
    #         img = frame.to_ndarray(format="bgr24")
    #         img = process(img)
    #         new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
    #         new_frames.append(new_frame)

    #     return new_frames
        

webrtc_ctx = webrtc_streamer(
    key="AI trainer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)