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

# %% 
@st.cache
def get_mp_tools(mp):
    # Pre-trained pose estimation model from Google Mediapipe
    mp_pose = mp.solutions.pose
    # Supported Mediapipe visualization tools
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, mp_drawing
mp_pose, mp_drawing = get_mp_tools(mp)

## Parameters
actions = np.array(['curl', 'press', 'squat'])
sequence_length = 30
num_input_values = 33*4

## Variables
current_action = ''
curl_counter = 0
press_counter = 0
squat_counter = 0
curl_stage = None
press_stage = None
squat_stage = None

## Build Model
def attention_block(inputs, time_steps):
    """
    Attention layer for deep neural network
    
    """
    # Attention weights
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    
    # Attention vector
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Luong's multiplicative score
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul') 
    
    return output_attention_mul

# @st.cache(allow_output_mutation=True)
# def build_model(HIDDEN_UNITS):
#     # Input
#     inputs = Input(shape=(sequence_length, num_input_values))
#     # Bi-LSTM
#     lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
#     # Attention
#     attention_mul = attention_block(lstm_out, sequence_length)
#     attention_mul = Flatten()(attention_mul)
#     # Fully Connected Layer
#     x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
#     x = Dropout(0.5)(x)
#     # Output
#     x = Dense(actions.shape[0], activation='softmax')(x)

#     # Bring it all together
#     model = Model(inputs=[inputs], outputs=x)

#     ## Load Model
#     load_dir = "./models/LSTM_Attention.h5"
#     model.load_weights(load_dir)
#     return model
# HIDDEN_UNITS = 256
# model = build_model(HIDDEN_UNITS)

@st.cache(allow_output_mutation=True)
def build_lstm(sequence_length, num_input_values):
    lstm = Sequential()
    lstm.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(sequence_length, num_input_values)))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(128, return_sequences=False, activation='relu'))
    lstm.add(Dense(128, activation='relu'))
    lstm.add(Dense(64, activation='relu'))
    lstm.add(Dense(actions.shape[0], activation='softmax'))
    return lstm
model = build_lstm(sequence_length, num_input_values)

## App
st.write("# AI Personal Trainer Web App")
st.write("❗❗ Disclaimer: I'm currently working to engineer some better features for the exercise recognition model. Currently, the model uses the the x, y, and z coordinates of each anatomical landmark from the MediaPipe pose model. I'm developing and testing models that use normalized coordinates and/or joint angles as features.")
st.write("Stay Tuned!")
st.write("## Settings")

threshold1 = st.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)

st.write("## Activate the AI 🤖🏋️‍♂️")

def mediapipe_detection(image, model):
    """
    This function detects human pose estimation keypoints from webcam footage
    
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    """
    This function draws keypoints and landmarks detected by the human pose estimation model
    
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )
    
def extract_keypoints(results):
    """
    Processes and organizes the keypoints detected from the pose estimation model 
    to be used as inputs for the exercise decoder models
    
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

def calculate_angle(a, b, c):
    """
    Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another
    
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

def get_coordinates(landmarks, mp_pose, side, joint):
    """
    Retrieves x and y coordinates of a particular keypoint from the pose estimation model
        
    Args:
        landmarks: processed keypoints from the pose estimation model
        mp_pose: Mediapipe pose estimation model
        side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
        joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.
    
    """
    coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
    x_coord_val = landmarks[coord.value].x
    y_coord_val = landmarks[coord.value].y
    return [x_coord_val, y_coord_val]  

def viz_joint_angle(image, angle, joint):
    """
    Displays the joint angle value near the joint within the image frame
    
    """
    cv2.putText(image, str(int(angle)), 
                tuple(np.multiply(joint, [640, 480]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
    return

def count_reps(image, current_action, landmarks, mp_pose):
    """
    Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
    
    """
    
    global curl_counter, press_counter, squat_counter, curl_stage, press_stage, squat_stage

    if current_action == 'curl':
        # Get coords
        shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
        wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')
        
        # calculate elbow angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # curl counter logic
        if angle < 30:
            curl_stage = "up" 
        if angle > 140 and curl_stage =='up':
            curl_stage="down"  
            curl_counter +=1
        press_stage = None
        squat_stage = None
            
        # Viz joint angle
        viz_joint_angle(image, angle, elbow)
        
    elif current_action == 'press':
        
        # Get coords
        shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
        wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        # Compute distances between joints
        shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
        shoulder2wrist_dist = abs(math.dist(shoulder,wrist))
        
        # Press counter logic
        if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
            press_stage = "up"
        if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (press_stage =='up'):
            press_stage='down'
            press_counter += 1
        curl_stage = None
        squat_stage = None
            
        # Viz joint angle
        viz_joint_angle(image, elbow_angle, elbow)
        
    elif current_action == 'squat':
        # Get coords
        # left side
        left_shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        left_hip = get_coordinates(landmarks, mp_pose, 'left', 'hip')
        left_knee = get_coordinates(landmarks, mp_pose, 'left', 'knee')
        left_ankle = get_coordinates(landmarks, mp_pose, 'left', 'ankle')
        # right side
        right_shoulder = get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
        right_hip = get_coordinates(landmarks, mp_pose, 'right', 'hip')
        right_knee = get_coordinates(landmarks, mp_pose, 'right', 'knee')
        right_ankle = get_coordinates(landmarks, mp_pose, 'right', 'ankle')
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate hip angles
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Squat counter logic
        thr = 165
        if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
            squat_stage = "down"
        if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage =='down'):
            squat_stage='up'
            squat_counter += 1
        curl_stage = None
        press_stage = None
            
        # Viz joint angles
        viz_joint_angle(image, left_knee_angle, left_knee)
        viz_joint_angle(image, left_hip_angle, left_hip)
        
    else:
        pass
    
def prob_viz(res, actions, input_frame, colors):
    """
    This function displays the model prediction probability distribution over the set of exercise classes
    as a horizontal bar graph
    
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):        
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

#%%
# st.write("Click on the checkbox to use the exercise recognition and rep counter AI!")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# cap = cv2.VideoCapture(0)
# while run:
#     # Set mediapipe model 
#     with mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2) as pose:
#         while cap.isOpened():

#             # Read feed
#             ret, frame = cap.read()

#             # Make detection
#             image, results = mediapipe_detection(frame, pose)
            
#             # Draw landmarks
#             draw_landmarks(image, results)
            
#             # 2. Prediction logic
#             keypoints = extract_keypoints(results)        
#             sequence.append(keypoints)      
#             sequence = sequence[-sequence_length:]
                
#             if len(sequence) == sequence_length:
#                 res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]           
#                 predictions.append(np.argmax(res))
#                 current_action = actions[np.argmax(res)]
#                 confidence = np.max(res)
                
#             #3. Viz logic
#                 # Erase current action variable if no probability is above threshold
#                 if confidence < threshold3:
#                     current_action = ''

#                 # Viz probabilities
#                 image = prob_viz(res, actions, image, colors)
                
#                 # Count reps
#                 try:
#                     landmarks = results.pose_landmarks.landmark
#                     count_reps(
#                         image, current_action, landmarks, mp_pose)
#                 except:
#                     pass

#                 # Display graphical information
#                 cv2.rectangle(image, (0,0), (640, 40), colors[np.argmax(res)], -1)
#                 cv2.putText(image, 'curl ' + str(curl_counter), (3,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(image, 'press ' + str(press_counter), (240,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(image, 'squat ' + str(squat_counter), (490,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
#             # Show to screen
#             # cv2.imshow('OpenCV Feed', image)
#             FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         cap.release()
#         cv2.destroyAllWindows()

#%% 
# class VideoTransformer:
#     def __init__(self):
#         ## Parameters
#         self.actions = np.array(['curl', 'press', 'squat'])
#         self.sequence_length = 30
#         self.num_input_values = 33*4
#         self.colors = [(245,117,16), (117,245,16), (16,117,245)]

#         ## Variables
#         self.sequence = []
#         self.predictions = []
#         self.res = []
#         self.current_action = ''
#         self.curl_counter = 0
#         self.press_counter = 0
#         self.squat_counter = 0
#         self.curl_stage = None
#         self.press_stage = None
#         self.squat_stage = None

    
#     def mediapipe_detection(self, image, model):
#         """
#         This function detects human pose estimation keypoints from webcam footage
        
#         """
#         self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
#         self.image.flags.writeable = False                  # Image is no longer writeable
#         self.results = model.process(self.image)                 # Make prediction
#         self.image.flags.writeable = True                   # Image is now writeable 
#         self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
#         return self.image, self.results

#     def draw_landmarks(self, image, results):
#         """
#         This function draws keypoints and landmarks detected by the human pose estimation model
        
#         """
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                     )
        
#     def extract_keypoints(self, results):
#         """
#         Processes and organizes the keypoints detected from the pose estimation model 
#         to be used as inputs for the exercise decoder models
        
#         """
#         self.pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#         return self.pose

#     def calculate_angle(self, a, b, c):
#         """
#         Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another
        
#         """
#         a = np.array(a) # First
#         b = np.array(b) # Mid
#         c = np.array(c) # End
        
#         radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#         angle = np.abs(radians*180.0/np.pi)
        
#         if angle > 180.0:
#             angle = 360-angle
            
#         return angle 

#     def get_coordinates(self, landmarks, mp_pose, side, joint):
#         """
#         Retrieves x and y coordinates of a particular keypoint from the pose estimation model
            
#         Args:
#             landmarks: processed keypoints from the pose estimation model
#             mp_pose: Mediapipe pose estimation model
#             side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
#             joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.
        
#         """
#         coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
#         x_coord_val = landmarks[coord.value].x
#         y_coord_val = landmarks[coord.value].y
#         return [x_coord_val, y_coord_val]  

#     def viz_joint_angle(self, image, angle, joint):
#         """
#         Displays the joint angle value near the joint within the image frame
        
#         """
#         cv2.putText(image, str(int(angle)), 
#                     tuple(np.multiply(joint, [640, 480]).astype(int)), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                             )
#         return

#     def count_reps(self, image, current_action, landmarks, mp_pose):
#         """
#         Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        
#         """

#         if current_action == 'curl':
#             # Get coords
#             shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
#             elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
#             wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            
#             # calculate elbow angle
#             angle = self.calculate_angle(shoulder, elbow, wrist)
            
#             # curl counter logic
#             if angle < 30:
#                 self.curl_stage = "up" 
#             if angle > 140 and curl_stage =='up':
#                 self.curl_stage="down"  
#                 self.curl_counter +=1
#             self.press_stage = None
#             self.squat_stage = None
                
#             # Viz joint angle
#             self.viz_joint_angle(image, angle, elbow)
            
#         elif current_action == 'press':
            
#             # Get coords
#             shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
#             elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
#             wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')

#             # Calculate elbow angle
#             elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
#             # Compute distances between joints
#             shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
#             shoulder2wrist_dist = abs(math.dist(shoulder,wrist))
            
#             # Press counter logic
#             if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
#                 self.press_stage = "up"
#             if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (press_stage =='up'):
#                 self.press_stage='down'
#                 self.press_counter += 1
#             self.curl_stage = None
#             self.squat_stage = None
                
#             # Viz joint angle
#             self.viz_joint_angle(image, elbow_angle, elbow)
            
#         elif current_action == 'squat':
#             # Get coords
#             # left side
#             left_shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
#             left_hip = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
#             left_knee = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
#             left_ankle = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
#             # right side
#             right_shoulder = self.get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
#             right_hip = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
#             right_knee = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
#             right_ankle = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            
#             # Calculate knee angles
#             left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
#             right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
#             # Calculate hip angles
#             left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
#             right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
#             # Squat counter logic
#             thr = 165
#             if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
#                 self.squat_stage = "down"
#             if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage =='down'):
#                 self.squat_stage='up'
#                 self.squat_counter += 1
#             self.curl_stage = None
#             self.press_stage = None
                
#             # Viz joint angles
#             self.viz_joint_angle(image, left_knee_angle, left_knee)
#             self.viz_joint_angle(image, left_hip_angle, left_hip)
            
#         else:
#             pass
        
#     def prob_viz(self, res, actions, input_frame, colors):
#         """
#         This function displays the model prediction probability distribution over the set of exercise classes
#         as a horizontal bar graph
        
#         """
#         output_frame = input_frame.copy()
#         for num, prob in enumerate(res):        
#             cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#             cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
#         return output_frame

#     def recv(self, frame):
        
#         img = frame.to_ndarray(format="bgr24")
        
#         with mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2) as pose:

#             # Make detection
#             image, results = self.mediapipe_detection(img, pose)
            
#             # Draw landmarks
#             self.draw_landmarks(image, results)
            
#             # 2. Prediction logic
#             keypoints = self.extract_keypoints(results)        
#             self.sequence.append(keypoints)      
#             self.sequence = self.sequence[-self.sequence_length:]
                
#             if len(self.sequence) == self.sequence_length:
#                 res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]           
#                 self.predictions.append(np.argmax(res))
#                 current_action = self.actions[np.argmax(res)]
#                 confidence = np.max(res)
                
#             #3. Viz logic
#                 # Erase current action variable if no probability is above threshold
#                 if confidence < threshold3:
#                     self.current_action = ''

#                 # Viz probabilities
#                 image = self.prob_viz(res, self.actions, image, self.colors)
                
#                 # Count reps
#                 try:
#                     landmarks = results.pose_landmarks.landmark
#                     self.count_reps(
#                         image, self.current_action, landmarks, mp_pose)
#                 except:
#                     pass

#                 # Display graphical information
#                 cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
#                 cv2.putText(image, 'curl ' + str(self.curl_counter), (3,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(image, 'press ' + str(self.press_counter), (240,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(image, 'squat ' + str(self.squat_counter), (490,30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         return av.VideoFrame.from_ndarray(image, format="bgr24")

# ## Set up Streaming
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )
# ctx = webrtc_streamer(
#     key="example", 
#     video_processor_factory=VideoTransformer,
#     rtc_configuration=RTC_CONFIGURATION,
#     video_frame_callback=VideoTransformer,
#     # rtc_configuration={ # Add this line
#     #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     # }, 
#     media_stream_constraints={"video": True, "audio": False}, 
#     async_processing=True
#     )
 
 # %%
 
 
 
# ## Set up Streaming
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )
# ctx = webrtc_streamer(
#     key="example", 
#     video_processor_factory=VideoTransformer,
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration=RTC_CONFIGURATION,
#     video_frame_callback=VideoTransformer,
#     # rtc_configuration={ # Add this line
#     #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     # }, 
#     media_stream_constraints={"video": True, "audio": False}, 
#     async_processing=True
#     )

## Variables
sequence = []
predictions = []
res = []
colors = [(245,117,16), (117,245,16), (16,117,245)]

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    
    global sequence, predictions, res, colors, current_action
    
    img = frame.to_ndarray(format="bgr24")
    
    with mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2) as pose:
        # Make detection
        image, results = mediapipe_detection(img, pose)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)        
        sequence.append(keypoints)      
        sequence = sequence[-sequence_length:]
            
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]           
            predictions.append(np.argmax(res))
            current_action = actions[np.argmax(res)]
            confidence = np.max(res)
            
        #3. Viz logic
            # Erase current action variable if no probability is above threshold
            if confidence < threshold3:
                current_action = ''

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
            # Count reps
            try:
                landmarks = results.pose_landmarks.landmark
                count_reps(
                    image, current_action, landmarks, mp_pose)
            except:
                pass

            # Display graphical information
            cv2.rectangle(image, (0,0), (640, 40), colors[np.argmax(res)], -1)
            cv2.putText(image, 'curl ' + str(curl_counter), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'press ' + str(press_counter), (240,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'squat ' + str(squat_counter), (490,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
    return av.VideoFrame.from_ndarray(image, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )