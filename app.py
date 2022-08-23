import streamlit as st
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)

import numpy as np
import mediapipe as mp
import math

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

## Build and Load Model
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

@st.cache(allow_output_mutation=True)
def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    """
    Function used to build the deep neural network model on startup

    Args:
        HIDDEN_UNITS (int, optional): Number of hidden units for each neural network hidden layer. Defaults to 256.
        sequence_length (int, optional): Input sequence length (i.e., number of frames). Defaults to 30.
        num_input_values (_type_, optional): Input size of the neural network model. Defaults to 33*4 (i.e., number of keypoints x number of metrics).
        num_classes (int, optional): Number of classification categories (i.e., model output size). Defaults to 3.

    Returns:
        keras model: neural network with pre-trained weights
    """
    # Input
    inputs = Input(shape=(sequence_length, num_input_values))
    # Bi-LSTM
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    # Attention
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    # Fully Connected Layer
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    # Output
    x = Dense(num_classes, activation='softmax')(x)
    # Bring it all together
    model = Model(inputs=[inputs], outputs=x)

    ## Load Model Weights
    load_dir = "./models/LSTM_Attention.h5"  
    model.load_weights(load_dir)
    
    return model

HIDDEN_UNITS = 256
model = build_model(HIDDEN_UNITS)

## App
st.write("# AI Personal Fitness Trainer Web App")

st.markdown("❗❗ **Development Note** ❗❗")
st.markdown("Currently, the exercise recognition model uses the the x, y, and z coordinates of each anatomical landmark from the MediaPipe Pose model. These coordinates are normalized with respect to the image frame (e.g., the top left corner represents (x=0,y=0) and the bottom right corner represents(x=1,y=1)).")
st.markdown("I'm currently developing and testing two new feature engineering strategies:")
st.markdown("- Normalizing coordinates by the detected bounding box of the user")
st.markdown("- Using joint angles rather than keypoint coordaintes as features")            
st.write("Stay Tuned!")

st.write("## Settings")
threshold1 = st.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)

st.write("## Activate the AI 🤖🏋️‍♂️")

## Mediapipe
mp_pose = mp.solutions.pose # Pre-trained pose estimation model from Google Mediapipe
mp_drawing = mp.solutions.drawing_utils # Supported Mediapipe visualization tools
pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2) # mediapipe pose model

## Real Time Machine Learning and Computer Vision Processes
class VideoProcessor:
    def __init__(self):
        # Parameters
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = threshold3
        
        # Detection variables
        self.sequence = []
        self.current_action = ''

        # Rep counter logic variables
        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None
    
    @st.cache()    
    def draw_landmarks(self, image, results):
        """
        This function draws keypoints and landmarks detected by the human pose estimation model
        
        """
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
        return
    
    @st.cache()
    def extract_keypoints(self, results):
        """
        Processes and organizes the keypoints detected from the pose estimation model 
        to be used as inputs for the exercise decoder models
        
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return pose
    
    @st.cache()
    def calculate_angle(self, a,b,c):
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
    
    @st.cache()
    def get_coordinates(self, landmarks, mp_pose, side, joint):
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
    
    @st.cache()
    def viz_joint_angle(self, image, angle, joint):
        """
        Displays the joint angle value near the joint within the image frame
        
        """
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
        return
    
    @st.cache()
    def count_reps(self, image, landmarks, mp_pose):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        
        """
        
        if self.current_action == 'curl':
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            
            # calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # curl counter logic
            if angle < 30:
                self.curl_stage = "up" 
            if angle > 140 and self.curl_stage =='up':
                self.curl_stage="down"  
                self.curl_counter +=1
            self.press_stage = None
            self.squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, angle, elbow)
            
        elif self.current_action == 'press':           
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')

            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Compute distances between joints
            shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder,wrist))
            
            # Press counter logic
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage =='up'):
                self.press_stage='down'
                self.press_counter += 1
            self.curl_stage = None
            self.squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, elbow_angle, elbow)
            
        elif self.current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            left_hip = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
            left_knee = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
            left_ankle = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            # right side
            right_shoulder = self.get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
            right_hip = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
            right_knee = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
            right_ankle = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Squat counter logic
            thr = 165
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
                self.squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (self.squat_stage =='down'):
                self.squat_stage='up'
                self.squat_counter += 1
            self.curl_stage = None
            self.press_stage = None
                
            # Viz joint angles
            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)
            
        else:
            pass
        return
    
    @st.cache()
    def prob_viz(self, res, input_frame):
        """
        This function displays the model prediction probability distribution over the set of exercise classes
        as a horizontal bar graph
        
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
    
    @st.cache()
    def process(self, image):
        """
        Function to process the video frame from the user's webcam and run the fitness trainer AI

        Args:
            image (numpy array): input image from the webcam

        Returns:
            numpy array: processed image with keypoint detection and fitness activity classification visualized
        """
        # Pose detection model
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.draw_landmarks(image, results) 
        
        # Prediction logic
        keypoints = self.extract_keypoints(results)        
        self.sequence.append(keypoints.astype('float32',casting='same_kind'))      
        self.sequence = self.sequence[-self.sequence_length:]
        
        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            # interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(self.sequence, axis=0))
            # interpreter.invoke()
            # res = interpreter.get_tensor(self.output_details[0]['index'])
            
            self.current_action = self.actions[np.argmax(res)]
            confidence = np.max(res)
            
            # Erase current action variable if no probability is above threshold
            if confidence < self.threshold:
                self.current_action = ''

            # Viz probabilities
            image = self.prob_viz(res, image)
            
            # Count reps
            try:
                landmarks = results.pose_landmarks.landmark
                self.count_reps(
                    image, landmarks, mp_pose)
            except:
                pass

            # Display graphical information
            cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
            cv2.putText(image, 'curl ' + str(self.curl_counter), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'press ' + str(self.press_counter), (240,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'squat ' + str(self.squat_counter), (490,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          
        # return cv2.flip(image, 1)
        return image
    
    def recv(self, frame):
        """
        Receive and process video stream from webcam

        Args:
            frame: current video frame

        Returns:
            av.VideoFrame: processed video frame
        """
        img = frame.to_ndarray(format="bgr24")
        img = self.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        
## Stream Webcam Video and Run Model
# Options
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# Streamer
webrtc_ctx = webrtc_streamer(
    key="AI trainer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)