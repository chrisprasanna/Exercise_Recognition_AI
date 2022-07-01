# :robot::video_camera: Building an AI for Real-Time Exercise Recogntion using Computer Vision & Deep Learning :weight_lifting_man::muscle:
---

## Description


https://user-images.githubusercontent.com/88418264/176807706-960e19dd-4261-46f6-bdc0-cf8a6077cc82.mp4

## Installation
1. Download this repository and move it to your desired working directory
2. Download Anaconda if you haven't already
3. Open the Anaconda Prompt
4. Navigate to your working directory using the cd command
5. Run the following command in the Anaconda prompt:
  ```
  conda env create --name <NAME> --file environment.yml
  ```
  &nbsp;&nbsp;&nbsp;where <NAME> needs to be changed to the name of the conda virtual environment for this project. This environment contains all the package installations and dependencies for this project.
  
6. Run the following command in the Anaconda prompt:
  ```
  conda activate <NAME>
  ```
  &nbsp;&nbsp;&nbsp;This activates the conda environment containing all the required packages and their versions. 
  
7. Open Anaconda Navigator
8. Under the "Applications On" dropdown menu, select the newly created conda environment
9. Install and open Jupyter Notebook. NOTE: once you complete this step and if you're on a Windows device, you can call the insalled version of Jupyter Notebook within the conda environment directly from the start menu.  
10. Navigate to the ExerciseDecoder.ipynb file within the repository

## Features

- Implementation of Google MediaPipe's BlazePose model for real-time human pose estimation
- Computer vision tools (i.e., OpenCV) for color conversion, detecting cameras, detecting camera properties, displaying images, and custom graphics/visualization 
- Inferred 3D joint angle computation according to relative coordaintes of surrounding body landmarks
- Guided training data generation
- Data preprocessing and callback methods for efficient deep neural network training
- Implemented and customizable LSTM and Attenion-Based LSTM models
- Real-time visualization of joint angles, rep counters, and probability distribution of exercise classification predictions

## To-Do

- Add AI features that can detect poor form (e.g., leaning, fast eccentric, knees caving in, poor squat depth, etc.) and offer real-time advice for correction. 
- Translate AI to a portable embedded system that you can take outdoors or at a commercial gym. Components may include a microcontroller (e.g., Raspberry Pi), external USB camera, LED screen, battery, and 3D-printed case. 
- Optimize hyperparameters based on minimizing training time and cross-entropy loss on the validation dataset
- Add more exercise classes
- Add additional models. For instance, even though BlazePose is a type of CNN, there may be benefits to including convolutional layers to the exercise recognition AI

## Credits

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) for the pretrained human pose estimation model
- [Nicholas Renotte](https://github.com/nicknochnack) for tutorials on real-time action detection and pose estimation
- [Philippe Rémy](https://github.com/philipperemy/keras-attention-mechanism) for the attention mechanism implementation for Keras

## License
[MIT](https://github.com/chrisprasanna/Exercise_Recognition_AI/blob/main/LICENSE)

---
