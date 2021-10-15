## AnnoQual
A graphical user interface intended for time-locked video frame annotations and facial expression prediction quality control. 

Requirements:
- tkinter
- py-feat
- opencv-python

## Features
- Video player
- Custom annotations 
- Button to time-lock annotations to video frames

## Automation
The user may automate facial action unit coding by selecting "automate action unit data". This will utilize OpenFace's multiple face classification system to assess action unit intensities per frame of video. Other data includes head pose estimation, eye gaze estimation, facial landmarks, and confidence scores. You must clone the OpenFace github to utlize this function. Change file path for "FacialLandmarkVidMulti.exe" to the correct file path.

## Training/testing
The user can combine the annotation and automation data to train a machine learning algorithim (support vector machine) on binary pain classificaiton. Testing metrics via graphs will be returned to the user after training is finished. 

## Prediction
The user can use a binary pain classification model on novel data.


