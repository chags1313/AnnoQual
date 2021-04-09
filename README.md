## AnnoAutoPain
A tkinter gui with a manual annotation system for classifying pain vs no pain temporally in video data and a systatic framework/functionality for facial pain machine learning.


##Requirements:
- OpenCv
- PIL
- Pandas
- Matplotlib
- Sklearn
- Shap

## Annotation 
The launch screen of the gui opens to the binary pain annotator. A user may select a video file then begin annotating pain vs no pain by clicking "annotate". The buttons "pain", "no pain", and "na" can be used to classify the behaviors viewed. When annotation is finished, the user may save data to a CSV.

## Automation
The user may automate facial action unit coding by selecting "automate action unit data". This will utilize OpenFace's multiple face classification system to assess action unit intensities per frame of video. Other data includes head pose estimation, eye gaze estimation, facial landmarks, and confidence scores. You must clone the OpenFace github to utlize this function. Change file path for "FacialLandmarkVidMulti.exe" to the correct file path.

## Training/testing
The user can combine the annotation and automation data to train a machine learning algorithim (support vector machine) on binary pain classificaiton. Testing metrics via graphs will be returned to the user after training is finished. 

## Prediction
The user can use a binary pain classification model on novel data.


