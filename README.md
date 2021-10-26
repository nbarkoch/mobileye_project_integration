# mobileye_project_integration
The purpose of this project is to detect traffic lights from a moving camera (also tell if the light is green or red) and estimates the distance from it. 
with the following steps:  
## First step: detect candidates, tells if they are red or green
with Image Processing (including: mask, convolution, non-max suppression), using: numPy, OpenCV.  
## Second step: filter the traffic lights  
with Neural Network Model (which we trained with a dataset we created), using TensorFlow.  
## Third step: estimate distance from each traffic light  
with Structure From Motion (using Ego Motion equation)  

