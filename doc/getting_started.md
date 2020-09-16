# Installation of Dependencies
Project dependencies can be installed using <a href="https://anaconda.org/">Anaconda</a>. New project environment can be created using ```environment.yml``` file. 
Detailed steps involving this process can be found 
<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file">here</a>.
# Project Structure
>OEDR
>>carmaker_integration


>>depth_estimation  
Contains the <a href="../OEDR/depth_estimation/depth_estimation.ipynb">implementation</a> and <a href="../OEDR/depth_estimation/README.md">documentation</a> of Depth Estimation Subsystem.

>>lane_detection  
Contains the <a href="../OEDR/lane_detection/LaneDetection.ipynb">implementation</a> and <a href="../OEDR/lane_detection/README.md">documentation</a> of Lane Detection Subsystem.
>>>test
Contains test images for Traffic Light Classification Subsystem.

>>object_detection/YoloV3  
<a href="../OEDR/object_detection/YoloV3/readme.md">Details</a> regarding You Only Look Once Version 3.

>>traffic_light_classification  
Contains the <a href="../OEDR/traffic_light_classification/traffic_light_detection.ipynb">implementation</a> and <a href="../OEDR/traffic_light_classification/README.md">documentation</a> of Traffic Light Classification Subsystem.
>>>Data  
Contains test images for Traffic Light Classification Subsystem.

>>traffic_sign_classification  
Contains the <a href="../OEDR/traffic_sign_classification/trafficsign.py">implementation</a> and <a href="../OEDR/traffic_sign_classification/README.md">documentation</a> of Traffic Sign Classification Subsystem.

>doc  
>>images  
Contains the architecture images of overall system and its subsystems

>>presentation  
Contains the presentation explaining the OEDR system.
