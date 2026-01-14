# Accident

intelligent surveillance systems capable of real-time analysis and automated event detection. This project leverages such advancements to develop a Crash Detection and Alerting System using YOLOv4 (You Only Look Once) a state-of-the-art object detection algorithm and OpenCV, 
a popular computer vision library in Python. The goal is to build an application that can automatically analyze video footage, detect vehicles,
and identify potential crash scenarios based on the interaction of detected objects and their confidence scores.

The system works by analyzing video frames and using YOLOv4 to detect objects such as cars, buses, and trucks. When a high-confidence detection is observed—particularly if objects overlap in a manner indicative of a crash—the system triggers alerts through both visual cues (bounding boxes and warning messages) and an audible beep using the winsound module. 
These alerts provide immediate feedback and can help surveillance teams or emergency response units react more swiftly.

This solution is especially valuable for highway authorities, traffic control rooms, and public safety departments, where timely detection and reporting of accidents can save lives and minimize road blockages.
While the current system processes recorded video, it is designed to be scalable and adaptable for integration with real-time CCTV feeds and additional smart city components.

# Tools and libraries
•	OpenCV (cv2)
For video capture, frame processing, and rendering detection overlays.
•	NumPy (numpy)
Used for numerical operations and generating color arrays for class labeling.
•	winsound 
For generating audio alerts when a crash is detected.
•	YOLOv4 Pre-trained Model Files
o	yolov4.weights: Pre-trained weights file.
o	yolov4.cfg: Configuration file for the YOLOv4 model.
o	coco.names: Class labels for object detection (includes vehicle types like cars, trucks, buses, etc.).
