import cv2
import numpy as np
import winsound
import asyncio

# Function to detect collisions between bounding boxes
def detect_collisions(boxes):
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            # Calculate overlap area
            x_overlap = max(0, min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0]))
            y_overlap = max(0, min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1]))
            overlap_area = x_overlap * y_overlap
            # Calculate areas of individual bounding boxes
            area_box1 = box1[2] * box1[3]
            area_box2 = box2[2] * box2[3]
            # Calculate the ratio of overlap to the smallest bounding box area
            overlap_ratio = overlap_area / min(area_box1, area_box2)
            # Consider a collision if the overlap ratio is above a threshold
            if overlap_ratio > 0.5:
                collisions.append((i, j))
    return collisions

# Function to process frames asynchronously
async def process_frame(frame, net, classes, width, height):
    # Resize frame to reduce processing time
    resized_frame = cv2.resize(frame, (width // 2, height // 2))

    # Create a blob from the frame and perform a forward pass through the network
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for bounding boxes of cars
    car_boxes = []

    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by ensuring the confidence is greater than a threshold
            if confidence > 0.99 and classes[class_id] == 'car':
                # Scale the bounding box coordinates to the resized frame size
                center_x = int(detection[0] * (width // 2))
                center_y = int(detection[1] * (height // 2))
                w = int(detection[2] * (width // 2))
                h = int(detection[3] * (height // 2))
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add bounding box coordinates to the list
                car_boxes.append([x, y, w, h])

    # Detect collisions between bounding boxes of cars
    collisions = detect_collisions(car_boxes)

    return collisions, frame

# Function to detect accidents in the video using YOLO object detection
async def detect_accidents(input_video_path, output_video_path):
    # Load YOLO
    net = cv2.dnn.readNet(r"C:\Users\faraz\OneDrive\Desktop\yolov4.weights", r"C:\Users\faraz\OneDrive\Desktop\yolov4.cfg")

    # Load names of classes
    with open("../coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    # Get the video frame rate and resolution
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Buffer to store frames
    frame_buffer = []

    # Variables for frame subsampling
    skip_frames = 5  # Process every 5th frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % skip_frames != 0:
            continue

        # Process frames asynchronously
        tasks = [process_frame(frame, net, classes, width, height)]

        # Wait for tasks to complete
        results = await asyncio.gather(*tasks)

        # Process results
        collisions, frame = results[0]

        # If collisions detected, add frame to frame buffer
        if collisions:
            frame_buffer.append(frame)

        # If collision occurred within the frame buffer, display "Crash Detected" message and play a buzzer sound
        if len(frame_buffer) > 0:
            print("Crash Detected!")
            winsound.Beep(1000, 1000)  # Frequency, Duration (milliseconds)
            for buf_frame in frame_buffer:
                cv2.putText(buf_frame, "Crash Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(buf_frame)

        # Write the frame into the output video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clear frame buffer at the end of each iteration
        frame_buffer = []

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    input_video_path = r"C:\Users\faraz\OneDrive\Desktop\1533.mp4"  # Path to input video
    output_video_path = "C:\\Users\\faraz\\OneDrive\\Desktop\\output_video.mp4"  # Path to save output video
    asyncio.run(detect_accidents(input_video_path, output_video_path))
