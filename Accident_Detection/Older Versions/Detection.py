import cv2
import numpy as np

# Function to detect accidents in the video using YOLO object detection
def detect_accidents(input_video_path, output_video_path):
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

    # Define colors for different classes
    class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Dictionary to store previous positions of bounding boxes
    prev_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blob from the frame and perform a forward pass through the network
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Initialize lists for bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process each output layer
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter out weak detections by ensuring the confidence is greater than a threshold
                if confidence > 0.5:
                    # Scale the bounding box coordinates to the frame size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Add bounding box coordinates, confidences, and class IDs to their respective lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maximum suppression to eliminate overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Inside the main detection loop
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)  # Default color is green

            # Change color to red if confidence score is above 0.9 (90%)
            if confidences[i] > 0.9:
                color = (0, 0, 255)  # Change color to red

            # Draw bounding box with the determined color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame into the output video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    input_video_path = r"C:\Users\faraz\OneDrive\Desktop\15552.mp4"  # Path to input video
    output_video_path = "C:\\Users\\faraz\\OneDrive\\Desktop\\output_video.mp4"  # Path to save output video
    detect_accidents(input_video_path, output_video_path)
