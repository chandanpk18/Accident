import cv2
import numpy as np
import winsound
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client

# Function to play sound
def play_sound():
    frequency = 2500  # Set frequency to 2500 Hertz
    duration = 1000  # Set duration to 1000 milliseconds (1 second)
    winsound.Beep(frequency, duration)

# Function to send Email
def send_email(subject, body, to_email):
    from_email = "your_mail@gmail.com"       
    from_password = "xxxx xxxx xxxx xxxx"  # Use App password if 2FA is enabled

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Gmail SMTP server and send mail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to send SMS using Twilio
def send_sms(body, to_phone):
    # Twilio credentials (replace with your Twilio SID, Auth Token, and Twilio phone number)
    account_sid = 'Twilio_ssid' 
    auth_token = 'Twilio_auth'    
    from_phone = '+From_number'

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_phone,
        to=to_phone
    )

    print(f"SMS sent! SID: {message.sid}")

# Function to detect accidents in the video using YOLO object detection
def detect_accidents(input_video_path, output_video_path):
    # Load YOLO
    net = cv2.dnn.readNet(r"C:\\Users\\chand\\OneDrive\\Documents\\PGM\\Projects\\Accident_Detection\\yolov4.weights", 
                          r"C:\\Users\\chand\\OneDrive\\Documents\\PGM\\Projects\\Accident_Detection\\yolov4.cfg")

    # Load names of classes
    with open("C:\\Users\\chand\\OneDrive\\Documents\\PGM\\Projects\\Accident_Detection\\coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Video file not found or unable to open.")
        return  # Exit if video can't be opened

    # Get the video frame rate and resolution
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Variables for crash detection
    crash_detected = False
    crash_frame = None
    email_sent = False  # Flag to ensure email is sent only once
    frame_skip = 5  # Process every 5th frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting...")
            break  # Exit if frame reading fails

        # Increment frame count
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue  # Skip processing this frame

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

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

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

            if confidences[i] > 0.9:
                color = (0, 0, 255)  # Change color to red

                if confidences[i] > 0.99 and not crash_detected:
                    crash_detected = True
                    crash_frame = box  # Store the coordinates of the crash frame

            # Draw bounding box with the determined color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if crash_detected and not email_sent:
            cv2.rectangle(frame, (crash_frame[0], crash_frame[1]), 
                          (crash_frame[0] + crash_frame[2], crash_frame[1] + crash_frame[3]), (0, 0, 255), 2)
            cv2.putText(frame, "CRASH DETECTED!", (crash_frame[0], crash_frame[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 0), 2)

            play_sound()

            # send_email(
            #     subject="Accident Detected!",
            #     body="An accident has been detected! Immediate action required!",
            #     to_email="kkemparaju351@gmail.com"
            # )

            # Send SMS
            send_sms(
                body="Accident detected! Immediate action required!",
                to_phone="+918088609424"  
            )

            # Mark the email and SMS as sent
            email_sent = True
            crash_detected = False
            crash_frame = None

        out.write(frame)
        cv2.imshow('frame', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    input_video_path = r"C:\\Users\\chand\\OneDrive\\Documents\\PGM\\Projects\\Accident_Detection\\Videos\\101.mp4"
    output_video_path = "C:\\Users\\chand\\OneDrive\\Documents\\PGM\\Projects\\Accident_Detection\\output_video.mp4"
    detect_accidents(input_video_path, output_video_path)
