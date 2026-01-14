import cv2


# Function to detect accidents in the video
def detect_accidents(input_video_path, output_video_path):
    # Load the video
    cap = cv2.VideoCapture(input_video_path)

    # Get the video frame rate and resolution
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Initialize the background subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fgmask = subtractor.apply(frame)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate area of contour
            area = cv2.contourArea(contour)

            # If area exceeds a certain threshold, consider it as an accident
            if area > 1000:
                # Draw a red rectangle around the area of the accident
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
    input_video_path = r"C:\\Users\\chand\\PycharmProjects\\workspace\\Accident_Detection\\Videos\\1558.mp4"  # Path to input video
    output_video_path = "C:\\Users\\chand\\PycharmProjects\\workspace\\Accident_Detection\\output_video.mp4"  # Path to save output video
    detect_accidents(input_video_path, output_video_path)
