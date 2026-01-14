import cv2


def is_collision(box1, box2):
    # Extract coordinates of the two boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the corners of the two boxes
    x1_right = x1 + w1
    y1_bottom = y1 + h1
    x2_right = x2 + w2
    y2_bottom = y2 + h2

    # Check for collision
    if x1 < x2_right and x1_right > x2 and y1 < y2_bottom and y1_bottom > y2:
        return True
    else:
        return False


def draw_red_rectangle(frame, box):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_green_rectangle(frame, box):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def detect_cars_and_accidents(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    car_cascade = cv2.CascadeClassifier(r"C:\Users\chand\PycharmProjects\workspace\Accident_Detection\haarcascade_car.xml")
    if car_cascade.empty():
        print("Error: Haar cascade classifier not loaded.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        bounding_boxes = []
        for (x, y, w, h) in cars:
            bounding_boxes.append((x, y, w, h))

        colliding_boxes = []
        for i, box1 in enumerate(bounding_boxes):
            for j, box2 in enumerate(bounding_boxes):
                if i != j:
                    if is_collision(box1, box2):
                        colliding_boxes.append(i)
                        colliding_boxes.append(j)

        for i, box in enumerate(bounding_boxes):
            if i in colliding_boxes:
                draw_red_rectangle(frame, box)
            else:
                draw_green_rectangle(frame, box)

        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video_path = r"C:\\Users\\chand\\PycharmProjects\\workspace\\Accident_Detection\\Videos\\1558.mp4"  # Path to input video
    output_video_path = "C:\\Users\\chand\\PycharmProjects\\workspace\\Accident_Detection\\output_video.mp4"  # Path to save output video
    detect_cars_and_accidents(input_video_path, output_video_path)
