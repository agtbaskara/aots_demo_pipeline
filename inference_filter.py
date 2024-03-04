import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load a pretrained YOLOv8n model
model_detection = YOLO('yolov8n-face.pt')

# Load a pretrained AttractiveNet model
model = load_model("attractiveNet_mnv2.h5")

# Open the video file
video_capture = cv2.VideoCapture('person_small.mp4') # change to 0 for webcam

# Get the frame dimensions
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while video_capture.isOpened():
    # Read the next frame from the video
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (540, 960)) # Resize
    if not ret:
        break

    # Run inference on the frame
    results = model_detection(frame)

    # Process results list
    for result in results:
        # Skip the bounding box if none detected
        if len(result.boxes.xyxy.squeeze().tolist()) == 0:
            print("No faces detected.")
            continue

        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.boxes.xyxy.squeeze().tolist())

        # Calculate the bounding box width and height
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Calculate the area of the bounding box
        bbox_area = bbox_width * bbox_height

        # Skip the bounding box if it is too small
        if bbox_area < 2000:
            continue

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Predict the attractiveness of the cropped face
        cropped_face = frame[y1:y2, x1:x2] 
        preprocessed_frame = cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB), (350, 350)) / .255
        score = model.predict(np.expand_dims(preprocessed_frame, axis=0))

        # Draw the score value on the frame
        score_text = "Attractiveness: {:.2f}".format(score[0][0])
        cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with bounding boxes and scores to the output video
    output_video.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
video_capture.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()