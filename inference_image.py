import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load a pretrained YOLOv8n model
model_detection = YOLO('yolov8n-face.pt')

# Load a pretrained AttractiveNet model
model = load_model("attractiveNet_mnv2.h5")

# Read an image using OpenCV
frame = cv2.imread('person.jpg')

# Run inference on the source
results = model_detection(frame)  # list of Result objects

# Process results list
for result in results:
    # Skip the bounding box if none detected
    if len(result.boxes.xyxy.squeeze().tolist()) == 0:
        print("No faces detected.")
        continue

    # Extract the bounding box coordinates
    x1, y1, x2, y2 = map(int, result.boxes.xyxy.squeeze().tolist())

    # Draw the bounding box on the original image
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Predict the attractiveness of the cropped face
    cropped_face = frame[y1:y2, x1:x2] 
    preprocessed_frame = cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB), (350,350)) / .255
    score = model.predict(np.expand_dims(preprocessed_frame, axis=0))

    # Draw the score value on the image
    score_text = "Attractiveness: {:.2f}".format(score[0][0])
    cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(score[0][0])

# Save the final image
cv2.imwrite('result_image.jpg', frame)

# Display the result
cv2.imshow('Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()