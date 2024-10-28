import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8s")

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
# Set resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize Pose object with default parameters
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (Mediapipe expects RGB images)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect the pose landmarks
        results = pose.process(image)

        # Convert the image color back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks on the frame if any are detected
        if results.pose_landmarks:
            # Draw pose landmarks and connections on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Optional: Print the coordinates of each landmark
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(f'Landmark {idx}: ({cx}, {cy})')

        # Show the frame with pose landmarks
        cv2.imshow('Mediapipe Pose Estimation', image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
