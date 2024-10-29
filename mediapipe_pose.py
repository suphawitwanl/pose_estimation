import cv2
import mediapipe as mp

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(frame):
    # Initialize Pose object with default parameters
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
        # cv2.imshow('Mediapipe Pose Estimation', image)

    return image

# cv2.destroyAllWindows()
