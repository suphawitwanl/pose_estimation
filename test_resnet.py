import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained ResNet18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 36)  # Adjust for 18 keypoints (x, y)

# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open a connection to the webcam
cap = cv2.VideoCapture(2)  # Change 0 to your camera index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image and preprocess it
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = transform(pil_image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        keypoints = output.reshape(-1, 2).numpy()  # Convert to NumPy array for easy handling

    # Scale the keypoints back to the original frame size
    h, w, _ = frame.shape
    keypoints = (keypoints * [w / 224, h / 224]).astype(int)

    # Draw the keypoints on the frame
    for x, y in keypoints:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circles for each keypoint

    # Display the frame with keypoints overlay
    cv2.imshow("Keypoints", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
