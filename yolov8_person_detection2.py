import cv2
from ultralytics import YOLO
import mediapipe_pose

model = YOLO("yolov8s.pt")


# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(2)
# Set the resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (1280, 720))

    if not ret:
        print("Error: Could not read frame.")
        break

    res = frame.copy()

    # Apply fire detection
    results = model(frame)  # return a list of Results objects
    
    boxes = results[0].boxes
    objs = boxes.cpu().numpy()  # Convert bounding box to numpy
    obj_list = model.names
    # print(obj_list)

    count_person = 0
    if objs.shape[0] != 0:  # Check number of detected objs > 0
        current_frame = True
        for obj in objs:
            detected_obj = obj_list[int(obj.cls[0])]
            if detected_obj == "person":
                x0, y0, x1, y1 = obj.xyxy[0].astype(int)
                res = cv2.rectangle(res, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
                res = cv2.putText(res, f"{detected_obj}{count_person}", (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                person_crop = frame[int(y0):int(y1), int(x0):int(x1)]
                person_crop = mediapipe_pose.detect_pose(person_crop)
                cv2.imshow(f'Person{count_person}', person_crop)
                count_person += 1


    # Display the resulting frame
    cv2.imshow('Camera', res)

    # Press 'q' to exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
