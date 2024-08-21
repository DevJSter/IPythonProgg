import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Open the webcam
cap = cv2.VideoCapture(0)

# List of known recyclable items (simplified for this example)
recyclable_items = ["bottle", "can", "paper", "cardboard"]

# Map the indices from the MobileNet SSD model to human-readable labels
# This is a simplified example, ensure your mapping is correct
CLASSES = ["background", "bottle", "can", "car", "cat", "dog", "paper", "cardboard"]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only consider detections with confidence > 0.2
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
            
            print(f"Detected object: {label} with confidence: {confidence:.2f}")  # Debugging statement

            # Check if the object is recyclable
            if label in recyclable_items:
                recycle_label = "Recyclable"
                color = (0, 255, 0)  # Green for recyclable
            else:
                recycle_label = "Non-recyclable"
                color = (0, 0, 255)  # Red for non-recyclable

            # Draw bounding box and label
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label}: {recycle_label}", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Webcam - Recyclable Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
