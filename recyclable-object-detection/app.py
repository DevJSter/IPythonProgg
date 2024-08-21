import cv2
import numpy as np
import pickle
import mahotas

# Load the saved model and PCA transformer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('pca.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize the frame to a larger size (e.g., double the size of the original frame)
    height, width = frame.shape[:2]
    new_width = width * 2
    new_height = height * 2
    large_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Split the larger image into 50x50 pieces and make predictions
    x_values = range(0, int(new_width / 50) - 1)
    y_values = range(0, int(new_height / 50))
    w, h = 50, 50
    
    for y in y_values:
        for x in x_values:
            y2 = y * 50
            x2 = x * 50
            crop_img = large_frame[y2:y2 + h, x2:x2 + w]
            
            # Create features
            features = np.array(crop_img).flatten()
            features = pca.transform([features])[0]
            
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            rawimage_haralic = mahotas.features.haralick(gray).mean(axis=0)
            
            image_features = np.concatenate((features, rawimage_haralic), axis=0)
            prediction = model.predict([image_features])
            
            if prediction[0] == 0:
                cv2.circle(large_frame, (x2 + 15, y2 + 15), 30, (0, 255, 0), 3)
    
    # Display the resulting frame in a larger window
    cv2.imshow('Live Detection', large_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
