import cv2
import pickle
from skimage.feature import hog

# 1. Load pretrained model (.pkl)
with open("mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Convert to gray scale and resize to a 28x28 image
#    Extract features with HOG
def hog_features(img):
    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized = cv2.resize(gray, (28, 28))

    # Normalization (0-1)
    normalized = resized.astype("float32") / 255.0

    # Extract features
    hog_features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return hog_features.reshape(1, -1)


# 3. Initialize webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Could not open camera.")
    exit()

print("Press ESC to exit.")

# Array of labels
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error receiving frame.")
        break
    
    # 4. Predict
    features = hog_features(frame)
    prediction = model.predict(features)[0]
    letter = labels[prediction]

    # Draw a rectangle in screen
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 255, 0), 2)

    # Show prediction on screen
    text = f"Pred: {prediction} -> {letter}"
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real time sign recognition", frame)

    # ESC
    key = cv2.waitKey(1)
    if key == 27:
        print("Exit…")
        break

cam.release()
cv2.destroyAllWindows()
