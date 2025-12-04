import cv2
import numpy as np
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
    cv2.imwrite("preprocess.png", resized)

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

print("Press SPACE to capture image. ESC to exit.")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error receiving frame.")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    # SPACE
    if key == 32:
        img_name = "input.png"
        cv2.imwrite(img_name, frame)
        print(f"Image saved as {img_name}")

        # 4. Predict
        features = hog_features(frame)
        prediction = model.predict(features)[0]

        labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        print("\n==============================")
        print(f"   CLASS: {prediction}")
        print(f"   LETTER: {labels[prediction]}")
        print("==============================\n")

        break

    # Salir con ESC
    elif key == 27:
        print("Exit…")
        break

cam.release()
cv2.destroyAllWindows()
