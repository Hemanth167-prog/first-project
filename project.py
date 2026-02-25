#.\.venv\Scripts\Activate.ps1
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ===============================
# STEP 1: Load Dataset
# ===============================
DATASET_PATH = r"C:\Users\paras\OneDrive\Desktop\my_project_work\linking_the_projects_with_html\cv project\data"
CATEGORIES = ["with_mask", "without_mask"]

IMG_SIZE = 100

data = []
labels = []

for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img)
            labels.append(label)

        except:
            print("Error loading image:", img_path)

data = np.array(data) / 255.0
labels = np.array(labels)

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = to_categorical(labels)

# ===============================
# STEP 2: Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ===============================
# STEP 3: Build CNN Model
# ===============================
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

# ===============================
# STEP 4: Compile Model
# ===============================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ===============================
# STEP 5: Train Model
# ===============================
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# ===============================
# STEP 6: Save Model
# ===============================
model.save("mask_detector_model.h5")
print("✅ Model saved as mask_detector_model.h5")

# ===============================
# STEP 7: Plot Accuracy Graph
# ===============================
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()
## Part-2
import cv2
import numpy as np
from keras.models import load_model

# ===============================
# Load Trained Model
# ===============================
model = load_model("mask_detector_model.h5")

# ===============================
# Load Haar Cascade Face Detector
# ===============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===============================
# Start Webcam
# ===============================
cap = cv2.VideoCapture(0)

IMG_SIZE = 100

print("✅ Webcam started... Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Camera not working")
        break

    # Convert frame to grayscale (same as training)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray[y:y+h, x:x+w]

        try:
            # Resize face image
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            # Normalize
            face = face / 255.0

            # Reshape for model input
            face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

            # Predict
            prediction = model.predict(face)

            # Get class index
            class_index = np.argmax(prediction)

            if class_index == 0:
                label = "MASK"
                color = (0, 255, 0)  # Green
            else:
                label = "NO MASK"
                color = (0, 0, 255)  # Red

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Display label
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except:
            print("Error processing face")

    # Show webcam output
    cv2.imshow("Face Mask Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# Release Resources
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Program closed")
