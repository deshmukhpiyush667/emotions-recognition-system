import cv2
from keras.models import model_from_json, Sequential
import numpy as np

# Load model architecture
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json, custom_objects={'Sequential': Sequential})

# Load weights
model.load_weights("facialemotionmodel.h5")

# Haar Cascade
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocess function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {
0:'angry',
1:'disgust',
2:'fear',
3:'happy',
4:'neutral',
5:'sad',
6:'surprise'
}

# IMPORTANT FIX (Windows Camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FIX → detect using GRAY image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        face_img = gray[y:y+h , x:x+w]
        face_img = cv2.resize(face_img,(48,48))

        img = extract_features(face_img)

        pred = model.predict(img, verbose=0)

        prediction_label = labels[pred.argmax()]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(
            frame,
            prediction_label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Emotion Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()