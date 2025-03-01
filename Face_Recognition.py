import cv2
import os
import pyttsx3
from dataSetGenerator import dataSetGenerator

engine = pyttsx3.init()

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadePath)

while True:
    try:
        recognizer.read("trainer/trainer.yml")
        break
    except:
        print("No trained model found. Please train the model first.")
        name = input("Enter your name: ")
        dataSetGenerator(name)

# Load ID and names from the text file
id_file = os.path.join(path, "id_data.txt")
if not os.path.exists(id_file):
    print("ID file not found. Creating a new one.")
    open(id_file, "w").close()

with open(id_file, "r") as file:
    names = {
        int(line.split(",")[0]): line.split(",")[1].strip() for line in file.readlines()
    }

# Check if the user wants to add a new face
check = input("Do you want to add a new face? (y/n): ")
if check.lower() == "y":
    name = input("Enter your name: ")
    dataSetGenerator(name)

# Main face recognition loop
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
i = 0

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for x, y, w, h in faces:
        nbr_predicted, conf = recognizer.predict(frame[y : y + h, x : x + w])

        cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)

        # Retrieve name or mark as unknown
        nbr_name = names.get(nbr_predicted, "Unknown")
        if nbr_name != "Unknown":
            engine.say(f"{nbr_name}")
            engine.runAndWait()

        # Display name and confidence on the frame
        cv2.putText(
            frame,
            f"{nbr_name} -- {conf:.2f}",
            (x, y + h),
            font,
            1.1,
            (0, 255, 0),
            2,
        )

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
