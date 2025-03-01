import cv2
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "dataSet")

def load_ids_and_names(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        lines = file.readlines()
    return {int(line.split(",")[0]): line.split(",")[1].strip() for line in lines}


def save_id_and_name(file_path, user_id, name):
    with open(file_path, "a") as file:
        file.write(f"{user_id},{name}\n")


def dataSetGenerator(name="Unknown"):
    path = os.path.dirname(os.path.abspath(__file__))
    id_file = os.path.join(path, "id_data.txt")
    ids_and_names = load_ids_and_names(id_file)
    new_id = max(ids_and_names.keys(), default=0) + 1
    print(f"Your ID is: {new_id}")

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    i = 0
    offset = 50

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in faces:
            i += 1
            cv2.imwrite(
                os.path.join(dataset_dir, f"face-{new_id}.{i}.jpg"),
                gray[y - offset : y + h + offset, x - offset : x + w + offset],
            )
            cv2.rectangle(
                im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2
            )
            cv2.imshow(
                "im", im[y - offset : y + h + offset, x - offset : x + w + offset]
            )
            cv2.waitKey(100)
        if i > 20:
            cam.release()
            cv2.destroyAllWindows()
            break

    from trainer import trainer
    save_id_and_name(id_file, new_id, name.strip())
    trainer()


if __name__ == "__main__":
    name = input("Enter your name: ")
    dataSetGenerator(name)
