# Face Recognition System

## Installation

1. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Face Recognition System

1. Run the main face recognition script:
   ```sh
   python Face_Recognition.py
   ```
2. Enter your name to create a dataset. The system will automatically save your face data and recognize you.
3. Press `q` to quit.

## Adding a New Face

If you want to load another face into the system:

1. Run the dataset generator:
   ```sh
   python dataSetGenerator.py
   ```
2. Enter the name for the new face.

## Resetting the System

If you want to start from scratch:

- Delete the images inside the `dataSet` folder.
- Delete the `trainer.yml` file inside the `trainer` folder.
- Clear the contents of `id_data.txt`.

**Note:** Do **not** delete the `dataSet` and `trainer` folders themselves.

## Code Files Overview

### 1. Face_Recognition.py

- Loads the trained model and recognizes faces using OpenCV.
- Prompts the user to add a new face if required.
- Uses `pyttsx3` for text-to-speech to announce recognized names.

### 2. dataSetGenerator.py

- Captures images from the webcam and saves them as a dataset.
- Assigns a unique ID to each person and stores it in `id_data.txt`.
- Calls `trainer.py` to train the model after capturing images.

### 3. trainer.py

- Reads images from the `dataSet` folder.
- Trains the LBPH (Local Binary Patterns Histograms) face recognizer.
- Saves the trained model in `trainer/trainer.yml`.

## Notes

- Ensure your webcam is working properly before running the scripts.
- You need OpenCV installed with `opencv-contrib-python` to use face recognition models.

Enjoy using the Face Recognition System!

