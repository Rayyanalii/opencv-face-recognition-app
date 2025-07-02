# OpenCV Face Recognition Web App

A simple, interactive facial recognition system built with OpenCV and Streamlit.

## Features

- Upload face images with label
- Train OpenCV’s LBPH face recognizer
- Test predictions on new images
- See confidence scores
- Streamlit-based frontend UI
- Image preprocessing and face extraction

## Tech Stack

**Client:** Streamlit  
**Server:** Python, OpenCV, NumPy, PIL

## Installation

Clone the project and install dependencies:

    pip install -r requirements.txt

Run the Streamlit app:

    streamlit run app.py

## Screenshots

To be added later.

## Notes

- If the confidence score is greater than 70, it means a good match wasn't found.
- It's better for the model to recieve multiple images of a subject to perform better.
- If "faces found" is less than "total images", it usually means the image was unclear or the face wasn't detected.
- Uploaded images are resized for consistent detection.

## Authors

- [@Rayyanalii](https://github.com/Rayyanalii)
