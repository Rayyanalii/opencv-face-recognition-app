# OpenCV Face Recognition Web App

A web-based facial recognition system built using OpenCV and Streamlit. This app allows users to:

- Upload images of people with custom labels

- Train OpenCV’s built-in face recognition model (LBPH-based)

- Test recognition on newly uploaded images

- View recognition confidence scores

It’s a simple, intuitive way to explore OpenCV’s face recognition capabilities using an interactive UI.

## Tech Stack

    Python
    OpenCV (Face Detection & Recognition)
    Streamlit (Frontend Interface)
    NumPy, PIL.

## How It Works

### Upload Images

Upload frontal face images and assign a unique label for each person.

### Train Model

On the press of a button, the model is trained using the uploaded images and labels.

### Test Images

Upload a new image. If a trained face is detected, the model predicts the label with a confidence score.
