import cv2 as cv
import numpy as np

def train_model(features,labels):
    model = cv.face.LBPHFaceRecognizer_create()

    labels_np = np.array(labels)
    model.train(features,labels_np)

    model.save("trained_model.yml")
