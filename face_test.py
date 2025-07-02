import cv2 as cv
import numpy as np
from PIL import Image
import io

haarCascade = cv.CascadeClassifier("HaarCascades/frontal_face.xml")

def test_model(image,label_map):
    image_bytes = image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    faces_rect = haarCascade.detectMultiScale(img_np,1.1,5)
    if len(faces_rect)==0:
        return None, None, None, None

    for rectangle in faces_rect:
        (x,y,w,h) = rectangle
        cropped = img_np[y:y+h,x:x+w]
    
    model = cv.face.LBPHFaceRecognizer_create()
    model.read("trained_model.yml")

    label,confidence = model.predict(cropped)

    name = next(key for key, value in label_map.items() if value == label)
    text_x = x
    text_y = y - 10 if y - 10 > 10 else y + h + 25

    cv.rectangle(img_np,(x,y),(x+w,y+h),(0,255,0),2)
    cv.putText(img_np,name,(text_x,text_y),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    return label,confidence,img_np,name
